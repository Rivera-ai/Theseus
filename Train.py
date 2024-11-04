import os
import time
from glob import glob
from typing import Tuple
from copy import deepcopy
from collections import OrderedDict
import logging
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Para usar matplotlib sin GUI

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from einops import rearrange
import torchvision
from torch.amp.autocast_mode import autocast
import cv2
import numpy as np

from Dataset import DatasetFromCSV, PreprocessedDatasetFromCSV, GetTransformsVideo, DatasetFromJSON
from VAE import VideoAutoEncoderKL
from T5 import T5Encoder
from CLIP import CLIPEncoder
from TheseusModel import Theseus
from Diffusion import IDDPM
from Config import Config

def get_model_numel(model: torch.nn.Module) -> Tuple[int, int]:
    num_params = 0
    num_params_trainable = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_params_trainable += p.numel()
    return num_params, num_params_trainable

def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def update_ema(ema_model: torch.nn.Module,
               model: torch.nn.Module,
               decay: float = 0.9999) -> None:
    """
    Step the EMA model towards the current model.
    """
    with torch.no_grad():
        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())

        for name, param in model_params.items():
            if name == "pos_embed":
                continue
            # Asegurarse que requires_grad es False para el modelo EMA
            if param.requires_grad:
                ema_params[name].data.mul_(decay).add_((1 - decay) * param.data)

def create_logger(logging_dir):
    logging.basicConfig(level=logging.INFO,
                        format='[\033[34m%(asctime)s\033[0m] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler(f"{logging_dir}/log.txt")
                        ])
    return logging.getLogger(__name__)

def create_tensorboard_writer(exp_dir):
    tensorboard_dir = f"{exp_dir}/tensorboard"
    os.makedirs(tensorboard_dir, exist_ok=True)
    return SummaryWriter(tensorboard_dir)

def parse_args(training=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="model config file path")
    parser.add_argument("--seed", default=42, type=int, help="generation seed")
    parser.add_argument("--ckpt-path", type=str,
                        help="path to model ckpt")
    parser.add_argument("--batch-size", default=None, type=int,
                        help="batch size")

    if training:
        parser.add_argument("--wandb", default=None, type=bool,
                            help="enable wandb")
        parser.add_argument("--load", default=None, type=str,
                            help="path to continue training")
        parser.add_argument("--data-path", default=None, type=str,
                            help="path to data csv")

    return parser.parse_args()

def to_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        dtype_mapping = {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dtype not in dtype_mapping:
            raise ValueError
        return dtype_mapping[dtype]
    else:
        raise ValueError

def log_tensor_stats(name, tensor_or_dict, logger=None):
    """
    Log statistics about tensors with proper type handling.
    """
    def get_tensor_stats(tensor, key_name=""):
        stats = {
            "Shape": tensor.shape,
            "Type": tensor.dtype,
            "Device": tensor.device,
            "Range": [tensor.min().item(), tensor.max().item()],
            "Has NaN": torch.isnan(tensor).any().item(),
            "Has Inf": torch.isinf(tensor).any().item(),
        }
        
        # Solo calcular media y std para tipos flotantes
        if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
            stats["Mean"] = tensor.mean().item()
            stats["Std"] = tensor.std().item()
        
        # Para tensores binarios o de tipo entero, calcular otros stats relevantes
        else:
            if tensor.dtype in [torch.int32, torch.int64, torch.bool]:
                unique_vals = torch.unique(tensor)
                stats["Unique values"] = unique_vals.tolist()
                stats["Value counts"] = [(val.item(), (tensor == val).sum().item()) 
                                       for val in unique_vals]
        
        return stats

    if logger:
        logger.info(f"\n{'-'*20} {name} {'-'*20}")
        
        if isinstance(tensor_or_dict, dict):
            logger.info("Dictionary containing:")
            for k, v in tensor_or_dict.items():
                logger.info(f"\nKey: {k}")
                if torch.is_tensor(v):
                    stats = get_tensor_stats(v, k)
                    for stat_name, stat_value in stats.items():
                        if stat_name == "Range":
                            logger.info(f"{stat_name}: [{stat_value[0]:.4f}, {stat_value[1]:.4f}]")
                        elif stat_name in ["Mean", "Std"]:
                            logger.info(f"{stat_name}: {stat_value:.4f}")
                        else:
                            logger.info(f"{stat_name}: {stat_value}")
                            
                    # Guardar histograma solo para tensores flotantes
                    if v.dtype in [torch.float16, torch.float32, torch.float64]:
                        debug_dir = "debug_generation"
                        os.makedirs(debug_dir, exist_ok=True)
                        values = v.flatten().cpu().numpy()
                        hist_path = os.path.join(debug_dir, f'hist_{name.lower()}_{k}.png')
                        plt.figure(figsize=(10, 5))
                        plt.hist(values, bins=50)
                        plt.title(f"{name} - {k} Distribution")
                        plt.savefig(hist_path)
                        plt.close()
                else:
                    logger.info(f"Value: {v} (type: {type(v)})")
        
        elif torch.is_tensor(tensor_or_dict):
            stats = get_tensor_stats(tensor_or_dict)
            for stat_name, stat_value in stats.items():
                if stat_name == "Range":
                    logger.info(f"{stat_name}: [{stat_value[0]:.4f}, {stat_value[1]:.4f}]")
                elif stat_name in ["Mean", "Std"]:
                    logger.info(f"{stat_name}: {stat_value:.4f}")
                else:
                    logger.info(f"{stat_name}: {stat_value}")
            
            # Guardar histograma solo para tensores flotantes
            if tensor_or_dict.dtype in [torch.float16, torch.float32, torch.float64]:
                debug_dir = "debug_generation"
                os.makedirs(debug_dir, exist_ok=True)
                values = tensor_or_dict.flatten().cpu().numpy()
                hist_path = os.path.join(debug_dir, f'hist_{name.lower().replace(" ", "_")}.png')
                plt.figure(figsize=(10, 5))
                plt.hist(values, bins=50)
                plt.title(f"{name} Value Distribution")
                plt.savefig(hist_path)
                plt.close()

def generate_sample_video(
    model,
    scheduler,
    vae,
    text_encoder,
    prompt,
    num_frames,
    image_size,
    device,
    dtype,
    num_inference_steps=50,
    logger=None
):
    """Generate a sample video with latent value control."""
    was_training = model.training
    model.eval()
    vae.eval()
    
    try:
        with torch.no_grad():
            # Setup inicial
            latent_size = vae.get_latent_size([num_frames, image_size[0], image_size[1]])
            z_size = [vae.out_channels] + latent_size
            
            if logger:
                logger.info(f"\nLatent size: {z_size}")
            
            # Generar latents
            samples = scheduler.sample(
                model=model,
                text_encoder=text_encoder,
                z_size=z_size,
                prompts=[prompt],
                device=device
            )
            
            # Log de los latents originales
            if logger:
                logger.info("\n=== Original Latents ===")
                logger.info(f"Shape: {samples.shape}")
                logger.info(f"Range: [{samples.min().item():.4f}, {samples.max().item():.4f}]")
                logger.info(f"Mean: {samples.mean().item():.4f}")
                logger.info(f"Std: {samples.std().item():.4f}")
            
            # Normalizar los latents a un rango razonable
            # Los VAEs típicamente esperan latents en el rango [-4, 4]
            max_latent_value = 4.0
            samples = torch.clamp(samples, -max_latent_value, max_latent_value)
            
            # Escalar si la std es muy alta
            current_std = samples.std()
            if current_std > 1.0:
                samples = samples / current_std
                
            # Log de los latents normalizados
            if logger:
                logger.info("\n=== Normalized Latents ===")
                logger.info(f"Range: [{samples.min().item():.4f}, {samples.max().item():.4f}]")
                logger.info(f"Mean: {samples.mean().item():.4f}")
                logger.info(f"Std: {samples.std().item():.4f}")
            
            # Decodificar con mejor manejo de errores
            try:
                with autocast('cuda'):
                    video = vae.decode(samples)
                    
                    # Verificar NaNs inmediatamente
                    if torch.isnan(video).any():
                        if logger:
                            logger.warning("NaNs detected in decoded video. Trying fallback decoding...")
                        
                        # Intento de recuperación
                        samples = samples * 0.5  # Reducir la magnitud
                        video = vae.decode(samples.to(dtype))
                    
                    if torch.isnan(video).any():
                        raise ValueError("VAE produced NaNs even after fallback attempt")
                    
                    if logger:
                        logger.info("\n=== Decoded Video ===")
                        logger.info(f"Shape: {video.shape}")
                        logger.info(f"Range: [{video.min().item():.4f}, {video.max().item():.4f}]")
                        logger.info(f"Mean: {video.mean().item():.4f}")
                        logger.info(f"Std: {video.std().item():.4f}")
                    
                    return video
                    
            except Exception as e:
                if logger:
                    logger.error(f"Error during VAE decoding: {str(e)}")
                raise
            
    except Exception as e:
        if logger:
            logger.error(f"Error during generation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        raise
    finally:
        if was_training:
            model.train()
            
def save_video_frames(video_tensor, save_path, fps=8, logger=None):
    """
    Guarda frames de video manejando correctamente las dimensiones.
    Args:
        video_tensor: torch.Tensor de forma [B, C, T, H, W] o [C, T, H, W]
        save_path: ruta donde guardar el video
        fps: frames por segundo
        logger: logger opcional
    """
    if logger:
        logger.info(f"\n=== Video Tensor Analysis ===")
        logger.info(f"Shape: {video_tensor.shape}")
        logger.info(f"Range: [{video_tensor.min():.4f}, {video_tensor.max():.4f}]")
        logger.info(f"Mean: {video_tensor.mean():.4f}")
        logger.info(f"Std: {video_tensor.std():.4f}")

    # Remover dimensión de batch si existe
    if video_tensor.dim() == 5:
        video_tensor = video_tensor.squeeze(0)  # [B, C, T, H, W] -> [C, T, H, W]

    # Asegurar que estamos en CPU y formato float32
    video_tensor = video_tensor.cpu().float()

    # Normalizar a [0, 1]
    if video_tensor.min() < 0 or video_tensor.max() > 1:
        video_tensor = (video_tensor - video_tensor.min()) / (video_tensor.max() - video_tensor.min())

    # Convertir a uint8 [0, 255]
    video_uint8 = (video_tensor * 255).clamp(0, 255).numpy().astype(np.uint8)

    # Reordenar dimensiones: [C, T, H, W] -> [T, H, W, C]
    video = np.transpose(video_uint8, (1, 2, 3, 0))

    if logger:
        logger.info(f"Processed video shape: {video.shape}")
        logger.info(f"Final range: [{video.min()}, {video.max()}]")

    # Crear directorio para frames
    frames_dir = save_path.rsplit('.', 1)[0] + '_frames'
    os.makedirs(frames_dir, exist_ok=True)

    # Guardar frames individuales
    for i in range(video.shape[0]):
        try:
            frame = video[i]  # [H, W, C]
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
            cv2.imwrite(frame_path, frame_bgr)
            
            if logger:
                logger.info(f"Saved frame {i} - Range: [{frame.min()}, {frame.max()}], "
                          f"Mean: {frame.mean():.2f}")
        except Exception as e:
            if logger:
                logger.error(f"Error saving frame {i}: {str(e)}")

    # Guardar video
    try:
        height, width = video.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        for frame in video:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        if logger:
            logger.info(f"Successfully saved video to {save_path}")
    except Exception as e:
        if logger:
            logger.error(f"Error saving video: {str(e)}")

def save_reference_video(video_tensor, save_path, fps=8):
    """Save a reference video tensor as MP4 file using OpenCV."""
    # If video is in [-1, 1] range, normalize to [0, 1]
    if video_tensor.min() < 0:
        video_tensor = (video_tensor + 1) / 2
    
    # Ensure we're working with the first item if this is a batch
    if video_tensor.dim() == 5:  # [B, C, T, H, W]
        video_tensor = video_tensor[0]  # [C, T, H, W]
    
    # Reorder dimensions: [C, T, H, W] -> [T, H, W, C]
    video = video_tensor.permute(1, 2, 3, 0)
    
    # Move to CPU and convert to numpy
    video = (video * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    
    # OpenCV expects BGR format for saving
    if video.shape[-1] == 3:  # If RGB format
        video = video[..., ::-1]  # Convert RGB to BGR
        
    # Get video dimensions
    T, H, W, C = video.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (W, H), isColor=True)
    
    # Write frames
    try:
        for frame in video:
            out.write(frame)
    finally:
        out.release()
        
    # Save also as a backup in individual frames
    frames_dir = save_path.rsplit('.', 1)[0] + '_frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    for i, frame in enumerate(video):
        frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
        cv2.imwrite(frame_path, frame)

def main():
    # create configs
    cfg = Config()
    sample_prompts = ["Despegue de un avion de pasajero"]
    current_prompt_idx = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    dtype = to_torch_dtype(cfg.dtype)
    torch.set_default_dtype(dtype)

    # Setup experiment folder
    os.makedirs(cfg.outputs, exist_ok=True)
    experiment_index = len(glob(f"{cfg.outputs}/*"))
    experiment_dir = f"{cfg.outputs}/{experiment_index:03d}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Write config
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(cfg.__dict__, indent=2, sort_keys=False))
    
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    writer = create_tensorboard_writer(experiment_dir)

    # Prepare dataset
    if cfg.use_preprocessed_data:
        dataset = PreprocessedDatasetFromCSV(
            cfg.data_path,
            root=cfg.root,
            preprocessed_dir=cfg.preprocessed_dir)
    else:
        dataset = DatasetFromCSV(
            cfg.data_path,
            num_frames=cfg.num_frames,
            frame_interval=cfg.frame_interval,
            transform=GetTransformsVideo(cfg.image_size),
            root=cfg.root)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    logger.info(f"Dataset contains {len(dataset):,} videos ({cfg.data_path})")
    logger.info(f"Batch size: {cfg.batch_size}")

    # Model setup
    vae_out_channels = 4
    text_encoder_output_dim = cfg.text_encoder_output_dim
    input_size = (cfg.num_frames, *cfg.image_size)
    vae_down_factor = [1, 8, 8]
    latent_size = [input_size[i] // vae_down_factor[i] for i in range(3)]

    # Encoders
    vae = None
    text_encoder = None
    
    # Text encoder setup
    if "t5" in cfg.textenc_pretrained:
        text_encoder_cls = T5Encoder
    else:
        text_encoder_cls = CLIPEncoder

    if not cfg.use_preprocessed_data:
        vae = VideoAutoEncoderKL(
            cfg.vae_pretrained,
            cfg.subfolder,
            dtype=torch.float16).to(device)
        vae.eval()
        latent_size = vae.get_latent_size(input_size)
        vae_out_channels = vae.out_channels

        text_encoder = text_encoder_cls(
            from_pretrained=cfg.textenc_pretrained,
            model_max_length=cfg.model_max_length,
            dtype=dtype)
        text_encoder_output_dim = text_encoder.output_dim

    # Theseus model
    model = Theseus(
        input_size=latent_size,
        in_channels=vae_out_channels,
        caption_channels=text_encoder_output_dim,
        model_max_length=cfg.model_max_length,
        depth=cfg.depth,
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_heads,
        patch_size=cfg.patch_size,
        use_tpe_initially=cfg.use_tpe_initially,
        enable_temporal_attn=cfg.enable_temporal_attn,
        temporal_layer_type=cfg.temporal_layer_type,
        enable_mem_eff_attn=cfg.enable_mem_eff_attn,
        enable_flashattn=cfg.enable_flashattn,
        enable_grad_checkpoint=cfg.enable_grad_ckpt,
        debug=cfg.debug,
        class_dropout_prob=cfg.token_drop_prob,
    )

    # Free memory if using preprocessed data
    if cfg.use_preprocessed_data:
        if text_encoder is not None:
            del text_encoder
        if vae is not None:
            del vae

    model_numel, model_numel_trainable = get_model_numel(model)
    print(f"Trainable params: {model_numel_trainable}, Total params: {model_numel}")

    # Create EMA
    if cfg.use_ema:
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)
        ema.eval()

    model = model.to(device, dtype)
    model.train()

    scheduler = IDDPM(
        timestep_respacing="",
        noise_schedule=cfg.noise_schedule,
        learn_sigma=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0)

    # Training loop setup
    num_steps_per_epoch = len(dataloader)
    start_epoch = start_step = log_step = 0
    running_loss = 0.0

    # Resume training if needed
    if cfg.load is not None:
        logger.info(f"Loading checkpoint {cfg.load}")
        checkpoint = torch.load(cfg.load)
        model.load_state_dict(checkpoint['model'])
        if not cfg.load_weights_only:
            opt.load_state_dict(checkpoint['opt'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
        logger.info(f"Loaded checkpoint")
        del checkpoint

    if cfg.use_ema:
        update_ema(ema, model, decay=0)

    scaler = torch.amp.GradScaler()

    # Training loop
    for epoch in range(start_epoch, cfg.epochs):
        logger.info(f"Beginning epoch {epoch}...")

        if not cfg.use_preprocessed_data:
            # Get and save reference video
            first_batch = next(iter(dataloader))
            reference_video = first_batch["video"][0]
            
            # Log reference video stats
            logger.info(f"Reference video stats - Shape: {reference_video.shape}, Min: {reference_video.min().item():.4f}, Max: {reference_video.max().item():.4f}, Mean: {reference_video.mean().item():.4f}")
            
            ref_video_save_path = os.path.join(experiment_dir, f"reference_epoch_{epoch:03d}.mp4")
            save_video_frames(reference_video, ref_video_save_path, logger=logger)
            
            # Generate sample video with debugging
            prompt = sample_prompts[current_prompt_idx]
            current_prompt_idx = (current_prompt_idx + 1) % len(sample_prompts)
            
            # Check model mode
            logger.info(f"Model training mode: {model.training}")
            if cfg.use_ema:
                logger.info(f"EMA model training mode: {ema.training}")
            
            sample_video = generate_sample_video(
                model if not cfg.use_ema else ema,
                scheduler,
                vae,
                text_encoder,
                prompt,
                cfg.num_frames,
                cfg.image_size,
                device,
                dtype,
                logger=logger
            )
            
            generated_video_save_path = os.path.join(experiment_dir, f"generated_epoch_{epoch:03d}.mp4")
            save_video_frames(sample_video, generated_video_save_path, logger=logger)
            
            # Save the prompt and reference video text
            metadata_save_path = os.path.join(experiment_dir, f"epoch_{epoch:03d}_metadata.json")
            metadata = {
                "generation_prompt": prompt,
                "reference_video_text": first_batch["text"][0]  # Get text for first video
            }
            with open(metadata_save_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        with tqdm(range(start_step, num_steps_per_epoch),
                  desc=f"Epoch {epoch}",
                  total=num_steps_per_epoch,
                  initial=start_step) as pbar:

            for step in pbar:
                global_step = epoch * num_steps_per_epoch + step
                batch = next(iter(dataloader))

                if cfg.use_preprocessed_data:
                    x = batch['x'].to(device, dtype)
                    y = batch['y'].to(device, dtype)
                    mask = batch['mask'].to(device)
                    model_args = dict(y=y, mask=mask)
                else:
                    x = batch["video"].to(device, dtype)
                    y = batch["text"]
                    with torch.no_grad():
                        x = vae.encode(x)
                        model_args = text_encoder.encode(y)

                # Diffusion
                t = torch.randint(0, scheduler.num_timesteps, (x.shape[0],), device=device)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss_dict = scheduler.training_losses(model, x, t, model_args)
                    loss = loss_dict["loss"].mean() / cfg.accum_iter
                    loss_terms = {k: v.mean() / cfg.accum_iter 
                                for k, v in loss_dict.items() 
                                if k != "loss"}

                scaler.scale(loss).backward()

                if global_step % cfg.accum_iter == 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()

                if cfg.use_ema:
                    update_ema(ema, model)

                running_loss += loss.item()
                log_step += 1

                # Logging
                if (global_step + 1) % cfg.log_every == 0:
                    avg_loss = running_loss / log_step
                    pbar.set_postfix({
                        "loss": avg_loss,
                        "step": step,
                        "global_step": global_step
                    })
                    running_loss = 0
                    log_step = 0
                    
                    writer.add_scalar("loss", loss.item(), global_step)
                    for term, value in loss_terms.items():
                        writer.add_scalar(term, value.item(), global_step)

                # Save checkpoint
                if cfg.ckpt_every > 0 and (global_step + 1) % cfg.ckpt_every == 0:
                    model_states = ema.state_dict() if cfg.use_ema else model.state_dict()
                    checkpoint = {
                        "model": model_states,
                        "opt": opt.state_dict(),
                        "cfg": cfg,
                        "epoch": epoch,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(
                        f"Saved checkpoint at epoch {epoch} step {step + 1} "
                        f"global_step {global_step + 1} to {checkpoint_path}"
                    )

        start_step = 0

    logger.info("Done!")

if __name__ == "__main__":
    main()