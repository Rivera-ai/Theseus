import os
from glob import glob
from typing import Tuple, Optional
from copy import deepcopy
from collections import OrderedDict
import logging
import argparse
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from Dataset import DatasetFromCSV, GetTransformsVideo, DatasetFromJSON
from VAE import VideoAutoEncoderKL
from T5 import T5Encoder
from CLIP import CLIPEncoder
from Config import Config


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


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
        dtype = dtype_mapping[dtype]
        return dtype
    else:
        raise ValueError


def get_dataset(data_path, num_frames, frame_interval, image_size, root):
    """
    Create dataset based on file extension.
    """
    transform = GetTransformsVideo(image_size)
    
    # Determine file type from extension
    file_ext = os.path.splitext(data_path)[1].lower()
    
    if file_ext == '.csv':
        return DatasetFromCSV(
            data_path,
            num_frames=num_frames,
            frame_interval=frame_interval,
            transform=transform,
            root=root
        )
    elif file_ext == '.json':
        return DatasetFromJSON(
            data_path,
            num_frames=num_frames,
            frame_interval=frame_interval,
            transform=transform,
            root=root
        )
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Please use either .csv or .json files.")


def main():
    # create configs
    cfg = Config()

    # Configuración básica de dispositivo y semilla
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    dtype = to_torch_dtype(cfg.dtype)
    torch.set_default_dtype(dtype)

    # Setup an experiment folder:
    save_dir = os.path.join(cfg.root, cfg.preprocessed_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # write config to json
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(cfg.__dict__, indent=2, sort_keys=False))

    # prepare dataset using the appropriate loader based on file extension
    dataset = get_dataset(
        cfg.data_path,
        num_frames=cfg.num_frames,
        frame_interval=cfg.frame_interval,
        image_size=cfg.image_size,
        root=cfg.root
    )

    dataloader = DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        sampler=SequentialSampler(dataset),
        batch_size=cfg.preprocess_batch_size,
        drop_last=False,
    )

    print(f"Dataset contains {len(dataset):,} videos ({cfg.data_path})")
    print(f"Batch size: {cfg.preprocess_batch_size}")

    # video VAE
    vae = VideoAutoEncoderKL(cfg.vae_pretrained, cfg.subfolder, dtype=dtype)

    # text encoder
    if "t5" in cfg.textenc_pretrained:
        text_encoder_cls = T5Encoder
    else:
        text_encoder_cls = CLIPEncoder
    text_encoder = text_encoder_cls(from_pretrained=cfg.textenc_pretrained,
                                  model_max_length=cfg.model_max_length,
                                  dtype=dtype)

    # move to device
    vae = vae.to(device)
    vae.eval()
    text_encoder = text_encoder.to(device)
    text_encoder.eval()

    num_steps_per_epoch = len(dataloader)

    # encoder loop
    dataloader_iter = iter(dataloader)
    epoch = 0
    
    with tqdm(
            range(num_steps_per_epoch),
            desc=f"Processing videos",
            total=num_steps_per_epoch,
    ) as pbar:

        for step in pbar:
            # step
            batch = next(dataloader_iter)
            x = batch["video"].to(device, dtype)  # [B, C, T, H, W]
            # Use 'text' key for JSON dataset and cfg.text_key for CSV dataset
            y = batch["text"] if "text" in batch else batch[cfg.text_key]
            video_ids = batch["video_id"]

            # video and text encoding
            with torch.no_grad():
                x = vae.encode(x)
                model_args = text_encoder.encode(y)

                # save results to file
                for idx in range(len(video_ids)):
                    vid = video_ids[idx]
                    save_fpath = os.path.join(save_dir, vid + ".pt")
                    if not os.path.exists(save_fpath) or cfg.override_preprocessed:
                        saved_data = {
                            "x": x[idx].cpu(),
                            "y": model_args["y"][idx].cpu(),
                            "mask": model_args["mask"][idx].cpu(),
                            "video_id": vid,
                        }
                        torch.save(saved_data, save_fpath)

    print("Done!")


if __name__ == "__main__":
    main()