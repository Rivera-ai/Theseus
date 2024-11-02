class Config:
    def __init__(self) -> None:
        # Debug
        self.debug = False
        self.text_key = "text"
        self.use_preprocessed_data = False
        # Data processing
        self.num_frames = 16
        self.frame_interval = 4
        self.image_size = (256, 256)
        
        # Model architecture
        self.depth = 12
        self.hidden_size = 768
        self.num_heads = 12
        self.patch_size = (1, 2, 2)
        
        self.use_tpe_initially = True
        self.enable_temporal_attn = True
        # "conv3d" or "temporal_only_attn" or "spatial_temporal_attn"
        self.temporal_layer_type = "temporal_only_attn"

        self.enable_mem_eff_attn = False
        self.enable_flashattn = True
        self.enable_grad_ckpt = True
        
        # Training parameters
        self.noise_schedule = "linear"
        self.token_drop_prob = 0.1
        self.use_ema = True
        
        # Pre-trained models
        self.vae_pretrained = "madebyollin/sdxl-vae-fp16-fix"
        self.subfolder = ""
        self.textenc_pretrained = "stabilityai/stable-diffusion-xl-base-1.0"
        self.model_max_length = 77
        self.text_encoder_output_dim = 768
        
        # Dataset
        self.root = "/teamspace/studios/this_studio/data/Videos"  # Directorio donde están tus videos
        self.data_path = "/teamspace/studios/this_studio/data/dataset.csv"  # Tu archivo CSV
        self.num_workers = 4
        
        # Training settings
        self.dtype = "fp32"  # puedes cambiar a "fp16" si tienes problemas de memoria
        self.seed = 123
        self.outputs = "outputs"
        
        # Training loop
        self.epochs = 30
        self.log_every = 1
        self.ckpt_every = 1
        self.accum_iter = 1
        self.load = None
        
        # Optimization
        self.batch_size = 2  # ajusta según tu GPU
        self.lr = 3e-5
        self.grad_clip = 1.0
        
