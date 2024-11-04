import numbers
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

def is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))
    
    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())
    
    return True

def toTensor(clip):
    """Convert video clip to normalized float tensor."""
    is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
         raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    
    return clip.float() / 255.0

class ToTensorVideo:
    """Convert uint8 video tensor to float32 and scale to [0, 1]."""
    def __init__(self):
        super().__init__()

    def __call__(self, clip):
        return toTensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__

class UCFCenterCropVideo:
    """Center crop video preserving aspect ratio and value range."""
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size

    def __call__(self, clip):
        """
        Args:
            clip (Tensor): Video clip tensor [C, T, H, W] in range [0, 1]
        Returns:
            Tensor: Cropped video clip in same range as input
        """
        h, w = clip.shape[-2:]
        ch, cw = self.size
        
        if h < ch or w < cw:
            scale = max(ch/h, cw/w)
            new_h, new_w = int(h * scale), int(w * scale)
            clip = F.interpolate(
                clip, 
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            )
            h, w = new_h, new_w
        
        i = (h - ch) // 2
        j = (w - cw) // 2
        
        return clip[..., i:i + ch, j:j + w]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"

class NormalizeVideo:
    """Normalize video using ImageNet statistics."""
    def __init__(self):
        # ImageNet statistics
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1, 1)

    def __call__(self, clip):
        """
        Args:
            clip (Tensor): Video clip tensor [C, T, H, W] in range [0, 1]
        Returns:
            Tensor: Normalized video clip
        """
        if clip.min() < 0 or clip.max() > 1:
            raise ValueError(f"Expected input in [0, 1], got range [{clip.min():.4f}, {clip.max():.4f}]")
        
        # Mover media y std al mismo device que el clip
        self.mean = self.mean.to(clip.device)
        self.std = self.std.to(clip.device)
        
        # Normalizar usando estadísticas de ImageNet
        clip = (clip - self.mean) / self.std
        return clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean.squeeze()}, std={self.std.squeeze()})"

def GetTransformsVideo(resolution=256):
    """Get video transforms pipeline with better value range handling."""
    return transforms.Compose([
        ToTensorVideo(),              # uint8 [0, 255] -> float32 [0, 1]
        UCFCenterCropVideo(resolution),  # Mantiene rango [0, 1]
        NormalizeVideo(),             # [0, 1] -> normalizado con ImageNet
    ])

def debug_transform_pipeline(clip, transform_pipeline, logger=None):
    """Debug transformation pipeline with detailed stats."""
    def log_stats(name, tensor, extra_info=None):
        if logger:
            logger.info(f"\n=== {name} ===")
            logger.info(f"Shape: {tensor.shape}")
            logger.info(f"dtype: {tensor.dtype}")
            logger.info(f"Range: [{tensor.min():.4f}, {tensor.max():.4f}]")
            logger.info(f"Mean: {tensor.mean():.4f}")
            logger.info(f"Std: {tensor.std():.4f}")
            
            if extra_info:
                logger.info(f"Extra info: {extra_info}")
                
            # Verificar valores problemáticos
            logger.info(f"Has NaN: {torch.isnan(tensor).any().item()}")
            logger.info(f"Has Inf: {torch.isinf(tensor).any().item()}")
            
            # Porcentaje de valores en diferentes rangos
            if tensor.dtype in [torch.float32, torch.float64]:
                total_elements = tensor.numel()
                neg_1_to_0 = (tensor < 0).sum().item() / total_elements * 100
                zero_to_1 = ((tensor >= 0) & (tensor <= 1)).sum().item() / total_elements * 100
                above_1 = (tensor > 1).sum().item() / total_elements * 100
                
                logger.info(f"Values < 0: {neg_1_to_0:.2f}%")
                logger.info(f"Values in [0,1]: {zero_to_1:.2f}%")
                logger.info(f"Values > 1: {above_1:.2f}%")

    # Stats originales
    log_stats("Original clip", clip)
    
    transformed_clip = clip
    for t in transform_pipeline.transforms:
        transformed_clip = t(transformed_clip)
        log_stats(f"After {t.__class__.__name__}", transformed_clip, 
                 f"Transform: {t.__repr__()}")
    
    return transformed_clip