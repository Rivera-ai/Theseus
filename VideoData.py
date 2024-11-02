import numbers
import random
import torch
import torch.nn.functional as F

def is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))
    
    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())
    
    return True

def toTensor(clip):
    is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
         raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    
    return clip.float() / 255.0

class ToTensorVideo:
    def __init__(self):
        super().__init__()

    def __call__(self, clip):
        return toTensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__

def Hflip(clip):
    return clip.flip(-1)

class RandomHorizontalFlipVideo:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        if random.random() < self.p:
            clip = Hflip(clip)
        return clip

def ResizeScale(clip, target_size, interpolation_mode):
    assert len(target_size) == 2, f"target size should be tuple (height, width), instead got {target_size}"

    H, W = clip.shape[-2:]
    scale = target_size[0] / H

    return F.interpolate(clip, scale_factor=scale, mode=interpolation_mode, align_corners=False)

def Crop(clip, i, j, h, w):
    if len(clip.size()) != 4:
        raise ValueError("Clip should be a 4D tensor")
    
    return clip[..., i:i + h, j:j + w]

def Resize(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError( f"Target size should be tuple (height, width), instead got {target_size}")
    return F.interpolate(clip, size=target_size, mode=interpolation_mode, align_corners=False)

def ResizeCrop(clip, i, j, h, w, size, interpolation_mode="bilinear"):
    clip = Crop(clip, i, j, h, w)
    clip = Resize(clip, size, interpolation_mode)
    return clip

def CenterCrop(clip, crop_size):
    h, w = clip.shape[-2:]
    ch, cw = crop_size
    if h < ch or w < cw:
        raise ValueError("height and width must be no smaller than crop_size")

    i = int(round(0.5 * (h - ch)))
    j = int(round(0.5 * (w - cw)))

    return Crop(clip, i, j, ch, cw)

class UCFCenterCropVideo:
    def __init__(self, size, interpolation_mode="bilinear",):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        clip_resize = ResizeScale(clip=clip, target_size=self.size, interpolation_mode=self.interpolation_mode)
        clip_center_crop = CenterCrop(clip_resize, self.size)
        return clip_center_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"