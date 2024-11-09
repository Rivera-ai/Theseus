from typing import Callable, List, Optional, Tuple, Union
import math
from functools import partial
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
from Utils import forward_hook_func, backward_hook_func

approx_gelu = lambda: nn.GELU(approximate="tanh")

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def get_layernorm(hidden_size: torch.Tensor, eps: float, affine: bool, use_kernel: bool):
    if use_kernel:
        try:            
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(hidden_size, elementwise_affine=affine, eps=eps)
        except ImportError:
            raise RuntimeError("FusedLayerNorm not available. Please install apex.")
    else: 
        return nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)

class Conv3DLayer(nn.Module):
    def __init__(self, dim, inner_dim, enable_proj_out, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inner_dim1 = inner_dim
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, dim),
            nn.SiLU(),
            nn.Conv3d(dim, inner_dim1, kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
        )

        inner_dim2 = inner_dim if enable_proj_out else dim
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, inner_dim1),
            nn.SiLU(),
            nn.Conv3d(inner_dim1, inner_dim2, kernel_size=(3,3,3), stride=1, padding=(1,1,1),)
        )

        self.proj_out = None
        if enable_proj_out:
            self.proj_out = nn.Sequential(
                nn.GroupNorm(32, inner_dim2),
                nn.SiLU(),
                nn.Conv3d(inner_dim2, dim, kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
            )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        if self.proj_out is not None:
            x = self.proj_out(x)
        return x

class DiffLayer(nn.Module):
    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        concat_dim = 2 * dim
        self.proj_diff = nn.Conv3d(dim,
                                   dim,
                                   kernel_size=(1, 1, 1),
                                   stride=1,
                                   padding=0)
        self.proj_out = nn.Sequential(
            nn.GroupNorm(32, concat_dim),
            nn.SiLU(),
            nn.Conv3d(concat_dim,
                      dim,
                      kernel_size=(1, 1, 1),
                      stride=1,
                      padding=0),
        )

    def forward(self, x):
        """
        x: [b, c, t, h, w]
        """
        diff = F.pad(torch.diff(x, dim=2), (0, 0, 0, 0, 0, 1), "replicate")
        diff = self.proj_diff(diff)
        x = torch.concat([x, diff], dim=1)
        x = self.proj_out(x)

        return x

class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        enable_flashattn: bool = False,
        enable_mem_eff_attn: bool = False,
        register_hook=False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn = enable_flashattn
        self.enable_mem_eff_attn = enable_mem_eff_attn

        self.qkv = nn.Linear(in_features=dim,
                             out_features=dim * 3,
                             bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

        if register_hook:
            self.qkv.register_forward_hook(
                partial(forward_hook_func, name="qkv"))
            self.qkv.register_full_backward_hook(
                partial(backward_hook_func, name="qkv"))

    def forward(self,
                x: torch.Tensor,
                attn_bias: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        x_dtype = x.dtype

        # qkv project and reshape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        if self.enable_flashattn or self.enable_mem_eff_attn:
            qkv = qkv.permute(2, 0, 1, 3, 4)  #[3, B, N, num_heads, head_dim]
        else:
            qkv = qkv.permute(2, 0, 3, 1, 4)  #[3, B, num_heads, N, head_dim]

        # split q, k, v from the first dimension
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.enable_flashattn:
            from flash_attn import flash_attn_func

            if x_dtype == torch.float32:
                q = q.to(torch.float16)
                k = k.to(torch.float16)
                v = v.to(torch.float16)
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0,
                softmax_scale=self.scale,
            )  # [B, N, num_heads, head_dim]
            if x_dtype == torch.float32:
                x = x.to(x_dtype)
        elif self.enable_mem_eff_attn:
            import xformers.ops as xops

            # [B, N, num_heads, head_dim]
            if attn_bias is not None:
                attn_bias = attn_bias.to(q.dtype)
            x = xops.memory_efficient_attention(q,
                                                k,
                                                v,
                                                p=self.attn_drop.p,
                                                attn_bias=attn_bias)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # only transpose the last two dims
            attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(q.dtype)
            attn = self.attn_drop(attn)
            x = attn @ v  # [B, num_heads, N, head_dim]

        if not self.enable_flashattn:
            x = x.transpose(1, 2)

        # num_heads * head_dim = C
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MultiHeadCrossAttention(nn.Module):

    def __init__(self,
                 d_model,
                 num_heads,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 d_kv=None,
                 register_hook=False,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        if d_kv is None:
            d_kv = d_model
        self.kv_linear = nn.Linear(d_kv, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

        if register_hook:
            self.q_linear.register_forward_hook(
                partial(forward_hook_func, name="q_linear"))
            self.q_linear.register_full_backward_hook(
                partial(backward_hook_func, name="q_linear"))

            self.kv_linear.register_forward_hook(
                partial(forward_hook_func, name="kv_linear"))
            self.kv_linear.register_full_backward_hook(
                partial(backward_hook_func, name="kv_linear"))

            self.proj.register_forward_hook(
                partial(forward_hook_func, name="proj"))
            self.proj.register_full_backward_hook(
                partial(backward_hook_func, name="proj"))

    def forward(self, x, c, mask=None):
        B, N, C = x.shape

        q = self.q_linear(x).reshape(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(c).reshape(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        # default to use memory_efficient_attention
        import xformers.ops as xops
        attn_bias = None
        if mask is not None:
            attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)

        x = xops.memory_efficient_attention(q,
                                            k,
                                            v,
                                            p=self.attn_drop.p,
                                            attn_bias=attn_bias)

        x = x.view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PatchEmbed3D(nn.Module):
    """ #D video to Patch Embedding
    """

    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = 0
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size,
                              bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape

        #TODO(alanxu): padding

        x = self.proj(x)  # BCTHW
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BCN -> BNC
        x = self.norm(x)
        return x


class FinalLayer(nn.Module):

    def __init__(self, hidden_size, patch_size, out_channels, *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.norm_final = nn.LayerNorm(hidden_size,
                                       elementwise_affine=False,
                                       eps=1e-6)
        self.linear = nn.Linear(hidden_size,
                                patch_size * patch_size * out_channels,
                                bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class T2IFinalLayer(nn.Module):

    def __init__(self, hidden_size, patch_size_nd, out_channels, *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.norm_final = nn.LayerNorm(hidden_size,
                                       elementwise_affine=False,
                                       eps=1e-6)
        self.linear = nn.Linear(hidden_size,
                                patch_size_nd * out_channels,
                                bias=True)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size**0.5)

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2,
                                                                         dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.freq_emb_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(self.freq_emb_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) *
                          torch.arange(0, half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        phases = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(phases), torch.sin(phases)], dim=1)
        return emb

    def forward(self, t, dtype):
        t_emb = self.timestep_embedding(t, self.freq_emb_size)
        if t_emb.dtype != dtype:
            t_emb = t_emb.to(dtype)
        t_emb = self.mlp(t_emb)
        return t_emb


class CaptionEmbedder(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_size,
                 uncond_prob,
                 act_layer=nn.GELU(approximate="tanh"),
                 token_num=120):
        super().__init__()
        self.y_proj = Mlp(in_features=in_channels,
                          hidden_features=hidden_size,
                          out_features=hidden_size,
                          act_layer=act_layer,
                          drop=0.0)
        self.register_buffer(
            "y_embedding",
            torch.randn(token_num, in_channels) / in_channels**0.5)

        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding,
                              caption)

        return caption

    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption




def get_2d_sincos_pos_embed(embed_dim,
                            grid_size,
                            cls_token=False,
                            extra_tokens=0,
                            scale=1.0):
    """
    grid_size: int of the grid_height=grid_width or grid_height, grid_width)
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    half = embed_dim // 2
    omega = torch.arange(0, half) / half
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m, d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

