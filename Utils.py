from collections.abc import Iterable
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import numpy as np


def set_grad_checkpoint(model, use_fp32_attention=False, gc_step=1):
    assert isinstance(model, nn.Module)

    def set_attr(module):
        module.grad_checkpointing = True
        module.fp32_attention = use_fp32_attention
        module.grad_checkpointing_step = gc_step

    model.apply(set_attr)

def auto_grad_checkpoint(module, *args, **kwargs):

    if not isinstance(module, Iterable):
        return checkpoint(module, use_reentrant=True, *args, **kwargs)
    gc_step = 1

    return checkpoint_sequential(module, gc_step, use_reentrant=True, *args, *kwargs)

def debugprint(debug: bool= False):
    if debug: 
        def printfunc(*args):
            print(*args)
        return printfunc
    else:
        def nullfunc(*args):
            pass
        return nullfunc

def attn_mask_func(i, j, T, H, W, t_window, s_window, time_first, causal):
    S = H * W
    if time_first:
        s1, t1 = i // T, i % T
        s2, t2 = j // T, j % T
    else: 
        t1, s1 = i // S, i % S
        t2, s2 = j // S, j % S
    
    x1, y1 = s1 // W, s1 % W
    x2, y2 = s2 // W, s2 % W

    t_attn_radius = T if t_window < 1 else (t_window + 1) / 2
    s_attn_radius = max(H, W) if s_window < 1 else (s_window + 1) / 2

    t_cond = (t2 <= t1) if causal else True

    if abs(x1 - x2) < s_attn_radius and abs(y1 - y2) < s_attn_radius and abs(t1 - t2) < t_attn_radius and t_cond:
        return 1
    else: 
        return 0

def get_st_attn_mask(T, H, W, t_window, s_window, time_first=False):
    S = H * W
    N = T * S

    xx, yy = np.meshgrid(np.arange(N, dtype=int), np.arange(N, dtype=int))
    z = np.zeros_like(xx)
    for i in range(N):
        for j in range(N):
            x = xx[0][i]
            y = yy[j][0]
            z[i, j] = attn_mask_func(x, y, T, H, W, t_window, s_window, time_first, causal=True)
    
    return torch.from_numpy(z).to(torch.bool)

def get_attn_bias_from_mask(mask: torch.Tensor):
    """ 
    El valor valido es 1
    """
    not_mask = torch.logical_not(mask)
    bias = torch.zeros(not_mask.shape).masked_fill(not_mask, float("-inf"))
    return bias

def readtexts_from_file(fpath):
    texts = []
    with open(fpath) as f:
        while True:
            line = f.readline()
            if not line:
                break
            texts.append(line)
    
    return texts


def get_mmm(x: torch.Tensor):
    return x.max().cpu(), x.mean().cpu(), x.median().cpu()

def get_abs_mmm(x: torch.Tensor):
    x1 = x.abs()
    return x1.max().cpu(), x1.mean().cpu(), x1.median().cpu()

def forward_hook_func(module, inputs, outputs, name=""):
    print(f"forward hook: {name}")
    print("inputs", len(inputs))
    for inp in inputs:
        print(get_abs_mmm(inp))
    print("outs")
    print(get_abs_mmm(outputs))
    print("-------")


def backward_hook_func(module, grad_input, grad_output, name=""):
    print(f"backward hook: {name}")
    print("inputs", len(grad_input))
    for inp in grad_input:
        print(get_abs_mmm(inp))
    print("outs", len(grad_output))
    for out in grad_output:
        print(get_abs_mmm(out))
    print("-------")