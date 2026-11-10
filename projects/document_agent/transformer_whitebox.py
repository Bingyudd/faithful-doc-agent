"""Parameterised tensor checks and memory accounting, not a pretrained model."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def rms_norm(x, weight, eps=1e-6):
    if x.shape[-1] != weight.shape[0] or eps < 0:
        raise ValueError('Invalid RMSNorm configuration')
    work = x.float() if x.dtype in (torch.float16,torch.bfloat16) else x
    return (work * torch.rsqrt(work.square().mean(dim=-1,keepdim=True) + eps)).to(x.dtype) * weight


def rope(x, positions, theta=10000.0):
    """Interleaved rotary pairs for x [B,H,T,D], allowing relative positions."""
    if x.ndim != 4 or x.shape[-1] % 2 or positions.numel() != x.shape[-2] or theta <= 0:
        raise ValueError('Invalid rotary shape or base')
    dim = x.shape[-1]
    frequency = theta ** (-torch.arange(0,dim,2,device=x.device,dtype=x.dtype)/dim)
    angles = positions.to(device=x.device,dtype=x.dtype).reshape(1,1,-1,1) * frequency
    even, odd = x[...,0::2], x[...,1::2]
    rotated = torch.stack((even*angles.cos()-odd*angles.sin(), even*angles.sin()+odd*angles.cos()),dim=-1)
    return rotated.flatten(-2)


def swiglu(x, gate, up, down):
    if gate.shape != up.shape or gate.shape[0] != x.shape[-1] or down.shape != (gate.shape[1],x.shape[-1]):
        raise ValueError('Invalid SwiGLU weight shapes')
    return (F.silu(x @ gate) * (x @ up)) @ down


def gqa_attention(q, k, v, *, causal=True, query_start=None):
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError('Expected Q/K/V [B,H,T,D]')
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1] or q.shape[1] % k.shape[1]:
        raise ValueError('GQA heads must divide query heads and share dimensions')
    groups = q.shape[1] // k.shape[1]
    keys = k.repeat_interleave(groups,dim=1)
    values = v.repeat_interleave(groups,dim=1)
    scores = q @ keys.transpose(-2,-1) / math.sqrt(q.shape[-1])
    if causal:
        start = k.shape[-2]-q.shape[-2] if query_start is None else query_start
        if start < 0:
            raise ValueError('Invalid query/cache position')
        query_positions = torch.arange(q.shape[-2],device=q.device) + start
        key_positions = torch.arange(k.shape[-2],device=q.device)
        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~allowed,float('-inf'))
    return scores.softmax(dim=-1) @ values


def kv_cache_bytes(*, batch, tokens, layers, kv_heads, head_dim, element_bytes=2):
    if any(not isinstance(value,int) or value <= 0 for value in [batch,tokens,layers,kv_heads,head_dim,element_bytes]):
        raise ValueError('Cache dimensions and dtype size must be positive integers')
    return 2 * batch * tokens * layers * kv_heads * head_dim * element_bytes


def training_state_bytes(parameters, *, weight_bytes=2, gradient_bytes=2,
                         moment_bytes=4, master_weight_bytes=4):
    if parameters <= 0 or any(value < 0 for value in [weight_bytes,gradient_bytes,moment_bytes,master_weight_bytes]):
        raise ValueError('Invalid parameter or state sizes')
    parts = {'parameters':parameters*weight_bytes, 'gradients':parameters*gradient_bytes,
             'adam_first_moment':parameters*moment_bytes,'adam_second_moment':parameters*moment_bytes,
             'master_weights':parameters*master_weight_bytes}
    return {**parts,'total':sum(parts.values()),
            'excludes':'activations, temporary workspaces, quantization metadata, allocator and distributed overhead'}
