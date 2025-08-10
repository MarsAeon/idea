from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn as nn

class FusionAdapter(nn.Module):
    def __init__(self, dim: int = 256, mode: str = 'conv_attn', heads: int = 4):
        super().__init__()
        self.mode = mode
        self.proj_struct = nn.Conv2d(dim, dim, 1)
        self.proj_feat = nn.Conv2d(dim, dim, 1)
        if mode == 'conv_attn':
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        elif mode == 'film':
            self.gamma = nn.Conv2d(dim, dim, 1)
            self.beta = nn.Conv2d(dim, dim, 1)
        else:
            self.attn = None

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> Dict[str, Any]:
        s = self.proj_feat(student_feat)
        t = self.proj_struct(teacher_feat)
        if self.mode == 'conv_attn' and self.attn is not None:
            B, C, H, W = s.shape
            q = s.flatten(2).transpose(1, 2)  # (B, HW, C)
            k = t.flatten(2).transpose(1, 2)
            v = k
            fused, _ = self.attn(q, k, v)
            fused = fused.transpose(1, 2).reshape(B, C, H, W)
        elif self.mode == 'film':
            gamma = self.gamma(t)
            beta = self.beta(t)
            fused = s * (1 + gamma) + beta
        else:
            fused = 0.5 * (s + t)
        return {"fused": fused}
