from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptorHead(nn.Module):
    """简单局部描述子头

    输入: feature (B, C, H, W)
    输出:
      - dense_desc: (B, D, H, W) L2 归一化描述子
      - global_desc: (B, D) 全局池化描述子（便于检索/消融）
    """

    def __init__(self, in_channels: int, out_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_dim, 1),
        )

    def forward(self, feat: torch.Tensor) -> Dict[str, Any]:
        desc = self.proj(feat)
        desc = F.normalize(desc, dim=1)
        g = F.adaptive_avg_pool2d(desc, 1).flatten(1)
        return {"dense_desc": desc, "global_desc": g}


__all__ = ["DescriptorHead"]
