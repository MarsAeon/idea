from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointHead(nn.Module):
    """简单关键点检测头

    输入: feature (B, C, H, W)
    输出:
      - heatmap: (B, 1, H, W) 经 Sigmoid 的概率图
      - optional logits: (B, 65, H, W) SuperPoint 风格 (含 dustbin)，用于与现有教师对齐
    """

    def __init__(self, in_channels: int, hidden: int = 128, use_superpoint_logits: bool = False):
        super().__init__()
        self.use_superpoint_logits = use_superpoint_logits
        self.det_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        if use_superpoint_logits:
            self.kp_logits = nn.Sequential(
                nn.Conv2d(in_channels, hidden, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 65, 1),  # 64 cells + dustbin
            )
        else:
            self.kp_logits = None

    def forward(self, feat: torch.Tensor) -> Dict[str, Any]:
        heat = torch.sigmoid(self.det_head(feat))
        out: Dict[str, Any] = {"heatmap": heat}
        if self.kp_logits is not None:
            out["logits"] = self.kp_logits(feat)
        return out


__all__ = ["KeypointHead"]
