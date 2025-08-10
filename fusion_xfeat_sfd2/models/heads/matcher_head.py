from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn as nn


class SimpleMatcher(nn.Module):
    """最简匹配头（可选）
    输入: 两幅图的稠密或稀疏描述子张量，输出相似度矩阵或 topk 匹配索引。
    为了与现有 eval 兼容，这里主要作为占位/扩展位。
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    @torch.no_grad()
    def forward(self, d1: torch.Tensor, d2: torch.Tensor, topk: int = 512) -> Dict[str, Any]:
        # d1: (N1, D), d2: (N2, D), assumed L2-normalized
        sim = (d1 @ d2.t()) / self.temperature
        idx2 = torch.argmax(sim, dim=1)
        scores = sim[torch.arange(d1.size(0)), idx2]
        vals, order = torch.topk(scores, k=min(topk, scores.numel()))
        pairs = torch.stack([order, idx2[order]], dim=1)
        return {"pairs": pairs, "scores": vals}


__all__ = ["SimpleMatcher"]
