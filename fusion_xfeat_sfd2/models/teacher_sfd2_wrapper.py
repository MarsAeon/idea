from __future__ import annotations
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

try:
    from sfd2.nets.sfd2 import ResSegNetV2  # 真实 SFD2 网络
except Exception as e:  # noqa
    ResSegNetV2 = None
    print(f"[SFD2Teacher] Warn: cannot import ResSegNetV2: {e}")

class SFD2Teacher(nn.Module):
    """真实 SFD2 教师封装 (基于 ResSegNetV2)
    输出:
      feat: 主 dense 特征 (对齐学生维度, 默认 64 通道)
      struct: 结构/关键点置信图 (B,1,H/8,W/8) 由 keypoint score 聚合
      semantic_logit: 稳定性/语义 logits (若开启 require_stability)
      feat_levels: 多层特征 (list) 用于分层蒸馏
    说明:
      - ResSegNetV2 内部 256 通道主干, 描述子 outdim 可配置 (默认 128)
      - 为与学生 XFeat(64c) 蒸馏便捷, 通过 1x1 conv 降维到 64
      - 仅前向推理, 参数全部冻结
    """
    def __init__(self, weights: Optional[str] = None, project_dim: int = 64, desc_dim: int = 128, require_feature: bool = True, require_stability: bool = True):
        super().__init__()
        if ResSegNetV2 is None:
            # 回退到轻量占位 (保持接口)
            self.fallback = True
            self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1), nn.ReLU())
            self.enc2 = nn.Sequential(nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU())
            self.enc3 = nn.Sequential(nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU())
            self.enc4 = nn.Sequential(nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU())
            self.struct_head = nn.Conv2d(64, 1, 1)
            self.semantic_head = nn.Conv2d(64, 8, 1)
        else:
            self.fallback = False
            self.net = ResSegNetV2(outdim=desc_dim, require_feature=require_feature, require_stability=require_stability)
            if weights:
                self._load(weights)
            # 主干最后阶段输出 256c -> 投影至 project_dim (匹配学生)
            self.proj = nn.Conv2d(256, project_dim, 1)
        # 冻结
        for p in self.parameters():
            p.requires_grad = False

    def _load(self, path: str):
        try:
            state = torch.load(path, map_location='cpu')
            missing, unexpected = self.net.load_state_dict(state, strict=False)
            print(f"[SFD2Teacher] Loaded weights: {path} missing={missing} unexpected={unexpected}")
        except Exception as e:  # noqa
            print(f"[SFD2Teacher] Warn: cannot load weights {path}: {e}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        if self.fallback:
            f1 = self.enc1(x); f2 = self.enc2(f1); f3 = self.enc3(f2); f4 = self.enc4(f3)
            struct_map = self.struct_head(f4)
            semantic_logit = self.semantic_head(f4)
            return {
                'feat': f4,
                'struct': struct_map,
                'semantic_logit': semantic_logit,
                'feat_levels': [f1, f2, f3, f4]
            }
        # 复制自 ResSegNetV2 det_train 逻辑 (单张图像直接处理, 不拼接图对)
        # 为保持简洁, 直接复用内部模块
        m = self.net
        x1a = m.conv1a(x)
        x1b = m.conv1b(x1a); x1c = m.bn1b(x1b)
        x2a = m.conv2a(x1c); x2b = m.conv2b(x2a); x2c = m.bn2b(x2b)
        x3a = m.conv3a(x2c); x3b = m.conv3b(x3a); x3c = m.bn3b(x3b)
        x4 = m.conv4(x3c)
        # keypoint score (semi)
        cPa = m.convPa(x4)
        semi = m.convPb(cPa)  # (B,65,H/8,W/8 base stride ~8)
        semi_exp = torch.exp(semi)
        semi_norm = semi_exp / (semi_exp.sum(1, keepdim=True) + 1e-5)
        kp_score = semi_norm[:, :-1]  # (B,64,H/8,W/8)
        # 构造结构图: 使用平均通道聚合
        struct_map = kp_score.mean(1, keepdim=True)
        # 描述与稳定性
        cDa = m.convDa(x4)
        desc = m.convDb(cDa)  # (B,desc_dim,H/8,W/8)
        desc = torch.nn.functional.normalize(desc, dim=1)
        if getattr(m, 'ConvSta', None) is not None:
            stability = m.ConvSta(x4)  # (B,Cls,H/8,W/8)
        else:
            stability = None
        # 投影主干特征到与学生对齐通道
        feat_proj = self.proj(x4)
        out = {
            'feat': feat_proj,                   # (B,project_dim,H/8,W/8)
            'struct': struct_map,                # (B,1,H/8,W/8)
            'semantic_logit': stability,         # (B,Cs,H/8,W/8) 或 None
            'feat_levels': [x2c, x3c, x4],       # 多层蒸馏 (可扩展)
            'teacher_desc': desc,                # 额外提供 (B,desc_dim,H/8,W/8)
            'keypoint_semi': semi_norm,          # 原始 keypoint logits softmax
        }
        return out

__all__ = ['SFD2Teacher']
