from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn as nn

try:
    # 直接使用官方实现 (只读推理网络)，其 net 为 XFeatModel
    from accelerated_features.modules.xfeat import XFeat as _XFeatInference
    from accelerated_features.modules.model import XFeatModel
except Exception as e:  # noqa
    _XFeatInference = None
    XFeatModel = None
    print(f"[Warn] Cannot import accelerated_features XFeat: {e}")


class XFeatBackbone(nn.Module):
    """XFeat 训练 / 推理统一封装
    提供: feat(密集描述), det(关键点热力或logits), desc(全局/局部描述符)
    占位训练支持：如果需全量训练，可将 requires_grad=True 并加载 XFeatModel。
    """

    def __init__(self, pretrained_path: str | None = None, trainable: bool = False):
        super().__init__()
        self.trainable = trainable
        if trainable:
            if XFeatModel is None:
                raise RuntimeError("XFeatModel 未能导入，无法进行可训练模式")
            self.net = XFeatModel()
            if pretrained_path:
                self._load_weights(pretrained_path)
        else:
            if _XFeatInference is None:
                raise RuntimeError("无法导入推理版 XFeat，请确认 accelerated_features 在 PYTHONPATH")
            self.infer_module = _XFeatInference(weights=pretrained_path)
            self.net = self.infer_module.net  # 访问其底层模型 (便于统一接口)
            for p in self.net.parameters():
                p.requires_grad = False
        self.device_holder = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device_holder)

    def _load_weights(self, path: str):
        try:
            state = torch.load(path, map_location='cpu')
            missing, unexpected = self.load_state_dict(state, strict=False)
            print(f"[XFeatBackbone] Loaded weights. missing={missing}, unexpected={unexpected}")
        except Exception as e:  # noqa
            print(f"[XFeatBackbone] Warn: cannot load pretrained {path}: {e}")

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        # x: (B,3,H,W) or (B,1,H,W)
        if x.dim() != 4:
            raise ValueError("Input tensor must be BCHW")
        # XFeatModel forward 返回: feats(64,H/8,W/8), keypoints(65, H/8,W/8), heatmap(1,H/8,W/8)
        feats, key_logits, heat = self.net(x)
        # 规范化描述符 (局部密集)
        dense_desc = torch.nn.functional.normalize(feats, dim=1)
        # 为保持与之前占位接口一致: det 使用 heat (可靠性) 作为检测图
        out = {
            'feat': feats,              # raw dense feature (未归一化)
            'det': heat,                # detection heatmap (B,1,H/8,W/8)
            'key_logits': key_logits,   # 65 通道原始关键点 logits (可用于NMS/提点)
            'desc_dense': dense_desc,   # 归一化密集描述符 (B,64,H/8,W/8)
        }
        # 额外提供稀疏特征抽取 (训练中通常不用, 评估可用)
        if not self.trainable and hasattr(self, 'infer_module'):
            out_sparse = self.infer_module.detectAndCompute(x)
            # 将 batch list 聚合为 keys
            keypoints_list = []
            scores_list = []
            desc_list = []
            for d in out_sparse:
                keypoints_list.append(d['keypoints'])
                scores_list.append(d['scores'])
                desc_list.append(d['descriptors'])
            out['sparse'] = {
                'keypoints': keypoints_list,
                'scores': scores_list,
                'descriptors': desc_list,
            }
        # 兼容旧 trainer 期待 'desc'
        # 选取全局平均 pooled 描述 (B,64)
        out['desc'] = torch.nn.functional.adaptive_avg_pool2d(dense_desc, 1).squeeze(-1).squeeze(-1)
        return out
