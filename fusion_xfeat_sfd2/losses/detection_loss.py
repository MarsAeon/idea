import torch
import torch.nn.functional as F


def detection_loss(pred_heatmap: torch.Tensor, gt_heatmap: torch.Tensor, mask: torch.Tensor | None = None,
                   focal: bool = False, alpha: float = 0.25, gamma: float = 2.0):
    """关键点检测损失
    pred_heatmap: (B,1,H,W) logits
    gt_heatmap: (B,1,H,W) 0/1 or soft labels
    mask: 可选语义权重 (B,1,H,W)
    focal: 是否启用 Focal Loss 形式 (sigmoid)
    alpha,gamma: focal 参数
    若 gt 稀疏可在外部先生成 Gaussian 热力图.
    """
    if focal:
        # sigmoid focal
        p = torch.sigmoid(pred_heatmap)
        ce = F.binary_cross_entropy(p, gt_heatmap, reduction='none')
        pt = torch.where(gt_heatmap == 1, p, 1 - p)  # (B,1,H,W)
        focal_w = (alpha * (1-pt).pow(gamma)) * (gt_heatmap==1) + ((1-alpha) * pt.pow(gamma)) * (gt_heatmap==0)
        loss_map = focal_w * ce
    else:
        loss_map = F.binary_cross_entropy_with_logits(pred_heatmap, gt_heatmap, reduction='none')
    if mask is not None:
        loss_map = loss_map * (1.0 + mask)  # 简单提升语义区域权重
    return loss_map.mean()

__all__ = ['detection_loss']
