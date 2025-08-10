import torch
import torch.nn.functional as F


def multi_view_consistency_loss(desc_a: torch.Tensor, desc_b: torch.Tensor, proj_pairs: torch.Tensor, temperature: float = 0.07):
    """多视图一致性: proj_pairs (K,2) 表示 desc_a[i] 与 desc_b[j] 是同一3D点.
    简化：仅 InfoNCE 正对。
    """
    if proj_pairs.numel() == 0:
        return desc_a.new_tensor(0.0)
    a_idx = proj_pairs[:, 0]
    b_idx = proj_pairs[:, 1]
    a = F.normalize(desc_a[a_idx], dim=-1)
    b = F.normalize(desc_b[b_idx], dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def multi_view_consistency_loss(dense_desc_ref: torch.Tensor, dense_desc_tgt: torch.Tensor, H: torch.Tensor, bidirectional: bool = True, mask_border: int = 0):
    """基于单应性重投影的一致性:
    dense_desc_*: (B,C,H,W) 已归一化
    H: (B,3,3) ref->tgt
    bidirectional: 是否同时投影 tgt->ref 再平均
    mask_border: 忽略边缘像素宽度 (避免插值伪差)
    返回: L2 + 可选双向平均
    """
    B, C, Hh, Ww = dense_desc_ref.shape
    device = dense_desc_ref.device
    # 生成 ref 像素齐次坐标
    y, x = torch.meshgrid(torch.arange(Hh, device=device), torch.arange(Ww, device=device), indexing='ij')
    xy = torch.stack([x, y, torch.ones_like(x)], dim=-1).view(-1, 3)  # (N,3)
    xy = xy.unsqueeze(0).repeat(B, 1, 1)  # (B,N,3)

    def project(feat_src, feat_dst, H_mat):
        tgt_pts = (H_mat @ xy.transpose(1, 2)).transpose(1, 2)
        tgt_pts = tgt_pts[..., :2] / tgt_pts[..., 2:3].clamp(min=1e-8)
        tgt_x = tgt_pts[..., 0] / (Ww - 1) * 2 - 1
        tgt_y = tgt_pts[..., 1] / (Hh - 1) * 2 - 1
        grid = torch.stack([tgt_x, tgt_y], dim=-1).view(B, Hh, Ww, 2)
        sampled = F.grid_sample(feat_dst, grid, align_corners=True)
        # 有效 mask (坐标在 [-1,1])
        valid = (grid[...,0].abs() <= 1) & (grid[...,1].abs() <= 1)
        if mask_border > 0:
            inner = torch.zeros_like(valid)
            inner[:, mask_border:Hh-mask_border, mask_border:Ww-mask_border] = True
            valid = valid & inner
        diff = (sampled - feat_src) ** 2
        if valid.sum() == 0:
            return feat_src.new_tensor(0.0)
        loss = diff.sum(1)[valid].mean()
        return loss

    loss_fw = project(dense_desc_ref, dense_desc_tgt, H)
    if bidirectional:
        # 计算逆单应
        H_inv = torch.inverse(H)
        loss_bw = project(dense_desc_tgt, dense_desc_ref, H_inv)
        return 0.5 * (loss_fw + loss_bw)
    else:
        return loss_fw

__all__ = ['multi_view_consistency_loss']
