from __future__ import annotations
import torch
import numpy as np


def _warp_points(kpts: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    # kpts: (N,2) xy
    if kpts.numel() == 0:
        return kpts
    ones = torch.ones(len(kpts), 1, device=kpts.device)
    pts_h = torch.cat([kpts, ones], dim=1)  # (N,3)
    warped = (H @ pts_h.t()).t()
    warped = warped[:, :2] / warped[:, 2:3].clamp(min=1e-8)
    return warped


def evaluate_hpatches(model, dataloader, pixel_thresh=(1, 3, 5), top_k=500):
    model.eval()
    device = next(model.parameters()).device
    pixel_thresh = list(pixel_thresh)
    stats = {k: 0.0 for k in pixel_thresh}
    total_pairs = 0
    with torch.no_grad():
        for batch in dataloader:
            img_ref = batch['image'].to(device)  # (B,3,H,W)? 这里只支持 B=1
            img_tgt = batch['image_pair'].to(device)
            H = batch['homography'].to(device)  # (3,3)
            if img_ref.dim() == 3:
                img_ref = img_ref.unsqueeze(0)
            if img_tgt.dim() == 3:
                img_tgt = img_tgt.unsqueeze(0)

            # 抽取稀疏特征
            out_ref = model(img_ref)
            out_tgt = model(img_tgt)
            if 'sparse' not in out_ref or 'sparse' not in out_tgt:
                continue  # 需要推理模式下的稀疏输出
            kpts_r = out_ref['sparse']['keypoints'][0]
            desc_r = out_ref['sparse']['descriptors'][0]
            kpts_t = out_tgt['sparse']['keypoints'][0]
            desc_t = out_tgt['sparse']['descriptors'][0]

            # 限制 top_k
            if kpts_r.size(0) > top_k:
                kpts_r = kpts_r[:top_k]
                desc_r = desc_r[:top_k]
            if kpts_t.size(0) > top_k:
                kpts_t = kpts_t[:top_k]
                desc_t = desc_t[:top_k]

            if kpts_r.numel() == 0 or kpts_t.numel() == 0:
                continue
            # 余弦相似匹配 (MNN)
            sim = desc_r @ desc_t.t()
            idx_r2t = sim.argmax(dim=1)
            idx_t2r = sim.argmax(dim=0)
            mutual = torch.arange(len(idx_r2t), device=device)
            mutual_mask = idx_t2r[idx_r2t] == mutual
            mkpts_r = kpts_r[mutual_mask]
            mkpts_t = kpts_t[idx_r2t[mutual_mask]]
            if mkpts_r.numel() == 0:
                continue
            # 将参考点用 H 映射到目标坐标
            warped = _warp_points(mkpts_r, H)
            dists = torch.norm(warped - mkpts_t, dim=1)
            total_pairs += 1
            for th in pixel_thresh:
                stats[th] += (dists < th).float().mean().item()
    if total_pairs == 0:
        return {f"mma@{k}px": 0.0 for k in pixel_thresh}
    return {f"mma@{k}px": v / total_pairs for k, v in stats.items()}
