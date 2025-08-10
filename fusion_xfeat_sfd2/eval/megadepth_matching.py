from __future__ import annotations
import torch
import numpy as np
try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


def _essential_from_poses(pose1: torch.Tensor, pose2: torch.Tensor) -> torch.Tensor:
    """从两个相机位姿(4x4)计算本质矩阵 (假设已在同尺度).
    pose: T_wc (world->cam) 或 cam->world? 假设输入为 cam->world (常见 SfM 导出)，则需反转得到 R,t.
    这里简单假设 pose 为 4x4 [R|t] cam->world, 转为 world->cam: R^T, -R^T t
    E = [t]_x R  (t 为 归一化方向)
    """
    if pose1.shape[-2:] != (4,4) or pose2.shape[-2:] != (4,4):
        raise ValueError('pose must be (4,4)')
    def inv_pose(p):
        R = p[:3,:3]
        t = p[:3,3]
        R_wc = R; t_wc = t
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        return R_cw, t_cw
    R1, t1 = inv_pose(pose1.cpu().numpy())
    R2, t2 = inv_pose(pose2.cpu().numpy())
    # 相对变换 1->2  (R_rel, t_rel)
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    t_unit = t_rel / (np.linalg.norm(t_rel) + 1e-8)
    tx = np.array([[0, -t_unit[2], t_unit[1]], [t_unit[2], 0, -t_unit[0]], [-t_unit[1], t_unit[0], 0]], dtype=np.float64)
    E = tx @ R_rel
    return torch.from_numpy(E).float()

def evaluate_megadepth(model, dataloader, top_k=1000, dist_thresh=(1.0, 3.0, 5.0), use_geometry=True):
    """MegaDepth 匹配评估 (支持几何过滤):
    若 batch 提供 pose1/pose2 (4x4) 与 K1/K2，则:
      1. 稀疏特征 MNN 匹配
      2. 估计 Essential (RANSAC) 或使用 GT 相对姿势投影误差
      3. 统计内点率 & 像素重投影误差阈值 recall
    否则回退到原像素距离占位统计。
    返回: match@t, inlier_ratio, mean_reproj_error
    """
    device = next(model.parameters()).device
    model.eval()
    stats = {f"match@{t}": 0.0 for t in dist_thresh}
    stats.update({'inlier_ratio': 0.0, 'mean_reproj_error': 0.0})
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            img1 = batch['image']; img2 = batch['image_pair']
            if img1.dim()==3: img1 = img1.unsqueeze(0)
            if img2.dim()==3: img2 = img2.unsqueeze(0)
            img1 = img1.to(device); img2 = img2.to(device)
            out1 = model(img1); out2 = model(img2)
            if 'sparse' not in out1 or 'sparse' not in out2:
                continue
            k1 = out1['sparse']['keypoints'][0]
            d1 = out1['sparse']['descriptors'][0]
            k2 = out2['sparse']['keypoints'][0]
            d2 = out2['sparse']['descriptors'][0]
            if k1.size(0) == 0 or k2.size(0) == 0:
                continue
            if k1.size(0) > top_k: k1 = k1[:top_k]; d1 = d1[:top_k]
            if k2.size(0) > top_k: k2 = k2[:top_k]; d2 = d2[:top_k]
            # MNN 匹配
            sim = d1 @ d2.t()
            idx_12 = sim.argmax(dim=1)
            idx_21 = sim.argmax(dim=0)
            mutual = torch.arange(idx_12.size(0), device=device)
            mask = idx_21[idx_12] == mutual
            mk1 = k1[mask]; mk2 = k2[idx_12[mask]]
            if mk1.numel() == 0:
                continue
            total += 1
            if use_geometry and cv2 is not None and 'K1' in batch and 'K2' in batch and 'pose1' in batch and 'pose2' in batch:
                K1 = batch['K1']; K2 = batch['K2']
                pose1 = batch['pose1']; pose2 = batch['pose2']
                pts1 = mk1.detach().cpu().numpy().astype(np.float32)
                pts2 = mk2.detach().cpu().numpy().astype(np.float32)
                K1_np = K1.numpy().astype(np.float64); K2_np = K2.numpy().astype(np.float64)
                # 归一化像素 (假设同一相机内参? 若不同需各自归一)
                if 'E' in batch:
                    E_gt = batch['E'].numpy().astype(np.float64)
                else:
                    # 用估计 E
                    E_gt, inlier_mask = cv2.findEssentialMat(pts1, pts2, K1_np, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E_gt is None:
                    # 回退
                    dists = torch.norm(mk1 - mk2, dim=1)
                    for t in dist_thresh:
                        stats[f"match@{t}"] += (dists < t).float().mean().item()
                    continue
                # 计算内点 mask (若未返回则再估计一次)
                if 'inlier_mask' not in locals() or inlier_mask is None:
                    _, inlier_mask = cv2.findEssentialMat(pts1, pts2, K1_np, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if inlier_mask is None:
                    inlier_mask = np.ones((pts1.shape[0],1), dtype=np.uint8)
                inl = inlier_mask.ravel().astype(bool)
                inlier_ratio = inl.mean()
                stats['inlier_ratio'] += float(inlier_ratio)
                # 重投影误差 (将 pts1 通过 E 与 pts2 形成对极线距离)
                # 对极线 l2 = E * x1, 误差 = |x2^T l2|/sqrt(a^2+b^2)
                x1h = np.concatenate([pts1, np.ones((pts1.shape[0],1))], axis=1)
                x2h = np.concatenate([pts2, np.ones((pts2.shape[0],1))], axis=1)
                l2 = (E_gt @ x1h.T).T  # (N,3)
                num = np.abs(np.sum(l2 * x2h, axis=1))
                denom = np.sqrt(l2[:,0]**2 + l2[:,1]**2) + 1e-8
                geo_err = num / denom
                stats['mean_reproj_error'] += float(geo_err.mean())
                # 用像素距离统计匹配质量 (参考)
                dists = torch.norm(mk1 - mk2, dim=1)
                for t in dist_thresh:
                    stats[f"match@{t}"] += (dists < t).float().mean().item()
            else:
                dists = torch.norm(mk1 - mk2, dim=1)
                for t in dist_thresh:
                    stats[f"match@{t}"] += (dists < t).float().mean().item()
    if total == 0:
        return stats
    for k in stats:
        if k.startswith('match@'):
            stats[k] /= total
    if total > 0:
        stats['inlier_ratio'] /= max(1,total)
        stats['mean_reproj_error'] /= max(1,total)
    return stats

__all__ = ['evaluate_megadepth']
