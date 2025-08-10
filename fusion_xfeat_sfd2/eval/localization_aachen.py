from __future__ import annotations
import torch
import numpy as np
from pathlib import Path
try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None
from fusion_xfeat_sfd2.utils.visualization import draw_matches

def _dummy_project(kpts: torch.Tensor, K: torch.Tensor):
    # 占位：真实实现需将2D匹配回投到3D地图点；这里返回伪 3D (深度=1)
    z = torch.ones(len(kpts), 1, device=kpts.device)
    return torch.cat([kpts, z], dim=1)

def _to_numpy_kpts(kpts_tensor: torch.Tensor):
    return kpts_tensor.detach().cpu().numpy().astype(np.float32)

def _match_descriptors(desc1: np.ndarray, desc2: np.ndarray, topk: int = 500):
    # 余弦相似 -> 距离 (1 - cos)
    # desc shape (N,D) 已 L2
    sim = desc1 @ desc2.T  # (N1,N2)
    # 互选最近 (简单 MNN)
    idx2 = sim.argmax(axis=1)
    sim12 = sim[np.arange(sim.shape[0]), idx2]
    idx1_back = sim.argmax(axis=0)
    mutual = np.where(idx1_back[idx2] == np.arange(sim.shape[0]))[0]
    pairs = [(i, idx2[i], sim12[i]) for i in mutual]
    pairs.sort(key=lambda x: -x[2])
    return pairs[:topk]

def _safe_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def _extract_T(meta):
    """从 batch meta 中提取 Twc 或 Tcw（任一即可）。返回 (Twc, Tcw) 二选一，可能为 None。
    支持键: 'Twc','Tcw','pose','T' 等（4x4）。
    """
    cand = None
    for k in ['Twc','Tcw','pose','T']:
        if k in meta:
            v = meta[k]
            cand = _safe_numpy(v)
            break
    if cand is None:
        return None, None
    M = cand.squeeze()
    if M.shape == (4,4):
        if 'Tcw' in meta or (('Twc' not in meta) and 'Tcw' in ['pose','T']):
            Tcw = M
            Twc = np.linalg.inv(Tcw)
        else:
            Twc = M
            Tcw = np.linalg.inv(Twc)
        return Twc, Tcw
    return None, None

def _relative_from_meta(metaA, metaB):
    TwcA, TcwA = _extract_T(metaA)
    TwcB, TcwB = _extract_T(metaB)
    if TcwA is None or TcwB is None:
        return None
    # 相对位姿: T_12 = Tcw2 * Twc1
    T12 = TcwB @ TwcA
    R = T12[:3,:3]
    t = T12[:3,3]
    return R, t

def _rot_err_deg(R_est, R_gt):
    R = R_est.T @ R_gt
    tr = np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))

def _t_err_deg(t_est, t_gt):
    te = t_est.reshape(-1); tg = t_gt.reshape(-1)
    if np.linalg.norm(te) < 1e-9 or np.linalg.norm(tg) < 1e-9:
        return 180.0
    te = te / (np.linalg.norm(te) + 1e-12)
    tg = tg / (np.linalg.norm(tg) + 1e-12)
    cosang = float(np.clip(np.dot(te, tg), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

def evaluate_aachen(model, dataloader, thresholds=((0.25,2.0),(0.5,5.0),(1.0,10.0)), save_viz: bool=False, viz_dir: str='visualizations/aachen', max_draw: int=200):
    """Aachen 位姿评测 (简化真实实现):
    现在支持:
      - 2D-2D E 矩阵 + recoverPose 估计相对位姿 (R,t)，统计内点率
      - 如 meta 含 GT 位姿 (Twc/Tcw/pose)，计算相对位姿误差并按阈值统计 recall
      - 可选保存匹配连线可视化
    thresholds: 采用 (x, deg) 形式时，使用第二个值作为角度阈值，分别用于 R 与 t 的角度误差。
    """
    if cv2 is None:
        return {'error': 'opencv not installed'}
    model.eval()
    device = next(model.parameters()).device
    images = []
    metas = []
    with torch.no_grad():
        for batch in dataloader:
            img = batch['image']
            if img.dim()==3: img = img.unsqueeze(0)
            img = img.to(device)
            out = model(img)
            if 'sparse' not in out:
                continue
            kp = out['sparse']['keypoints'][0]  # (Nk,2)
            desc = out['sparse']['descriptors'][0]  # (Nk,D)
            images.append({'kpts': kp, 'desc': desc, 'meta': batch})
            metas.append(batch)
    if len(images) < 2:
        return {str(t): 0.0 for t in thresholds}
    total_pairs = 0
    inlier_acc = []
    match_acc = []
    rot_errs = []
    t_errs = []
    recalls = [0 for _ in range(len(thresholds))]
    have_gt_pose = False
    viz_path = Path(viz_dir)
    if save_viz:
        viz_path.mkdir(parents=True, exist_ok=True)
    for i in range(len(images)-1):
        A = images[i]; B = images[i+1]
        k1 = _to_numpy_kpts(A['kpts'])
        k2 = _to_numpy_kpts(B['kpts'])
        d1 = A['desc'].detach().cpu().numpy().astype(np.float32)
        d2 = B['desc'].detach().cpu().numpy().astype(np.float32)
        if k1.shape[0] < 8 or k2.shape[0] < 8:
            continue
        pairs = _match_descriptors(d1, d2, topk=800)
        if len(pairs) < 8:
            continue
        pts1 = np.array([k1[p[0]] for p in pairs], dtype=np.float32)
        pts2 = np.array([k2[p[1]] for p in pairs], dtype=np.float32)
        K = None
        if 'K' in A['meta'] and 'K' in B['meta']:
            K = _safe_numpy(A['meta']['K']).astype(np.float64)
        # 估计本质矩阵 + recoverPose
        if K is not None:
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.5)
        else:
            E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, prob=0.999, threshold=1.5)
        if E is None or mask is None:
            continue
        inliers = int(mask.sum())
        total = len(mask)
        inlier_ratio = inliers / max(1,total)
        total_pairs += 1
        inlier_acc.append(inlier_ratio)
        match_acc.append(total)
        R_est = None; t_est = None
        if K is not None:
            try:
                _, R_est, t_est, mask_rec = cv2.recoverPose(E, pts1, pts2, K)
                if mask_rec is not None:
                    inlier_ratio = float(mask_rec.sum()) / max(1,float(mask_rec.size))
                    inlier_acc[-1] = inlier_ratio
            except Exception:
                pass
        # GT 相对位姿与误差
        rel = _relative_from_meta(A['meta'], B['meta'])
        if rel is not None and R_est is not None and t_est is not None:
            have_gt_pose = True
            R_gt, t_gt = rel
            re = _rot_err_deg(R_est, R_gt)
            te = _t_err_deg(t_est, t_gt)
            rot_errs.append(re); t_errs.append(te)
            for ti, th in enumerate(thresholds):
                # 使用第二个阈值作为角度阈值（deg）
                deg_th = th[1] if isinstance(th, (list, tuple)) and len(th)>=2 else float(th)
                if re <= deg_th and te <= deg_th:
                    recalls[ti] += 1
        # 保存匹配可视化
        if save_viz:
            try:
                def to_u8(rgb_tensor: torch.Tensor):
                    t = rgb_tensor[0] if rgb_tensor.dim()==4 else rgb_tensor
                    t = t.detach().cpu().permute(1,2,0).numpy()
                    t = (np.clip(t, 0, 1) * 255).astype(np.uint8)
                    bgr = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
                    return bgr
                imgA = to_u8(A['meta']['image'])
                imgB = to_u8(B['meta']['image'])
                match_idx = np.array([[p[0], p[1]] for p in pairs], dtype=np.int32)
                vis = draw_matches(imgA, k1, imgB, k2, match_idx, max_draw=max_draw)
                cv2.imwrite(str(viz_path / f"matches_{i:04d}.png"), vis)
            except Exception:
                pass
    if total_pairs == 0:
        return {str(t): 0.0 for t in thresholds}
    mean_inlier = float(np.mean(inlier_acc))
    out = {str(thresholds[i]): (recalls[i]/total_pairs if have_gt_pose else mean_inlier) for i in range(len(thresholds))}
    out.update({'avg_matches': float(np.mean(match_acc)), 'avg_inlier_ratio': mean_inlier})
    if have_gt_pose and len(rot_errs)>0:
        out.update({'avg_rot_err_deg': float(np.mean(rot_errs)), 'avg_t_err_deg': float(np.mean(t_errs))})
    return out

__all__ = ['evaluate_aachen']
