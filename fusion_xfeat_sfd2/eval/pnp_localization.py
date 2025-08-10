from __future__ import annotations
from typing import Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np
import torch
try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

# 结构 JSON 预期格式（兼容多种键名）：
# {
#   "images": {
#       "image_key": {
#           "K": [[...],[...],[...]] (可选),
#           "observations": [ {"xy": [u,v], "pid": 123}, ... ]
#       }, ...
#   },
#   "points3d": { "123": {"xyz": [x,y,z]}, ... } 或 数组 [{"id":123,"xyz":[x,y,z]}]
# }

def _np(x):
    return np.asarray(x, dtype=np.float32)

def _load_structure(path: str) -> Tuple[Dict[str, Any], Dict[int, np.ndarray]]:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    images = data.get('images', {})
    pts3d = {}
    if isinstance(data.get('points3d'), dict):
        for k, v in data['points3d'].items():
            pid = int(k) if not isinstance(k, int) else k
            pts3d[pid] = _np(v['xyz']).reshape(3)
    elif isinstance(data.get('points3d'), list):
        for it in data['points3d']:
            pid = int(it.get('id'))
            pts3d[pid] = _np(it['xyz']).reshape(3)
    return images, pts3d

def _find_image_key(meta: Dict[str, Any]) -> str | None:
    for k in ('image_path','path','name','filename','file'):
        if k in meta:
            v = meta[k]
            if isinstance(v, (list, tuple)) and len(v)>0:
                v = v[0]
            return Path(str(v)).name
    return None

def _match_2d_to_obs(kpts: np.ndarray, obs_xy: np.ndarray, max_pix: float = 3.0) -> np.ndarray:
    # 返回 obs 索引，-1 表示未匹配
    if kpts.size == 0 or obs_xy.size == 0:
        return np.full((kpts.shape[0],), -1, dtype=np.int32)
    # 朴素最近邻
    d2 = ((kpts[:,None,:] - obs_xy[None,:,:])**2).sum(axis=2)
    idx = d2.argmin(axis=1)
    min_d = np.sqrt(d2[np.arange(kpts.shape[0]), idx])
    idx[min_d > max_pix] = -1
    return idx.astype(np.int32)

@torch.no_grad()
def evaluate_pnp_localization(model, dataloader, structure_json: str, min_inliers: int = 12, ransac_reproj_err: float = 4.0, match_radius: float = 3.0):
    if cv2 is None:
        return {'error': 'opencv not installed'}
    model.eval()
    device = next(model.parameters()).device
    struct_images, points3d = _load_structure(structure_json)
    if len(points3d) == 0:
        return {'error': 'no points3d'}
    successes = 0
    tried = 0
    inliers_all = []
    reproj_all = []
    for batch in dataloader:
        img = batch['image']
        if img.dim()==3: img = img.unsqueeze(0)
        img = img.to(device)
        out = model(img)
        if 'sparse' not in out:
            continue
        # 相机内参
        K = None
        if 'K' in batch:
            K = batch['K'].detach().cpu().numpy().astype(np.float64)
        # 图像键
        key = _find_image_key(batch)
        if key is None:
            continue
        info = struct_images.get(key)
        if info is None:
            continue
        if K is None and 'K' in info:
            K = np.asarray(info['K'], dtype=np.float64)
        if K is None:
            continue
        # 2D-3D 对应
        kpts = out['sparse']['keypoints'][0].detach().cpu().numpy().astype(np.float32)
        obs_list = info.get('observations', [])
        if len(obs_list) == 0:
            continue
        obs_xy = _np([o['xy'] for o in obs_list])
        obs_pid = np.asarray([int(o['pid']) for o in obs_list], dtype=np.int64)
        obs_idx = _match_2d_to_obs(kpts, obs_xy, max_pix=match_radius)
        valid = obs_idx >= 0
        if valid.sum() < min_inliers:
            tried += 1
            continue
        img_pts = kpts[valid]
        pids = obs_pid[obs_idx[valid]]
        obj_pts = np.stack([points3d[int(pid)] for pid in pids], axis=0).astype(np.float64)
        # PnP + RANSAC
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(obj_pts, img_pts, K, None, flags=cv2.SOLVEPNP_ITERATIVE,
                                                      reprojectionError=ransac_reproj_err, iterationsCount=1000, confidence=0.999)
        tried += 1
        if not ok or inliers is None:
            continue
        inl = int(inliers.shape[0])
        if inl < min_inliers:
            continue
        successes += 1
        inliers_all.append(inl)
        # 计算重投影误差（仅统计内点）
        proj, _ = cv2.projectPoints(obj_pts[inliers[:,0]], rvec, tvec, K, None)
        proj = proj.reshape(-1,2).astype(np.float32)
        err = np.linalg.norm(proj - img_pts[inliers[:,0]], axis=1)
        reproj_all.append(float(err.mean()))
    if tried == 0:
        return {'error': 'no_valid_images'}
    res = {
        'pnp_success_rate': successes / tried,
        'avg_inliers': float(np.mean(inliers_all)) if inliers_all else 0.0,
        'avg_reproj_err': float(np.mean(reproj_all)) if reproj_all else 0.0,
        'tried': tried,
        'successes': successes,
    }
    return res

__all__ = ['evaluate_pnp_localization']
