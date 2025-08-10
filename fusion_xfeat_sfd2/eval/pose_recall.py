from __future__ import annotations
import torch
import numpy as np
try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

def _q_to_R(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) -> R"""
    if q.shape[-1] == 7:  # (x,y,z,qw,qx,qy,qz) variant guard
        q = q[3:]
    w,x,y,z = q
    n = w*w + x*x + y*y + z*z + 1e-8
    s = 2.0 / n
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*y, s*y*z, s*z*z
    R = np.array([
        [1-(yy+zz), xy-wz, xz+wy],
        [xy+wz, 1-(xx+zz), yz-wx],
        [xz-wy, yz+wx, 1-(xx+yy)]
    ], dtype=np.float64)
    return R

def _pose7_to_RT(pose: torch.Tensor|np.ndarray):
    """pose: (7,) xyz+qwqxqyqz -> R,t (camera-to-world)"""
    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()
    xyz = pose[:3]
    q = pose[3:]
    R = _q_to_R(q)
    t = xyz
    return R, t

def _relative_RT(R1,t1,R2,t2):
    """camera1->world, camera2->world => relative 1->2 (R_rel, t_rel)"""
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    return R_rel, t_rel

def _rot_err_deg(R_gt, R_est):
    dR = R_est @ R_gt.T
    cos = np.clip((np.trace(dR)-1)/2, -1,1)
    return np.degrees(np.arccos(cos))

def _trans_err(R_gt,t_gt,R_est,t_est):
    return np.linalg.norm(t_gt - t_est)

def _essential_to_RT(E, K):
    if cv2 is None:
        return None
    # recoverPose expects normalized coordinates; we assume caller checks
    return E  # placeholder unused directly

def evaluate_pose_recall(model, dataloader, thresholds=((0.25,2.0),(0.5,5.0),(1.0,10.0)), max_db=30, min_matches=30):
    """基于相对位姿误差的召回评估：
    - 抽取全部样本特征，分成 query / db (依 is_query 标志；否则全部视为 db)
    - 对每个 query 与前 max_db 个 db 匹配（MNN 余弦），估计 Essential 并 recoverPose
    - 若两者都含 GT pose: 计算相对 GT 与估计位姿误差 (平移范数 & 旋转角度)；
      满足任一阈值即计入对应召回
    - 返回 recall@阈值 + 平均内点率
    限制：缺少内参时使用像素坐标近似同 K (可能低精度)；无 GT 时跳过样本。
    """
    if cv2 is None:
        return {str(t):0.0 for t in thresholds} | {'avg_inlier_ratio':0.0,'evaluated_pairs':0}
    device = next(model.parameters()).device
    model.eval()
    # 收集特征
    entries = []
    with torch.no_grad():
        for batch in dataloader:
            img = batch['image']
            if img.dim()==3: img = img.unsqueeze(0)
            img = img.to(device)
            out = model(img)
            if 'sparse' not in out:
                continue
            kpts = out['sparse']['keypoints'][0]  # (N,2)
            desc = out['sparse']['descriptors'][0]
            pose = batch.get('pose')
            K = batch.get('K')
            entries.append({
                'kpts': kpts.detach().cpu(),
                'desc': desc.detach().cpu(),
                'pose': pose,
                'K': K,
                'is_query': batch.get('is_query', False)
            })
    if not entries:
        return {str(t):0.0 for t in thresholds} | {'avg_inlier_ratio':0.0,'evaluated_pairs':0}
    queries = [e for e in entries if e['is_query']] or entries
    dbs = [e for e in entries if not e['is_query']] or entries
    recalls = np.zeros(len(thresholds), dtype=np.float64)
    inlier_rs = []
    evaluated = 0
    for q in queries:
        # 仅保留有 GT pose 的 query
        if q['pose'] is None: continue
        for db in dbs[:max_db]:
            if db is q or db['pose'] is None: continue
            k1 = q['kpts']; k2 = db['kpts']
            d1 = q['desc']; d2 = db['desc']
            if k1.size(0)<min_matches or k2.size(0)<min_matches: continue
            # 余弦相似 MNN
            sim = d1 @ d2.t()
            i12 = sim.argmax(1)
            i21 = sim.argmax(0)
            mutual = torch.arange(i12.size(0))
            mask = i21[i12] == mutual
            mk1 = k1[mask]; mk2 = k2[i12[mask]]
            if mk1.size(0) < min_matches: continue
            pts1 = mk1.numpy().astype(np.float32)
            pts2 = mk2.numpy().astype(np.float32)
            # 内参 (若缺失假设同一K，焦距=max边长*1.2)
            if q['K'] is not None and db['K'] is not None:
                Kq = q['K'].numpy().astype(np.float64)
            else:
                h = max(q['kpts'][:,1].max().item(), db['kpts'][:,1].max().item())+1
                w = max(q['kpts'][:,0].max().item(), db['kpts'][:,0].max().item())+1
                f = max(h,w)*1.2
                Kq = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)
            E, inlier_mask = cv2.findEssentialMat(pts1, pts2, Kq, method=cv2.RANSAC, prob=0.999, threshold=1.5)
            if E is None or inlier_mask is None: continue
            inlier_ratio = float(inlier_mask.mean())
            inlier_rs.append(inlier_ratio)
            # 恢复相对位姿
            _, R_est, t_est, _ = cv2.recoverPose(E, pts1, pts2, Kq)
            # 归一化 t_est
            t_est = t_est.squeeze(); t_est = t_est / (np.linalg.norm(t_est)+1e-8)
            # GT 相对
            Rq, tq = _pose7_to_RT(q['pose'])
            Rd, td = _pose7_to_RT(db['pose'])
            R_gt_rel, t_gt_rel = _relative_RT(Rq, tq, Rd, td)
            t_gt_rel_u = t_gt_rel / (np.linalg.norm(t_gt_rel)+1e-8)
            # 旋转 / 平移误差
            r_err = _rot_err_deg(R_gt_rel, R_est)
            t_err = np.linalg.norm(t_gt_rel_u - t_est)
            evaluated += 1
            for i,(t_th,r_th) in enumerate(thresholds):
                if t_err <= t_th and r_err <= r_th:
                    recalls[i] += 1
    if evaluated == 0:
        return {str(t):0.0 for t in thresholds} | {'avg_inlier_ratio':0.0,'evaluated_pairs':0}
    out = {str(thresholds[i]): float(recalls[i]/evaluated) for i in range(len(thresholds))}
    out['avg_inlier_ratio'] = float(np.mean(inlier_rs)) if inlier_rs else 0.0
    out['evaluated_pairs'] = evaluated
    return out

__all__ = ['evaluate_pose_recall']