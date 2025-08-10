import cv2
import numpy as np
import torch
from pathlib import Path

__all__ = [
    'draw_keypoints_heatmap', 'draw_matches', 'save_semantic_map'
]

def _to_uint8(img: np.ndarray):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
    return img

def draw_keypoints_heatmap(gray: np.ndarray, heat: torch.Tensor, cmap=cv2.COLORMAP_JET):
    if isinstance(heat, torch.Tensor):
        heat = heat.detach().cpu().float().squeeze().numpy()
    heat = (heat - heat.min()) / (heat.max() + 1e-6)
    hH, hW = heat.shape
    gray = cv2.resize(gray, (hW, hH))
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray_color = cv2.cvtColor(_to_uint8(gray / 255.0), cv2.COLOR_GRAY2BGR)
    heat_color = cv2.applyColorMap(_to_uint8(heat), cmap)
    overlay = (0.4 * heat_color + 0.6 * gray_color).astype(np.uint8)
    return overlay

def draw_matches(img0: np.ndarray, kpts0: np.ndarray, img1: np.ndarray, kpts1: np.ndarray, matches: np.ndarray, max_draw=200):
    if matches.shape[0] > max_draw:
        sel = np.random.choice(matches.shape[0], max_draw, replace=False)
        matches = matches[sel]
    h = max(img0.shape[0], img1.shape[0])
    out = np.zeros((h, img0.shape[1]+img1.shape[1], 3), dtype=np.uint8)
    out[:img0.shape[0], :img0.shape[1]] = img0
    out[:img1.shape[0], img0.shape[1]:] = img1
    offset = img0.shape[1]
    for (i, j) in matches:
        pt0 = tuple(np.round(kpts0[i]).astype(int))
        pt1 = tuple(np.round(kpts1[j]).astype(int) + np.array([offset, 0]))
        color = tuple(np.random.randint(0,255,3).tolist())
        cv2.line(out, pt0, pt1, color, 1, cv2.LINE_AA)
        cv2.circle(out, pt0, 2, color, -1, cv2.LINE_AA)
        cv2.circle(out, pt1, 2, color, -1, cv2.LINE_AA)
    return out

def save_semantic_map(path: Path, sem_logits: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    if sem_logits is None:
        return
    probs = torch.softmax(sem_logits.detach().cpu(), dim=1)
    conf, cls = probs.max(1)
    cls = cls.squeeze().numpy().astype(np.uint8)
    conf = conf.squeeze().numpy()
    H, W = cls.shape
    color_map = np.random.RandomState(0).randint(0,255,(256,3),dtype=np.uint8)
    color_img = color_map[cls]
    conf_norm = (conf - conf.min())/(conf.max()+1e-6)
    conf_color = cv2.applyColorMap(_to_uint8(conf_norm), cv2.COLORMAP_VIRIDIS)
    stacked = np.concatenate([color_img, conf_color], axis=1)
    cv2.imwrite(str(path), stacked)
