from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

__all__ = ["AachenDataset"]


def _read_rgb(path: Path, resize_max: Optional[int] = 1024) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    arr = np.array(img)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    if resize_max:
        _, h, w = t.shape
        scale = resize_max / max(h, w)
        if scale < 1.0:
            nh, nw = int(round(h * scale)), int(round(w * scale))
            t = torch.nn.functional.interpolate(t.unsqueeze(0), (nh, nw), mode='bilinear', align_corners=False)[0]
    return t

class AachenDataset(Dataset):
    """Aachen Day-Night 数据集加载。
    期望结构:
      root/
        images/images_upright/db/xxx.jpg
        images/images_upright/query/night/xxx.jpg
        poses/ (可选 JSON 或 txt 形式)
        intrinsics.json (可选)
        split.json (定义 train/val/query 列表)
    返回:
      image: 图像张量
      path: 路径
      pose: (7,) xyz+四元数 (若存在)
      K: (3,3) 内参 (若存在)
      Twc/Tcw: (4,4) 齐次位姿（若能从 pose 恢复）
      is_query: bool
    """
    def __init__(self, root: str | Path, split: str = 'train', resize_max: int = 1024):
        self.root = Path(root)
        self.split = split
        self.resize_max = resize_max
        if not self.root.exists():
            raise FileNotFoundError(f"Aachen root not found: {self.root}")
        self.split_meta = self._load_split()
        self.poses = self._try_load_json(self.root / 'poses' / 'poses.json') or {}
        self.intrinsics = self._try_load_json(self.root / 'intrinsics.json') or {}
        self.items = self._build_items()

    def _load_split(self):
        sp = self.root / 'split.json'
        if sp.exists():
            with open(sp, 'r') as f:
                return json.load(f)
        # fallback: 简单遍历 query + db
        db = list((self.root / 'images' / 'images_upright' / 'db').rglob('*.jpg'))
        query = list((self.root / 'images' / 'images_upright' / 'query').rglob('*.jpg'))
        return {'train': [str(p) for p in db], 'val': [str(p) for p in query], 'query': [str(p) for p in query]}

    @staticmethod
    def _try_load_json(path: Path):
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _quat_to_rot(self, q: np.ndarray) -> np.ndarray:
        # q: (4,) w,x,y,z 或 x,y,z,w，自适应常见两种排列
        if abs(np.linalg.norm(q) - 1.0) > 1e-3:
            q = q / (np.linalg.norm(q) + 1e-12)
        if abs(q[0]) >= max(abs(q[1]), abs(q[2]), abs(q[3])):
            w,x,y,z = q
        else:
            x,y,z,w = q
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        R = np.array([
            [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
            [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
            [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
        ], dtype=np.float32)
        return R

    def _pose_to_Twc(self, pose: Any) -> Optional[np.ndarray]:
        # pose 期望 [tx,ty,tz,qw,qx,qy,qz] 或 [tx,ty,tz,qx,qy,qz,qw]
        arr = np.array(pose, dtype=np.float32).reshape(-1)
        if arr.size != 7:
            return None
        t = arr[:3]
        q = arr[3:]
        R = self._quat_to_rot(q)
        Twc = np.eye(4, dtype=np.float32)
        Twc[:3,:3] = R
        Twc[:3, 3] = t
        return Twc

    def _build_items(self):
        key = 'train' if self.split == 'train' else 'query'
        paths = self.split_meta.get(self.split, self.split_meta.get(key, []))
        items = []
        for p in paths:
            pp = Path(p)
            if not pp.is_absolute():
                pp = self.root / p
            if not pp.exists():
                continue
            rel = str(pp.relative_to(self.root))
            pose = self.poses.get(rel)
            K = self.intrinsics.get(rel)
            Twc = self._pose_to_Twc(pose) if pose is not None else None
            Tcw = (np.linalg.inv(Twc) if Twc is not None else None)
            items.append({'path': pp, 'rel': rel, 'pose': pose, 'K': K, 'is_query': 'query' in rel, 'Twc': Twc, 'Tcw': Tcw})
        if not items:
            raise RuntimeError(f"No Aachen items for split={self.split}")
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        img = _read_rgb(it['path'], self.resize_max)
        sample: Dict[str, Any] = {
            'image': img,
            'path': str(it['path']),
            'is_query': it['is_query']
        }
        if it['pose'] is not None:
            pose_arr = np.array(it['pose'], dtype=np.float32)
            sample['pose'] = torch.from_numpy(pose_arr)
        if it['K'] is not None:
            K_arr = np.array(it['K'], dtype=np.float32).reshape(3, 3)
            sample['K'] = torch.from_numpy(K_arr)
        if it.get('Twc') is not None:
            sample['Twc'] = torch.from_numpy(it['Twc'])
        if it.get('Tcw') is not None:
            sample['Tcw'] = torch.from_numpy(it['Tcw'])
        return sample
