from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

__all__ = ["RobotCarDataset"]

CONDITION_DIRS = {
    'day': ['sun', 'overcast', 'cloudy'],
    'night': ['night'],
    'rain': ['rain'],
    'snow': ['snow'],
}


def _read_image(path: Path, resize_max: Optional[int] = 720) -> torch.Tensor:
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

class RobotCarDataset(Dataset):
    """Oxford RobotCar 数据加载。
    期望结构:
      root/
        images/route_name/condition/XXXX.jpg
        poses/route_name/condition/XXXX.json  (或统一 poses.json 索引 rel path)
        intrinsics.json (可选)
        split.json (定义 train/val/test 列表)
    字段:
      image, path, pose(若有), condition, route
    """
    def __init__(self, root: str | Path, split: str = 'train', condition_filter: Optional[str] = None, resize_max: int = 720):
        self.root = Path(root)
        self.split = split
        self.condition_filter = condition_filter
        self.resize_max = resize_max
        if not self.root.exists():
            raise FileNotFoundError(f"RobotCar root not found: {self.root}")
        self.split_meta = self._load_split()
        self.poses_index = self._try_load_json(self.root / 'poses.json') or {}
        self.items = self._scan_items()

    def _load_split(self):
        sp = self.root / 'split.json'
        if sp.exists():
            with open(sp, 'r') as f:
                return json.load(f)
        # fallback: 简单全部
        all_imgs = list((self.root / 'images').rglob('*.jpg'))
        return {'train': [str(p) for p in all_imgs], 'val': [str(p) for p in all_imgs[:len(all_imgs)//10]], 'test': [str(p) for p in all_imgs[len(all_imgs)//10:len(all_imgs)//8]]}

    @staticmethod
    def _try_load_json(path: Path):
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _scan_items(self):
        paths = self.split_meta.get(self.split, [])
        items = []
        for p in paths:
            pp = Path(p)
            if not pp.is_absolute():
                pp = self.root / p
            if not pp.exists():
                continue
            rel = str(pp.relative_to(self.root))
            cond = rel.split('/')[2] if 'images' in rel else 'unknown'
            if self.condition_filter and self.condition_filter != cond:
                continue
            pose = self.poses_index.get(rel)
            items.append({'path': pp, 'rel': rel, 'pose': pose, 'condition': cond})
        if not items:
            raise RuntimeError(f"No RobotCar items for split={self.split}")
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        img = _read_image(it['path'], self.resize_max)
        sample: Dict[str, Any] = {
            'image': img,
            'path': str(it['path']),
            'condition': it['condition']
        }
        if it['pose'] is not None:
            pose_arr = np.array(it['pose'], dtype=np.float32)
            sample['pose'] = torch.from_numpy(pose_arr)
        return sample
