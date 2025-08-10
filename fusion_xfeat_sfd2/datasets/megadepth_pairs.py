from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

__all__ = ["MegaDepthPairsDataset"]


def _read_image(path: Path, resize_max: Optional[int] = None) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    if resize_max is not None:
        _, h, w = tensor.shape
        scale = resize_max / max(h, w)
        if scale < 1.0:
            nh, nw = int(round(h * scale)), int(round(w * scale))
            tensor = torch.nn.functional.interpolate(tensor.unsqueeze(0), (nh, nw), mode='bilinear', align_corners=False)[0]
    return tensor

class MegaDepthPairsDataset(Dataset):
    """加载预生成的图像对 (例如通过 SfM 或官方脚本生成)。
    期望结构:
      root/
        images/
        pairs/xxx_pairs.txt  (每行: rel_path1 rel_path2)
        intrinsics.json (可选)
        poses.json (可选)
    返回字段:
      image, image_pair, path1, path2
    若提供 poses/intrinsics 可扩展加入相机矩阵 / 相对姿。"""
    def __init__(self, root: str | Path, split: str = 'train', pairs_file: Optional[str] = None, resize_max: int = 640):
        self.root = Path(root)
        self.split = split
        self.resize_max = resize_max
        img_root = self.root / 'images'
        if not img_root.exists():
            raise FileNotFoundError(f"MegaDepth images folder missing: {img_root}")
        if pairs_file is None:
            # 默认文件名
            pairs_file = self.root / 'pairs' / f'{split}_pairs.txt'
        self.pairs_path = Path(pairs_file)
        if not self.pairs_path.exists():
            raise FileNotFoundError(f"Pairs file not found: {self.pairs_path}")
        self.pairs: List[Tuple[str, str]] = self._load_pairs(self.pairs_path)
        # 可选相机姿态
        self.poses = self._try_load_json(self.root / 'poses.json')
        self.intrinsics = self._try_load_json(self.root / 'intrinsics.json')

    @staticmethod
    def _load_pairs(path: Path) -> List[Tuple[str, str]]:
        pairs = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
        if not pairs:
            raise RuntimeError(f"No valid pairs in {path}")
        return pairs

    @staticmethod
    def _try_load_json(path: Path):
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        rel1, rel2 = self.pairs[idx]
        p1 = self.root / 'images' / rel1
        p2 = self.root / 'images' / rel2
        if not p1.exists() or not p2.exists():
            raise FileNotFoundError(f"Missing image in pair: {p1} {p2}")
        img1 = _read_image(p1, self.resize_max)
        img2 = _read_image(p2, self.resize_max)
        sample: Dict[str, Any] = {
            'image': img1,
            'image_pair': img2,
            'path1': str(p1),
            'path2': str(p2)
        }
        # 可选: 相机姿态 (如果有)
        if self.poses and rel1 in self.poses and rel2 in self.poses:
            sample['pose1'] = torch.tensor(self.poses[rel1]['pose'], dtype=torch.float32)
            sample['pose2'] = torch.tensor(self.poses[rel2]['pose'], dtype=torch.float32)
        if self.intrinsics and rel1 in self.intrinsics and rel2 in self.intrinsics:
            sample['K1'] = torch.tensor(self.intrinsics[rel1]['K'], dtype=torch.float32)
            sample['K2'] = torch.tensor(self.intrinsics[rel2]['K'], dtype=torch.float32)
        return sample
