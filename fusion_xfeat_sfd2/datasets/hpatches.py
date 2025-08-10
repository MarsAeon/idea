from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

__all__ = ["HPatchesDataset"]


def _load_image(path: Path) -> torch.Tensor:
    img = Image.open(path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr[..., None].repeat(3, axis=2)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return tensor


def _resize_with_scale(img: torch.Tensor, size: Optional[int]) -> Tuple[torch.Tensor, float, float]:
    if size is None:
        return img, 1.0, 1.0
    _, h, w = img.shape
    if max(h, w) == size:
        return img, 1.0, 1.0
    if h >= w:
        new_h = size
        new_w = int(round(w * size / h))
    else:
        new_w = size
        new_h = int(round(h * size / w))
    img_resized = torch.nn.functional.interpolate(img[None], (new_h, new_w), mode='bilinear', align_corners=False)[0]
    sy = new_h / h
    sx = new_w / w
    return img_resized, sy, sx


class HPatchesDataset(Dataset):
    """HPatches 数据集加载器 (sequence-level)。

    目录结构 (标准):
      root/
        i_ajuntament/
          1.ppm ... 6.ppm
          H_1_2 ... H_1_6
        v_adam/
          ...

    返回样本字段:
      image: 参考图 (Tensor, 3,H,W)
      image_pair: 目标图 (Tensor, 3,H,W)
      homography: 将参考图坐标映射到目标图 (3,3)  (已按 resize 缩放)
      sequence: 序列名
      idx_target: 目标索引 (>=1)
    训练阶段随机选 target，非训练阶段循环枚举。
    """

    def __init__(self,
                 root: str | Path,
                 split: str = 'test',
                 resize_max: Optional[int] = 480,
                 enumerate_pairs: bool = False,
                 seed: int = 0):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"HPatches root not found: {self.root}")
        self.split = split
        self.resize_max = resize_max
        self.enumerate_pairs = enumerate_pairs  # 若为 True: 每个序列生成多个样本(1->k)
        self.rng = random.Random(seed)
        self.sequences = self._scan_sequences()
        self.index_map: List[Tuple[int, int]] = self._build_index()  # (seq_idx, target_idx)

    def _scan_sequences(self):
        seqs = []
        for seq_dir in sorted(self.root.iterdir()):
            if not seq_dir.is_dir():
                continue
            # 支持 1..6 或 1..5
            images = []
            for i in range(1, 7):
                p = seq_dir / f"{i}.ppm"
                if p.exists():
                    images.append(p)
            if len(images) < 2:
                continue
            homos = {}
            # H_1_j  j>=2
            for j in range(2, len(images) + 1):
                hp = seq_dir / f"H_1_{j}"
                if hp.exists():
                    try:
                        H = np.loadtxt(hp).reshape(3, 3)
                        homos[j] = H
                    except Exception:
                        pass
            seqs.append({
                'name': seq_dir.name,
                'images': images,
                'homos': homos  # key: target index (1-based)
            })
        if not seqs:
            raise RuntimeError(f"No valid HPatches sequences under {self.root}")
        return seqs

    def _build_index(self):
        mapping = []
        for si, seq in enumerate(self.sequences):
            n = len(seq['images'])
            if self.enumerate_pairs:
                for tgt in range(2, n + 1):
                    mapping.append((si, tgt))
            else:
                # 仅保留一个占位 target，真实采样时随机
                mapping.append((si, -1))
        return mapping

    def __len__(self):
        return len(self.index_map)

    def _sample_target(self, seq: Dict[str, Any]) -> int:
        n = len(seq['images'])
        if n < 2:
            return 1
        return self.rng.randint(2, n)

    @staticmethod
    def _scale_homography(H: np.ndarray, sy_ref: float, sx_ref: float, sy_tgt: float, sx_tgt: float) -> np.ndarray:
        # 若原 H 将 ref 像素映射到 tgt 像素: x_tgt = H * x_ref
        # 缩放后: x_ref' = S_ref * x_ref, x_tgt' = S_tgt * x_tgt
        # => x_tgt' = S_tgt * H * S_ref^{-1} * x_ref'
        S_ref = np.array([[sx_ref, 0, 0], [0, sy_ref, 0], [0, 0, 1]], dtype=np.float64)
        S_tgt = np.array([[sx_tgt, 0, 0], [0, sy_tgt, 0], [0, 0, 1]], dtype=np.float64)
        Hs = S_tgt @ H @ np.linalg.inv(S_ref)
        return Hs

    def __getitem__(self, idx: int):
        seq_idx, tgt_flag = self.index_map[idx]
        seq = self.sequences[seq_idx]
        ref_path = seq['images'][0]
        if tgt_flag == -1:
            tgt_idx = self._sample_target(seq)
        else:
            tgt_idx = tgt_flag
        tgt_path = seq['images'][tgt_idx - 1]  # 1-based

        ref_img = _load_image(ref_path)
        tgt_img = _load_image(tgt_path)
        ref_img_resized, sy_r, sx_r = _resize_with_scale(ref_img, self.resize_max)
        tgt_img_resized, sy_t, sx_t = _resize_with_scale(tgt_img, self.resize_max)

        # Homography 若不存在用单位矩阵
        H = seq['homos'].get(tgt_idx, np.eye(3, dtype=np.float64))
        Hs = self._scale_homography(H, sy_r, sx_r, sy_t, sx_t)

        sample = {
            'image': ref_img_resized,         # 供单图训练使用
            'image_pair': tgt_img_resized,    # 用于匹配/一致性
            'homography': torch.from_numpy(Hs).float(),
            'sequence': seq['name'],
            'idx_target': tgt_idx,
            'orig_size_ref': torch.tensor(ref_img.shape[1:]),
            'orig_size_tgt': torch.tensor(tgt_img.shape[1:]),
        }
        return sample
