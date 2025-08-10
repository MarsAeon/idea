import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 简单聚合：从可视化目录选取若干匹配图，拼接网格

def grid_of_images(img_paths, cols=3, resize_to=(640, 360)):
    imgs = []
    for p in img_paths:
        try:
            im = Image.open(p).convert('RGB')
            if resize_to:
                im = im.resize(resize_to)
            imgs.append(np.asarray(im))
        except Exception:
            pass
    if not imgs:
        return None
    rows = (len(imgs) + cols - 1) // cols
    h, w, c = imgs[0].shape
    canvas = np.ones((rows*h, cols*w, c), dtype=np.uint8) * 255
    for i, im in enumerate(imgs):
        r = i // cols
        c_ = i % cols
        canvas[r*h:(r+1)*h, c_*w:(c_+1)*w] = im
    return Image.fromarray(canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--viz_dir', type=str, default='visualizations/aachen')
    ap.add_argument('--out_dir', type=str, default='docs/figs')
    ap.add_argument('--max_imgs', type=int, default=9)
    args = ap.parse_args()

    viz = Path(args.viz_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    imgs = sorted(list(viz.glob('matches_*.png')))
    imgs = imgs[:args.max_imgs]
    grid = grid_of_images(imgs)
    if grid is not None:
        out_path = out / 'aachen_matches_grid.png'
        grid.save(out_path)
        print(f'Saved {out_path}')

if __name__ == '__main__':
    main()
