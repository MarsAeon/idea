import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_last(jsonl_path: Path):
    last = {}
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                last.update(obj)
            except Exception:
                pass
    return last


def plot_pose_recalls(last: dict, out_path: Path):
    # keys like '(0.25, 2.0)', '(0.5, 5.0)' ...
    items = [(k, v) for k, v in last.items() if isinstance(k, str) and k.startswith('(') and k.endswith(')')]
    if not items:
        return False
    # sort by the numeric angle threshold inside the tuple string
    def sort_key(kv):
        k = kv[0]
        try:
            tup = eval(k)
            if isinstance(tup, (list, tuple)) and len(tup) >= 2:
                return float(tup[1])
        except Exception:
            pass
        return float('inf')

    items.sort(key=sort_key)
    labels = [k for k, _ in items]
    values = [float(v) for _, v in items]

    plt.figure(figsize=(6, 3.2))
    bars = plt.bar(labels, values, color="#4472c4")
    plt.ylim(0, 1.0)
    plt.ylabel('Recall')
    plt.title('Aachen Pose Recall')
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True, help='JSONL 指标日志路径')
    ap.add_argument('--out', default='docs/figs/pose_recall.png', help='输出图片路径')
    args = ap.parse_args()

    last = load_last(Path(args.jsonl))
    ok = plot_pose_recalls(last, Path(args.out))
    if ok:
        print(f'Saved {args.out}')
    else:
        print('未发现可用的 Pose Recall 指标，跳过绘图。')


if __name__ == '__main__':
    main()
