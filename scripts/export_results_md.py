import argparse
import json
import csv
from pathlib import Path

TEMPLATE = """
# 实验结果汇总

## 1. Aachen 相对位姿与PnP
- 平均内点率: {avg_inlier_ratio:.4f}
- 平均匹配数: {avg_matches:.1f}
- 平均旋转角误差 (deg): {avg_rot_err_deg}
- 平均平移角误差 (deg): {avg_t_err_deg}
- Pose Recall @ 阈值: {pose_recalls}

## 2. HPatches / MegaDepth
- 关键指标: {other_metrics}

生成时间: {generated}
"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--jsonl', type=str, required=True, help='训练/评测期间的 JSONL 指标日志')
    p.add_argument('--csv', type=str, default='', help='可选 CSV 指标日志')
    p.add_argument('--out', type=str, required=True, help='导出的 Markdown 路径')
    args = p.parse_args()

    jsonl = Path(args.jsonl)
    if not jsonl.exists():
        raise FileNotFoundError(jsonl)

    last = {}
    with jsonl.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                last.update(obj)
            except Exception:
                pass

    # 可选 CSV 聚合
    other = {}
    if args.csv:
        csvp = Path(args.csv)
        if csvp.exists():
            with csvp.open('r', encoding='utf-8') as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    other = row

    # 组装输出
    pose_recalls = {k: v for k, v in last.items() if k.startswith('(') and k.endswith(')')}
    md = TEMPLATE.format(
        avg_inlier_ratio=float(last.get('avg_inlier_ratio', 0.0)),
        avg_matches=float(last.get('avg_matches', 0.0)),
        avg_rot_err_deg=last.get('avg_rot_err_deg', 'N/A'),
        avg_t_err_deg=last.get('avg_t_err_deg', 'N/A'),
        pose_recalls=pose_recalls or 'N/A',
        other_metrics=other or 'N/A',
        generated=last.get('time', 'unknown')
    )

    Path(args.out).write_text(md, encoding='utf-8')
    print(f'Wrote {args.out}')

if __name__ == '__main__':
    main()
