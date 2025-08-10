#!/usr/bin/env python
from __future__ import annotations
import argparse, yaml, copy, subprocess, hashlib, json
from pathlib import Path

BASE_CONFIG = 'fusion_xfeat_sfd2/configs/base.yaml'
OUTPUT_DIR = Path('fusion_xfeat_sfd2/configs/ablation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ABLATION_GRID = [
    {'name': 'no_struct', 'loss.weights.struct': 0.0},
    {'name': 'no_semantic', 'loss.weights.semantic': 0.0},
    {'name': 'no_consistency', 'loss.weights.consistency': 0.0},
    {'name': 'no_sem_distill', 'loss.weights.semantic_distill': 0.0},
    {'name': 'no_multilevel_distill', 'distill.disable_levels': True},
]


def deep_set(cfg: dict, key: str, value):
    parts = key.split('.')
    cur = cfg
    for p in parts[:-1]:
        if p not in cur: cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def config_hash(cfg: dict) -> str:
    m = hashlib.md5(json.dumps(cfg, sort_keys=True).encode())
    return m.hexdigest()[:8]


def generate_variants(base_cfg: dict):
    variants = []
    for item in ABLATION_GRID:
        cfg = copy.deepcopy(base_cfg)
        for k,v in item.items():
            if k == 'name':
                continue
            deep_set(cfg, k, v)
        tag = item['name']
        h = config_hash(cfg)
        out_path = OUTPUT_DIR / f"{tag}_{h}.yaml"
        with open(out_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        variants.append({'name': tag, 'path': str(out_path)})
    return variants


def run_variant(var, stage: str, dry: bool=False):
    cmd = ['python', '-m', 'fusion_xfeat_sfd2.app_train', '--stage', stage, '--config', var['path']]
    print('[RUN]', ' '.join(cmd))
    if not dry:
        subprocess.run(cmd, check=False)


def aggregate_results(csv_path='fusion_xfeat_sfd2/results/exp_log.csv', metric_filter=('finetune','struct_loss')):
    from collections import defaultdict
    import pandas as pd  # optional; fallback to manual parse if absent
    records = []
    try:
        import pandas as pd  # noqa
        df = pd.read_csv(csv_path)
        for _,row in df.iterrows():
            if row['stage']==metric_filter[0] and row['metric']==metric_filter[1]:
                records.append(row.to_dict())
    except Exception:
        # manual parse
        if not Path(csv_path).exists():
            return []
        with open(csv_path,'r',encoding='utf-8') as f:
            header = f.readline().strip().split(',')
            for line in f:
                parts = line.strip().split(',')
                if len(parts)!=len(header): continue
                rec = dict(zip(header, parts))
                if rec['stage']==metric_filter[0] and rec['metric']==metric_filter[1]:
                    records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='finetune')
    ap.add_argument('--generate_only', action='store_true')
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--metric_stage', default='finetune')
    ap.add_argument('--metric_name', default='struct_loss')
    args = ap.parse_args()

    with open(BASE_CONFIG,'r',encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)

    variants = generate_variants(base_cfg)
    print(f"Generated {len(variants)} variants -> {OUTPUT_DIR}")

    if not args.generate_only:
        for v in variants:
            run_variant(v, args.stage, dry=args.dry)

    # 汇总
    recs = aggregate_results(metric_filter=(args.metric_stage, args.metric_name))
    if recs:
        # 简单按 metric 排序
        try:
            recs_sorted = sorted(recs, key=lambda r: float(r['value']))
        except Exception:
            recs_sorted = recs
        print('\nAblation Summary (stage='+args.metric_stage+', metric='+args.metric_name+')')
        print('config_hash,value,epoch,notes')
        for r in recs_sorted:
            print(f"{r['config_hash']},{r['value']},{r['epoch']},{r.get('notes','')}")
    else:
        print('No records aggregated yet.')

if __name__ == '__main__':
    main()
