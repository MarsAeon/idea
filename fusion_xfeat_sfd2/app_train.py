"""主入口：根据阶段配置运行不同 Trainer 与评测."""
from pathlib import Path
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from fusion_xfeat_sfd2.models import XFeatBackbone, SFD2Teacher, FusionAdapter
from fusion_xfeat_sfd2.trainers import PretrainTrainer, DistillTrainer, SemanticTrainer, FinetuneTrainer
from fusion_xfeat_sfd2.datasets import HPatchesDataset, MegaDepthPairsDataset, AachenDataset, RobotCarDataset
from fusion_xfeat_sfd2.eval.hpatches_eval import evaluate_hpatches
from fusion_xfeat_sfd2.eval.megadepth_matching import evaluate_megadepth
from fusion_xfeat_sfd2.eval.pose_recall import evaluate_pose_recall
from fusion_xfeat_sfd2.utils.visualization import draw_keypoints_heatmap, save_semantic_map
from fusion_xfeat_sfd2.eval.pnp_localization import evaluate_pnp_localization

TRAINER_MAP = {
    'pretrain': PretrainTrainer,
    'distill': DistillTrainer,
    'semantic': SemanticTrainer,
    'finetune': FinetuneTrainer,
}

def build_datasets(cfg):
    root = cfg['dataset']['root']
    train_name = cfg['dataset']['train']['name']
    val_name = cfg['dataset']['val']['name']
    # 简化选择
    def make(name, split):
        if name == 'hpatches':
            return HPatchesDataset(root, split, enumerate_pairs=True if split!='train' else False)
        elif name == 'megadepth_pairs':
            return MegaDepthPairsDataset(root, split)
        elif name == 'aachen':
            return AachenDataset(root, split)
        elif name == 'robotcar':
            return RobotCarDataset(root, split)
        else:
            return HPatchesDataset(root, split)
    return {
        'train': make(train_name, cfg['dataset']['train']['splits'][0]),
        'val': make(val_name, cfg['dataset']['val']['splits'][0]),
    }

def run_eval(model, ds, name, pose_recall=False, thresholds=((0.25,2.0),(0.5,5.0),(1.0,10.0)), viz_cfg=None, cfg=None):
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    viz_results = []
    metrics = {}
    # PnP 结构化定位（若提供 structure json）
    if cfg is not None:
        pose_cfg = cfg.get('pose_eval', {})
        struct_json = pose_cfg.get('pnp_structure_json', None)
        if struct_json and Path(struct_json).exists():
            pnp_res = evaluate_pnp_localization(model, loader, structure_json=struct_json,
                                                min_inliers=pose_cfg.get('min_inliers',12),
                                                ransac_reproj_err=pose_cfg.get('ransac_reproj_err',4.0))
            print('[Eval][PnP]', pnp_res)
            metrics['pnp'] = pnp_res
    if pose_recall:
        pr = evaluate_pose_recall(model, loader, thresholds=thresholds)
        metrics['pose_recall'] = pr
    if name == 'hpatches':
        metrics['hpatches'] = evaluate_hpatches(model, loader)
    elif name == 'megadepth_pairs':
        metrics['megadepth'] = evaluate_megadepth(model, loader)
    # 可视化 (仅对前 N 张保存)
    if viz_cfg and viz_cfg.get('draw_semantic', True):
        out_dir = Path(viz_cfg.get('output_dir', 'visualizations')) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for idx, batch in enumerate(loader):
                if idx >= 5:  # 限制数量
                    break
                img = batch['image']  # (1,3,H,W)
                img_u8 = (img[0].permute(1,2,0).cpu().numpy()*255).clip(0,255).astype('uint8')
                if img_u8.shape[2]==1:
                    gray = img_u8[...,0]
                else:
                    import cv2
                    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
                out = model(img.to(next(model.parameters()).device))
                heat = out.get('det')
                if heat is not None:
                    hm = draw_keypoints_heatmap(gray, heat)
                    import cv2
                    cv2.imwrite(str(out_dir / f"heat_{idx}.png"), hm)
                sem_logits = out.get('semantic_logit') or out.get('semantic')
                if sem_logits is not None:
                    save_semantic_map(out_dir / f"semantic_{idx}.png", sem_logits)
    print(f"[Eval][{name}]", metrics)
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='fusion_xfeat_sfd2/configs/base.yaml')
    ap.add_argument('--stage', default='pretrain', choices=list(TRAINER_MAP.keys())+['eval'])
    ap.add_argument('--pretrained', default='accelerated_features/weights/xfeat.pt')
    ap.add_argument('--pose_recall', action='store_true', help='Run pose recall evaluation if possible')
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    backbone = XFeatBackbone(pretrained_path=args.pretrained, trainable=(args.stage not in ['pretrain','eval']))
    teacher = SFD2Teacher()
    adapter = FusionAdapter()
    model_bundle = {'backbone': backbone, 'teacher': teacher, 'adapter': adapter}

    datasets = build_datasets(cfg)

    if args.stage == 'eval':
        viz_cfg = cfg.get('visualization', {}) if cfg.get('logging', {}).get('save_matches_visualization', True) else None
        run_eval(backbone, datasets['val'], cfg['dataset']['val']['name'], pose_recall=args.pose_recall,
                 thresholds=tuple(map(tuple, cfg.get('eval', {}).get('aachen', {}).get('thresholds', [(0.25,2.0),(0.5,5.0),(1.0,10.0)]))), viz_cfg=viz_cfg, cfg=cfg )
        return

    trainer_cls = TRAINER_MAP[args.stage]
    trainer = trainer_cls(cfg, model_bundle, datasets)
    trainer.train()

if __name__ == '__main__':
    main()
