from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import time
import json
import subprocess
import torch
from torch.utils.data import DataLoader
import math

from ..utils.seed import set_global_seed
from ..utils.checkpoint import save_checkpoint, get_git_hash
from ..utils.config import load_config, config_hash

class BaseTrainer:
    def __init__(self, cfg: Dict[str, Any], model_bundle: Dict[str, Any], datasets: Dict[str, Any]):
        self.cfg = cfg
        self.model_bundle = model_bundle
        self.datasets = datasets
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_global_seed(cfg.get('seed', 42), cfg.get('cudnn_deterministic', True))
        self._build_loaders()
        self.output_dir = Path(cfg.get('output_dir', 'experiments/outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_csv = Path('fusion_xfeat_sfd2/results/exp_log.csv')
        self.results_csv.parent.mkdir(parents=True, exist_ok=True)
        self.json_metrics_path = Path('fusion_xfeat_sfd2/results/metrics.jsonl')
        self.cfg_hash = config_hash(cfg)
        self._dump_env()
        # scheduler 相关参数缓存
        sch_cfg = self.cfg.get('train', {}).get('scheduler', {})
        self.warmup_epochs = sch_cfg.get('warmup_epochs', 0)

    def _dump_env(self):
        freeze_path = self.output_dir / 'pip_freeze.txt'
        if not freeze_path.exists():
            try:
                out = subprocess.check_output(['pip', 'freeze']).decode()
                freeze_path.write_text(out, encoding='utf-8')
            except Exception:
                pass

    def _build_loaders(self):
        train_ds = self.datasets.get('train')
        val_ds = self.datasets.get('val')
        bs = self.cfg['dataset']['train']['batch_size']
        num_workers = self.cfg['hardware']['num_workers']
        self.train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    def build_optimizer(self, params):
        opt_cfg = self.cfg['train']['optimizer']
        name = opt_cfg.get('name', 'adamw').lower()
        lr = opt_cfg.get('lr', 1e-4)
        wd = opt_cfg.get('weight_decay', 0.0)
        if name == 'adamw':
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=tuple(opt_cfg.get('betas', (0.9,0.999))))
        elif name == 'adam':
            opt = torch.optim.Adam(params, lr=lr, weight_decay=wd, betas=tuple(opt_cfg.get('betas', (0.9,0.999))))
        else:
            raise ValueError(f"Unsupported optimizer: {name}")
        # 记录基础 lr 以便 warmup 计算
        for g in opt.param_groups:
            g['base_lr'] = lr
        return opt

    def build_scheduler(self, optimizer, max_epochs: int):
        sch_cfg = self.cfg.get('train', {}).get('scheduler', {})
        name = sch_cfg.get('name', None)
        if not name:
            return None
        name = name.lower()
        warmup = sch_cfg.get('warmup_epochs', 0)
        if name == 'cosine':
            # 若含 warmup, 真正余下周期 = max_epochs - warmup (至少 1)
            t_max = max(1, max_epochs - warmup)
            min_lr = sch_cfg.get('min_lr', 0.0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)
        else:
            raise ValueError(f"Unsupported scheduler: {name}")
        return scheduler

    def apply_warmup(self, optimizer, epoch: int, step: int, steps_per_epoch: int):
        if self.warmup_epochs <= 0:
            return
        total_warmup_steps = self.warmup_epochs * steps_per_epoch
        current_step = (epoch - 1) * steps_per_epoch + step
        if current_step > total_warmup_steps:
            return
        warmup_ratio = current_step / max(1, total_warmup_steps)
        for g in optimizer.param_groups:
            base_lr = g.get('base_lr', g['lr'])
            g['lr'] = base_lr * warmup_ratio

    def train(self):
        raise NotImplementedError

    def _save(self, epoch: int, state: Dict[str, Any]):
        state = dict(state)
        state['config_hash'] = self.cfg_hash
        ckpt_path = self.output_dir / f'epoch_{epoch}.pt'
        save_checkpoint(state, ckpt_path)

    def _log(self, msg: str):
        print(time.strftime('%Y-%m-%d %H:%M:%S'), msg)

    def _append_result(self, stage: str, epoch: int, metric: str, value: float, notes: str = ''):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')},{stage},{self.cfg_hash},{get_git_hash()},{epoch},{metric},{value:.6f},{notes}\n"
        header = "datetime,stage,config_hash,git_hash,epoch,metric,value,notes\n"
        if not self.results_csv.exists():
            self.results_csv.write_text(header, encoding='utf-8')
        with self.results_csv.open('a', encoding='utf-8') as f:
            f.write(line)
        # JSON 行输出
        if self.cfg.get('logging', {}).get('json_metrics', False):
            rec = {
                'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                'stage': stage,
                'config_hash': self.cfg_hash,
                'git_hash': get_git_hash(),
                'epoch': epoch,
                'metric': metric,
                'value': float(value),
                'notes': notes
            }
            with self.json_metrics_path.open('a', encoding='utf-8') as jf:
                jf.write(json.dumps(rec, ensure_ascii=False) + '\n')
