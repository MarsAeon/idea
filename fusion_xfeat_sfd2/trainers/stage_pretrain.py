from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn.functional as F
from .base_trainer import BaseTrainer

class PretrainTrainer(BaseTrainer):
    def train(self):
        model = self.model_bundle['backbone'].to(self.device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self.build_optimizer(params)
        max_epochs = self.cfg['train']['max_epochs']
        scheduler = self.build_scheduler(optimizer, max_epochs)
        steps_per_epoch = max(len(self.train_loader), 1)
        for epoch in range(1, max_epochs + 1):
            model.train()
            running = 0.0
            for step, batch in enumerate(self.train_loader, 1):
                img = batch['image'].to(self.device)
                out = model(img)
                det_map = out['det']
                gt = torch.zeros_like(det_map)
                det_loss = F.binary_cross_entropy(det_map, gt)
                self.apply_warmup(optimizer, epoch, step, steps_per_epoch)
                optimizer.zero_grad(set_to_none=True)
                det_loss.backward(); optimizer.step()
                running += det_loss.item()
                if step % self.cfg.get('print_interval',50)==0:
                    lr_now = optimizer.param_groups[0]['lr']
                    self._log(f"[Pretrain][Ep{epoch}] step={step} lr={lr_now:.2e} loss={det_loss.item():.4f}")
            if scheduler:
                scheduler.step()
            avg_loss = running / steps_per_epoch
            self._append_result('pretrain', epoch, 'det_loss', avg_loss)
            self._save(epoch, {'model': model.state_dict()})
            self._log(f"[Pretrain] Epoch {epoch} avg_loss={avg_loss:.4f}")
