from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn.functional as F
from .base_trainer import BaseTrainer
from ..losses import structural_distill_loss
from ..utils.adaptive_weight import AdaptiveLossWeighter

class DistillTrainer(BaseTrainer):
    def train(self):
        model = self.model_bundle['backbone'].to(self.device)
        teacher = self.model_bundle['teacher'].to(self.device)
        adapter = self.model_bundle['adapter'].to(self.device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self.build_optimizer(params)
        max_epochs = self.cfg['train']['max_epochs']
        scheduler = self.build_scheduler(optimizer, max_epochs)
        steps_per_epoch = max(len(self.train_loader), 1)
        dyn_cfg = self.cfg.get('distill', {})
        use_dyn = dyn_cfg.get('dynamic_layer_weight', True)
        eps = dyn_cfg.get('variance_eps', 1e-6)
        norm_mode = dyn_cfg.get('layer_weight_norm', 'l1')
        # 多损失自适应加权
        aw_cfg = self.cfg.get('adaptive_loss', {'enabled': True, 'method': 'uncertainty', 'init_log_var': 0.0})
        weighter = AdaptiveLossWeighter(aw_cfg).to(self.device)
        static_w = self.cfg['loss']['weights']
        for epoch in range(1, max_epochs + 1):
            model.train(); adapter.train(); teacher.eval()
            running = 0.0
            for step, batch in enumerate(self.train_loader, 1):
                img = batch['image'].to(self.device)
                stu_out = model(img)
                with torch.no_grad():
                    tea_out = teacher(img)
                fused = adapter(stu_out['feat'], tea_out['feat'])['fused']
                # 主特征蒸馏
                d_main = structural_distill_loss(stu_out['feat'], fused.detach())
                # 分层蒸馏
                d_levels = 0.0
                if 'feat_levels' in tea_out:
                    layer_losses = []
                    for tl in tea_out['feat_levels']:
                        s_pool = torch.nn.functional.adaptive_avg_pool2d(stu_out['feat'], tl.shape[-2:])
                        layer_losses.append(F.mse_loss(s_pool, tl.detach(), reduction='mean'))
                    if use_dyn and len(layer_losses) > 1:
                        losses_tensor = torch.stack(layer_losses)
                        inv = 1.0 / (losses_tensor.detach() + eps)
                        if norm_mode == 'softmax':
                            w = torch.softmax(inv, dim=0)
                        else:
                            w = inv / inv.sum()
                        d_levels = (w * losses_tensor).sum()
                    else:
                        d_levels = torch.stack(layer_losses).mean()
                # 描述符蒸馏
                if 'teacher_desc' in tea_out and 'desc_dense' in stu_out:
                    t_desc = tea_out['teacher_desc']
                    s_desc = torch.nn.functional.adaptive_avg_pool2d(stu_out['desc_dense'], t_desc.shape[-2:])
                    d_desc = F.mse_loss(s_desc, t_desc.detach())
                else:
                    d_desc = torch.tensor(0.0, device=self.device)
                # 组装多损失并加权
                loss_dict = {
                    'struct': d_main + 0.5 * d_levels,
                    'distill': d_desc,
                }
                total, _ = weighter.apply(loss_dict, static_w)
                self.apply_warmup(optimizer, epoch, step, steps_per_epoch)
                optimizer.zero_grad(set_to_none=True)
                total.backward(); optimizer.step()
                running += float(total)
                if step % self.cfg.get('print_interval',50)==0:
                    lr_now = optimizer.param_groups[0]['lr']
                    self._log(f"[Distill][Ep{epoch}] step={step} lr={lr_now:.2e} total={float(total):.4f}")
            if scheduler:
                scheduler.step()
            avg = running / steps_per_epoch
            self._append_result('distill', epoch, 'distill_total', avg)
            self._save(epoch, {'model': model.state_dict(), 'adapter': adapter.state_dict(), 'aw_state': weighter.state_dict()})
            self._log(f"[Distill] Epoch {epoch} avg={avg:.4f}")
