from __future__ import annotations
import torch
from .base_trainer import BaseTrainer
from ..losses import (detection_loss, descriptor_contrastive_loss,
                      structural_distill_loss, semantic_proto_loss, multi_view_consistency_loss, semantic_distill_loss)
import torch.nn.functional as F
from ..losses.semantic_loss import GlobalPrototypeBank, semantic_proto_loss_with_bank, apply_dual_threshold
from ..utils.adaptive_weight import AdaptiveLossWeighter

class FinetuneTrainer(BaseTrainer):
    def __init__(self, cfg, model_bundle, datasets):
        super().__init__(cfg, model_bundle, datasets)
        pseudo_cfg = self.cfg['loss'].get('semantic_pseudo', {})
        self.use_bank = pseudo_cfg.get('use_memory_bank', True)
        self.bank = None
        self.weighter = AdaptiveLossWeighter(self.cfg.get('adaptive_loss', {'enabled': True, 'method': 'uncertainty'})).to(self.device)
        self.static_w = self.cfg['loss']['weights']

    def train(self):
        backbone = self.model_bundle['backbone'].to(self.device)
        teacher = self.model_bundle['teacher'].to(self.device)
        adapter = self.model_bundle['adapter'].to(self.device)
        params = [p for p in backbone.parameters() if p.requires_grad]
        optimizer = self.build_optimizer(params)
        max_epochs = self.cfg['train']['max_epochs']
        scheduler = self.build_scheduler(optimizer, max_epochs)
        steps_per_epoch = max(len(self.train_loader), 1)
        w = self.cfg['loss']['weights']
        dyn_cfg = self.cfg.get('distill', {})
        use_dyn = dyn_cfg.get('dynamic_layer_weight', True)
        eps = dyn_cfg.get('variance_eps', 1e-6)
        norm_mode = dyn_cfg.get('layer_weight_norm', 'l1')
        pseudo_cfg = self.cfg['loss'].get('semantic_pseudo', {})
        high_th = pseudo_cfg.get('high_thresh', 0.7)
        low_th = pseudo_cfg.get('low_thresh', 0.4)
        mid_w = pseudo_cfg.get('mid_weight', 0.5)
        momentum = pseudo_cfg.get('memory_bank_momentum', 0.9)
        for epoch in range(1, max_epochs + 1):
            backbone.train(); adapter.train(); teacher.eval()
            run_det=run_desc=run_struct=run_sem=run_cons=run_sem_dist=0.0
            for step, batch in enumerate(self.train_loader,1):
                img = batch['image'].to(self.device)
                img_pair = batch.get('image_pair')
                H = batch.get('homography')
                out = backbone(img)
                with torch.no_grad():
                    tea_out = teacher(img)
                fused = adapter(out['feat'], tea_out['feat'])['fused']
                gt_heat = torch.zeros_like(out['det'])
                det_l = detection_loss(out['det'], gt_heat)
                desc = out['desc']
                if desc.size(0) > 1:
                    pos_idx = torch.arange(desc.size(0), device=desc.device)
                    d_loss, _ = descriptor_contrastive_loss(desc, desc, pos_idx)
                else:
                    d_loss = desc.new_tensor(0.0)
                dist_l_main = structural_distill_loss(out['feat'], fused.detach())
                # 分层蒸馏动态权重
                dist_l_levels = 0.0
                if 'feat_levels' in tea_out:
                    layer_losses = []
                    for tl in tea_out['feat_levels']:
                        s_pool = torch.nn.functional.adaptive_avg_pool2d(out['feat'], tl.shape[-2:])
                        layer_losses.append(F.mse_loss(s_pool, tl.detach(), reduction='mean'))
                    if use_dyn and len(layer_losses)>1:
                        losses_tensor = torch.stack(layer_losses)
                        inv = 1.0 / (losses_tensor.detach() + eps)
                        if norm_mode=='softmax':
                            w_dyn = torch.softmax(inv, dim=0)
                        else:
                            w_dyn = inv / inv.sum()
                        dist_l_levels = (w_dyn * losses_tensor).sum()
                    else:
                        dist_l_levels = torch.stack(layer_losses).mean()
                if 'teacher_desc' in tea_out and 'desc_dense' in out:
                    t_desc = tea_out['teacher_desc']
                    s_desc = torch.nn.functional.adaptive_avg_pool2d(out['desc_dense'], t_desc.shape[-2:])
                    dist_l_desc = F.mse_loss(s_desc, t_desc.detach())
                else:
                    dist_l_desc = desc.new_tensor(0.0)
                dist_l = dist_l_main + 0.5*dist_l_levels + 0.5*dist_l_desc
                # 语义 (使用 bank + 双阈值像素级)
                sem_dist_l = desc.new_tensor(0.0)
                if 'semantic_logit' in tea_out and tea_out['semantic_logit'] is not None:
                    t_logits = tea_out['semantic_logit']
                    probs = torch.softmax(t_logits, dim=1)
                    conf, labels_pix = probs.max(1)
                    valid_mask, weight_mask = apply_dual_threshold(conf, high_th, low_th, mid_w)
                    labels_pix[~valid_mask] = -1
                    if self.use_bank and self.bank is None:
                        self.bank = GlobalPrototypeBank(out['desc_dense'].size(1), t_logits.size(1), momentum, device=out['desc_dense'].device)
                    # 采样像素
                    dense = out.get('desc_dense')
                    B,C,Hh,Ww = dense.shape
                    flat_feat = dense.permute(0,2,3,1).reshape(-1,C)
                    flat_lab = labels_pix.view(-1)
                    flat_w = weight_mask.view(-1)
                    valid_id = (flat_lab>=0).nonzero(as_tuple=False).squeeze(1)
                    if valid_id.numel()>0:
                        max_pix=4096
                        if valid_id.numel()>max_pix:
                            sel = valid_id[torch.randperm(valid_id.numel(), device=valid_id.device)[:max_pix]]
                        else:
                            sel = valid_id
                        feats_sel = flat_feat[sel]
                        labs_sel = flat_lab[sel]
                        sem_l = semantic_proto_loss_with_bank(feats_sel, labs_sel, self.bank) * flat_w[sel].mean().clamp(min=0.1)
                    else:
                        sem_l = desc.new_tensor(0.0)
                    # 语义蒸馏（若 student 有 logits，可换成学生语义头；此处复用 feat -> teacher）
                    sem_dist_l = semantic_distill_loss(out['feat'], t_logits) if t_logits.shape[2:]==out['feat'].shape[2:] else sem_dist_l
                else:
                    sem_l = semantic_proto_loss(desc, torch.randint(0,3,(desc.size(0),), device=desc.device), num_classes=3)
                if img_pair is not None and H is not None:
                    if img_pair.dim()==3: img_pair = img_pair.unsqueeze(0)
                    img_pair = img_pair.to(self.device)
                    H = H.to(self.device)
                    out_tgt = backbone(img_pair)
                    cons_l = multi_view_consistency_loss(out['desc_dense'], out_tgt['desc_dense'], H.unsqueeze(0) if H.dim()==2 else H)
                else:
                    cons_l = desc.new_tensor(0.0)
                # 多损失不确定性加权
                loss_dict = {
                    'det': det_l,
                    'desc': d_loss,
                    'struct': dist_l,
                    'semantic': sem_l,
                    'consistency': cons_l,
                    'semantic_distill': sem_dist_l,
                }
                total, _ = self.weighter.apply(loss_dict, self.static_w)
                self.apply_warmup(optimizer, epoch, step, steps_per_epoch)
                optimizer.zero_grad(set_to_none=True)
                total.backward(); optimizer.step()
                run_det += float(det_l); run_desc += float(d_loss); run_struct += float(dist_l); run_sem += float(sem_l); run_cons += float(cons_l); run_sem_dist += float(sem_dist_l)
                if step % self.cfg.get('print_interval',50)==0:
                    lr_now = optimizer.param_groups[0]['lr']
                    self._log(f"[Finetune][Ep{epoch}] step={step} lr={lr_now:.2e} total={float(total):.4f}")
            if scheduler:
                scheduler.step()
            n = steps_per_epoch
            self._append_result('finetune', epoch, 'det_loss', run_det/n)
            self._append_result('finetune', epoch, 'desc_loss', run_desc/n)
            self._append_result('finetune', epoch, 'struct_loss', run_struct/n)
            self._append_result('finetune', epoch, 'sem_loss', run_sem/n)
            self._append_result('finetune', epoch, 'cons_loss', run_cons/n)
            self._append_result('finetune', epoch, 'sem_distill_loss', run_sem_dist/n)
            self._save(epoch, {'model': backbone.state_dict(), 'adapter': adapter.state_dict(), 'aw_state': self.weighter.state_dict()})
            self._log(f"[Finetune] Epoch {epoch} total_det={run_det/n:.4f}")
