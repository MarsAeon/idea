from __future__ import annotations
import torch
from .base_trainer import BaseTrainer
from ..losses import semantic_proto_loss
from ..losses.semantic_loss import generate_pseudo_labels, semantic_pixel_proto_loss, GlobalPrototypeBank, semantic_proto_loss_with_bank, apply_dual_threshold
from ..utils.adaptive_weight import AdaptiveLossWeighter

class SemanticTrainer(BaseTrainer):
    def __init__(self, cfg, model_bundle, datasets):
        super().__init__(cfg, model_bundle, datasets)
        pseudo_cfg = self.cfg['loss'].get('semantic_pseudo', {})
        self.use_bank = pseudo_cfg.get('use_memory_bank', True)
        self.bank = None
        self.weighter = AdaptiveLossWeighter(self.cfg.get('adaptive_loss', {'enabled': True, 'method': 'uncertainty'})).to(self.device)
        self.static_w = self.cfg['loss']['weights']

    def train(self):
        model = self.model_bundle['backbone'].to(self.device)
        teacher = self.model_bundle.get('teacher').to(self.device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self.build_optimizer(params)
        max_epochs = self.cfg['train']['max_epochs']
        scheduler = self.build_scheduler(optimizer, max_epochs)
        steps_per_epoch = max(len(self.train_loader), 1)
        pseudo_cfg = self.cfg['loss'].get('semantic_pseudo', {})
        conf_th = pseudo_cfg.get('confidence_thresh', 0.5)
        img_pool = pseudo_cfg.get('image_pooling', True)
        use_pixel = not img_pool
        high_th = pseudo_cfg.get('high_thresh', 0.7)
        low_th = pseudo_cfg.get('low_thresh', 0.4)
        mid_w = pseudo_cfg.get('mid_weight', 0.5)
        momentum = pseudo_cfg.get('memory_bank_momentum', 0.9)
        for epoch in range(1, max_epochs + 1):
            model.train(); teacher.eval()
            running = 0.0; cover_rate_acc = 0.0
            for step, batch in enumerate(self.train_loader,1):
                img = batch['image'].to(self.device)
                out = model(img)
                with torch.no_grad():
                    tea_out = teacher(img)
                t_logits = tea_out.get('semantic_logit')
                if t_logits is not None:
                    labels, mask = generate_pseudo_labels(t_logits, conf_th, img_pool)
                    if use_pixel:
                        if 'desc_dense' in out:
                            dense = out['desc_dense']
                        else:
                            dense = torch.nn.functional.normalize(out['feat'], dim=1)
                        probs = torch.softmax(t_logits, dim=1)
                        conf, _ = probs.max(1)
                        valid_mask, weight_mask = apply_dual_threshold(conf, high_th, low_th, mid_w)
                        labels[~valid_mask] = -1
                        if self.use_bank and self.bank is None:
                            self.bank = GlobalPrototypeBank(dense.size(1), t_logits.size(1), momentum, device=dense.device)
                        if self.use_bank:
                            B,C,H,W = dense.shape
                            max_pix = 4096
                            flat_feat = dense.permute(0,2,3,1).reshape(-1,C)
                            flat_lab = labels.view(-1)
                            flat_weight = weight_mask.view(-1)
                            valid_id = (flat_lab>=0).nonzero(as_tuple=False).squeeze(1)
                            if valid_id.numel() > max_pix:
                                sel = valid_id[torch.randperm(valid_id.numel(), device=valid_id.device)[:max_pix]]
                            else:
                                sel = valid_id
                            feats_sel = flat_feat[sel]
                            labels_sel = flat_lab[sel]
                            sem_loss = semantic_proto_loss_with_bank(feats_sel, labels_sel, self.bank)
                            w_mean = flat_weight[sel].mean().clamp(min=0.1)
                            sem_loss = sem_loss * w_mean
                        else:
                            sem_loss = semantic_pixel_proto_loss(dense, labels, num_classes=t_logits.size(1))
                        valid_pix = (labels>=0).float().mean()
                        cover_rate_acc += valid_pix.item()
                    else:
                        feats = out['desc']
                        valid_mask = mask.view(mask.size(0), -1).mean(1)
                        cover_rate_acc += valid_mask.mean().item()
                        if self.use_bank and self.bank is None:
                            self.bank = GlobalPrototypeBank(feats.size(-1), t_logits.size(1), momentum, device=feats.device)
                        if self.use_bank:
                            sem_loss = semantic_proto_loss_with_bank(feats, labels, self.bank)
                        else:
                            sem_loss = semantic_proto_loss(feats, labels, num_classes=t_logits.size(1))
                else:
                    feats = out['desc']
                    rand_labels = torch.randint(0, 3, (feats.size(0),), device=feats.device)
                    sem_loss = semantic_proto_loss(feats, rand_labels, num_classes=3)
                total, _ = self.weighter.apply({'semantic': sem_loss}, self.static_w)
                self.apply_warmup(optimizer, epoch, step, steps_per_epoch)
                optimizer.zero_grad(set_to_none=True)
                total.backward(); optimizer.step()
                running += float(total)
                if step % self.cfg.get('print_interval',50)==0:
                    lr_now = optimizer.param_groups[0]['lr']
                    self._log(f"[Semantic][Ep{epoch}] step={step} lr={lr_now:.2e} total={float(total):.4f}")
            if scheduler:
                scheduler.step()
            avg = running / steps_per_epoch
            cover_epoch = cover_rate_acc / max(1, steps_per_epoch)
            self._append_result('semantic', epoch, 'semantic_total', avg)
            self._append_result('semantic', epoch, 'pseudo_cover', cover_epoch)
            self._save(epoch, {'model': model.state_dict(), 'aw_state': self.weighter.state_dict()})
            self._log(f"[Semantic] Epoch {epoch} avg={avg:.4f} cover={cover_epoch:.3f}")
