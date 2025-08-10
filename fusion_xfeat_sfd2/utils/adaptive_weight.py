from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn

class AdaptiveLossWeighter(nn.Module):
    """
    多损失自适应加权：支持不确定性加权 (Kendall & Gal, CVPR'18) 与 GradNorm 风格权重。
    用法：total, details = weighter.apply(losses, static_weights)
      - losses: {name: tensor}
      - static_weights: {name: float}
    uncertainty: total = Σ [ exp(-s_i) * w_i * L_i + s_i ]
    gradnorm:    total = Σ [ \tilde{w}_i * w_i * L_i ]，\tilde{w}_i 动态归一化（无二阶梯度近似）。
    """
    def __init__(self, cfg: Dict):
        super().__init__()
        self.enabled = bool(cfg.get('enabled', True))
        self.method = cfg.get('method', 'uncertainty')  # 'uncertainty' | 'gradnorm' | 'none'
        init_log_var = float(cfg.get('init_log_var', 0.0))
        self._init_log_var = init_log_var
        self.log_vars = nn.ParameterDict()
        # GradNorm 风格参数
        self.alpha = float(cfg.get('alpha', 0.5))
        self.ema = float(cfg.get('ema', 0.9))
        self.eps = 1e-8
        self.register_buffer('_initialized', torch.tensor(0, dtype=torch.uint8))
        self.init_losses: Dict[str, float] = {}
        self.dyn_weights: Dict[str, float] = {}
        self.last_weights: Dict[str, float] = {}

    def _get_param(self, name: str) -> nn.Parameter:
        if name not in self.log_vars:
            p = nn.Parameter(torch.tensor(self._init_log_var, dtype=torch.float32))
            self.log_vars[name] = p
        return self.log_vars[name]

    def _apply_uncertainty(self, losses: Dict[str, torch.Tensor], static_weights: Dict[str, float]):
        total = 0.0
        weights_out = {}
        for name, L in losses.items():
            s_i = self._get_param(name)
            w_i = static_weights.get(name, 1.0)
            w_unc = torch.exp(-s_i)
            total = total + (w_unc * w_i * L + s_i)
            # 记录有效权重（exp(-s_i)*w_i），用于日志
            weights_out[name] = float((w_unc.detach().cpu() * w_i).item() if isinstance(w_unc, torch.Tensor) else w_i)
        self.last_weights = weights_out
        return total, weights_out

    def _apply_gradnorm(self, losses: Dict[str, torch.Tensor], static_weights: Dict[str, float]):
        # 记录初始损失
        names = list(losses.keys())
        with torch.no_grad():
            if not bool(self._initialized.item()):
                for k in names:
                    self.init_losses[k] = float(losses[k].detach().cpu().item()) + self.eps
                    self.dyn_weights[k] = 1.0
                self._initialized.fill_(1)
        # 计算相对训练速率 r_i = (L_i / L0_i)^alpha
        ratios = {}
        with torch.no_grad():
            for k in names:
                L = float(losses[k].detach().cpu().item())
                r = (L / (self.init_losses.get(k, L + self.eps))) ** self.alpha
                ratios[k] = r
            # 目标均衡：w_tmp_i ∝ 1 / r_i
            inv = {k: 1.0 / (v + self.eps) for k, v in ratios.items()}
            s = sum(inv.values()) + self.eps
            normed = {k: v / s * len(inv) for k, v in inv.items()}  # 令和约等于 task 数
            # EMA 平滑
            for k in names:
                prev = self.dyn_weights.get(k, 1.0)
                self.dyn_weights[k] = self.ema * prev + (1 - self.ema) * normed[k]
        # 计算总损失
        total = 0.0
        weights_out = {}
        for k in names:
            eff_w = self.dyn_weights[k] * static_weights.get(k, 1.0)
            total = total + eff_w * losses[k]
            weights_out[k] = float(eff_w)
        self.last_weights = weights_out
        return total, weights_out

    def apply(self, losses: Dict[str, torch.Tensor], static_weights: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self.enabled or self.method == 'none' or len(losses) == 0:
            total = sum(static_weights.get(k, 1.0) * v for k, v in losses.items())
            self.last_weights = {k: static_weights.get(k, 1.0) for k in losses.keys()}
            return total, self.last_weights
        if self.method == 'uncertainty':
            return self._apply_uncertainty(losses, static_weights)
        if self.method == 'gradnorm':
            return self._apply_gradnorm(losses, static_weights)
        # 默认回退
        total = sum(static_weights.get(k, 1.0) * v for k, v in losses.items())
        self.last_weights = {k: static_weights.get(k, 1.0) for k in losses.keys()}
        return total, self.last_weights

    def get_current_weights(self) -> Dict[str, float]:
        return dict(self.last_weights)
