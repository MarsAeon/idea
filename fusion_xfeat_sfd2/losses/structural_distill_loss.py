import torch
import torch.nn.functional as F


def structural_distill_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor, mode: str = 'l2'):
    if mode == 'l2':
        return F.mse_loss(student_feat, teacher_feat)
    elif mode == 'smoothl1':
        return F.smooth_l1_loss(student_feat, teacher_feat)
    else:
        raise ValueError(f"Unknown mode {mode}")
