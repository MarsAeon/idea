import torch
import torch.nn.functional as F


def semantic_proto_loss(feats: torch.Tensor, labels: torch.Tensor, num_classes: int, proto_momentum: float = 0.9):
    """原型语义聚类损失 (临时内存简化实现). labels: (N,) -1 表示忽略.
    这里使用 batch 内动态原型; 完整版可引入全局内存 bank.
    """
    device = feats.device
    valid = labels >= 0
    feats = feats[valid]
    labels = labels[valid]
    if feats.numel() == 0:
        return feats.new_tensor(0.0)

    # 计算 batch 原型
    protos = []
    proto_labels = []
    for c in labels.unique():
        mask = labels == c
        protos.append(feats[mask].mean(0))
        proto_labels.append(c)
    protos = torch.stack(protos, 0)  # (C',D)
    proto_labels = torch.stack(proto_labels)

    feats_n = F.normalize(feats, dim=-1)
    protos_n = F.normalize(protos, dim=-1)
    logits = feats_n @ protos_n.t() / 0.07

    # 构建映射 labels -> proto 索引
    label_to_idx = {int(c.item()): i for i, c in enumerate(proto_labels)}
    target = torch.tensor([label_to_idx[int(l.item())] for l in labels], device=device)
    loss = F.cross_entropy(logits, target)
    return loss


def semantic_distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0, mask: torch.Tensor | None = None):
    """语义蒸馏: KL(student||teacher) 使用 soft targets.
    student_logits / teacher_logits: (B,C,H,W)
    mask: (B,1,H,W) 可选加权
    """
    # 调整形状
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student/teacher logits shape mismatch")
    t = temperature
    # teacher 分布
    with torch.no_grad():
        p_t = F.softmax(teacher_logits / t, dim=1)
    log_p_s = F.log_softmax(student_logits / t, dim=1)
    kl = F.kl_div(log_p_s, p_t, reduction='none')  # (B,C,H,W)
    kl = kl.sum(1)  # (B,H,W)
    if mask is not None:
        kl = kl * mask.squeeze(1)
    loss = (kl.mean()) * (t * t)
    return loss


def generate_pseudo_labels(teacher_logits: torch.Tensor, confidence_thresh: float = 0.5, image_pooling: bool = True):
    """基于教师语义 logits 生成伪标签
    teacher_logits: (B,C,H,W)
    返回: labels_tensor (B or B,H,W) 以及 mask(ignore=-1) (B,H,W)
    - 若 image_pooling=True: 对每张图 softmax 后全局平均取类别 argmax -> (B,)
    - 否则: 每像素 argmax + 置信度过滤
    被过滤像素标记 -1.
    """
    with torch.no_grad():
        probs = F.softmax(teacher_logits, dim=1)  # (B,C,H,W)
        conf, cls = probs.max(1)  # (B,H,W)
        if image_pooling:
            # 对整图取加权平均概率再 argmax
            pooled = probs.mean(dim=[2,3])  # (B,C)
            labels = pooled.argmax(1)       # (B,)
            mask = (pooled.max(1).values >= confidence_thresh)  # (B,)
            # 将 mask 展开为像素级广播用于后续 kl/distill 加权
            pixel_mask = mask.float()[:,None,None].expand_as(conf)
            return labels, pixel_mask  # labels:(B,), pixel_mask:(B,H,W)
        else:
            mask_pix = conf >= confidence_thresh
            labels = cls.clone()
            labels[~mask_pix] = -1
            return labels, mask_pix.float()


def semantic_pixel_proto_loss(feats: torch.Tensor, labels: torch.Tensor, num_classes: int, temperature: float = 0.07):
    """像素级语义原型损失
    feats: (B,C,H,W)  (例如 desc_dense 或 teacher 投影特征)
    labels: (B,H,W)   值域 [0, num_classes-1] 或 -1 表示忽略
    流程: 取有效像素 -> 计算每类原型 -> InfoNCE / prototype classification
    """
    B, C, H, W = feats.shape
    device = feats.device
    labels_flat = labels.view(-1)
    feats_flat = feats.permute(0,2,3,1).reshape(-1, C)
    valid = labels_flat >= 0
    if valid.sum() == 0:
        return feats.new_tensor(0.0)
    feats_valid = feats_flat[valid]
    labels_valid = labels_flat[valid]
    protos = []
    proto_labels = []
    for c in labels_valid.unique():
        mask = labels_valid == c
        protos.append(feats_valid[mask].mean(0))
        proto_labels.append(c)
    protos = torch.stack(protos, 0)  # (C',D)
    proto_labels = torch.stack(proto_labels)
    feats_n = F.normalize(feats_valid, dim=-1)
    protos_n = F.normalize(protos, dim=-1)
    logits = feats_n @ protos_n.t() / temperature
    # label 映射
    label_to_idx = {int(c.item()): i for i, c in enumerate(proto_labels)}
    target = torch.tensor([label_to_idx[int(l.item())] for l in labels_valid], device=device)
    loss = F.cross_entropy(logits, target)
    return loss


class GlobalPrototypeBank:
    """简单全局原型存储：类别 -> 向量 (EMA 更新)
    feats_dim: 特征维度
    num_classes: 类别数
    momentum: EMA 系数
    """
    def __init__(self, feats_dim: int, num_classes: int, momentum: float = 0.9, device: torch.device | None = None):
        self.num_classes = num_classes
        self.momentum = momentum
        self.device = device or torch.device('cpu')
        self.protos = torch.zeros(num_classes, feats_dim, device=self.device)
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor):
        # feats: (N,D), labels: (N,)
        for c in labels.unique():
            if c < 0: continue
            mask = labels == c
            vec = feats[mask].mean(0)
            if not self.initialized[c]:
                self.protos[c] = vec
                self.initialized[c] = True
            else:
                self.protos[c] = self.momentum * self.protos[c] + (1 - self.momentum) * vec

    def get(self):
        # 返回标准化原型 (仅已初始化)
        protos = self.protos.clone()
        protos = F.normalize(protos, dim=-1)
        mask = self.initialized
        return protos, mask


def semantic_proto_loss_with_bank(feats: torch.Tensor, labels: torch.Tensor, bank: GlobalPrototypeBank, temperature: float = 0.07):
    device = feats.device
    valid = labels >= 0
    feats_v = feats[valid]
    labels_v = labels[valid]
    if feats_v.numel() == 0:
        return feats.new_tensor(0.0)
    feats_n = F.normalize(feats_v, dim=-1)
    bank.update(feats_v.detach(), labels_v.detach())
    protos, mask = bank.get()
    # 仅选择已初始化类别
    idx_valid_classes = mask.nonzero(as_tuple=False).squeeze(1)
    protos_sel = protos[idx_valid_classes]
    if protos_sel.size(0) == 0:
        return feats.new_tensor(0.0)
    logits = feats_n @ protos_sel.t() / temperature
    # labels -> 压缩到 idx_valid_classes 下标
    mapping = {int(c.item()): i for i,c in enumerate(idx_valid_classes)}
    target = torch.tensor([mapping[int(l.item())] for l in labels_v], device=device)
    loss = F.cross_entropy(logits, target)
    return loss


def apply_dual_threshold(conf_map: torch.Tensor, high: float, low: float, mid_weight: float):
    """返回 (labels_mask, weight_mask)
    conf_map: (B,H,W)
    labels_mask: bool (有效像素)
    weight_mask: (B,H,W) 中间置信度降权
    规则: conf>=high weight=1; low<=conf<high weight=mid_weight; conf<low 忽略
    """
    high_mask = conf_map >= high
    low_mask = conf_map < low
    valid = ~low_mask
    weights = torch.ones_like(conf_map)
    mid_zone = (~high_mask) & valid
    weights[mid_zone] = mid_weight
    weights[low_mask] = 0
    return valid, weights

__all__ = ['GlobalPrototypeBank','semantic_proto_loss_with_bank','apply_dual_threshold']
