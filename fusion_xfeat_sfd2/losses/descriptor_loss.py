import torch
import torch.nn.functional as F


def descriptor_contrastive_loss(desc1: torch.Tensor, desc2: torch.Tensor, pos_idx: torch.Tensor, temperature: float = 0.07, hard_neg: bool = True, neg_topk: int = 20):
    """对比损失 (InfoNCE) + 可选难负挖掘
    desc1/desc2: (N,D)
    pos_idx: (N,)  desc1[i] 对应 desc2[pos_idx[i]] 为正样本
    hard_neg: 是否限制负样本为 top-K 相似难负
    neg_topk: 选取前 K 相似作为负, 缓解易负冗余
    返回: loss, 正确率
    """
    desc1 = F.normalize(desc1, dim=-1)
    desc2 = F.normalize(desc2, dim=-1)
    sim = desc1 @ desc2.t()  # (N,N)
    # 构建正标签: 每行正样本列为 pos_idx[i]
    N = desc1.size(0)
    arange = torch.arange(N, device=desc1.device)
    logits = sim / temperature
    if hard_neg:
        # 对每行, 取去除正样本后 topK 值与索引, 再与正样本拼接构造子 logits
        mask = torch.zeros_like(sim, dtype=torch.bool)
        mask[arange, pos_idx] = True  # 正样本位置
        neg_mask = ~mask
        # 相似度排序
        sim_neg = sim.masked_fill(mask, -1e9)
        topk_vals, topk_idx = sim_neg.topk(min(neg_topk, N-1), dim=1)
        # 拼接正样本
        pos_vals = sim[arange, pos_idx].unsqueeze(1)
        sub_logits = torch.cat([pos_vals, topk_vals], dim=1) / temperature
        target = torch.zeros(N, dtype=torch.long, device=desc1.device)
        loss = F.cross_entropy(sub_logits, target)
        acc = (sub_logits.argmax(dim=1) == 0).float().mean().item()
        return loss, acc
    else:
        labels = pos_idx
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean().item()
        return loss, acc

__all__ = ['descriptor_contrastive_loss']
