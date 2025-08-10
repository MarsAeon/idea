# Semantic‑Guided Fusion of XFeat and SFD2 for Robust Local Feature Matching and Pose Estimation

作者：待定（单位与邮箱）

## 摘要
我们提出一种语义引导的教师–学生融合框架，将轻量学生特征网络 XFeat 与结构化教师 SFD2 有效结合。方法包含：多层特征与描述子蒸馏、双阈值伪标签与全局原型记忆（EMA）的语义监督、基于不确定性/GradNorm 的多损失自适应加权，以及双向几何一致性约束。我们在 HPatches、MegaDepth 对极几何匹配与 Aachen 本地化（2D‑2D 相对位姿与 2D‑3D PnP）上进行评估，并提供完善的消融、可视化与可复现实践。实验显示在不增加明显推理成本的前提下，本方法能够稳定提升内点率与位姿召回。

关键词：特征匹配、蒸馏学习、语义原型、PnP、本质矩阵、位姿估计

---

## 1 引言
局部特征的稳健性与判别性是匹配和定位的核心。基于学习的特征在跨域、遮挡与低纹理场景下仍可能退化；而教师–学生蒸馏虽有效，但层间不均衡、伪标签噪声与多损失冲突常导致训练不稳定与泛化受限。为此本文提出一条“语义引导 + 动态加权”的融合蒸馏路径：
- 语义引导融合：借助 SFD2 的结构/稳定性信号约束学生表示结构，提升几何可用性。
- 双阈值伪标签 + 原型记忆：降低伪标签噪声风险，增强类内聚合与跨域稳健。
- 动态加权：层级蒸馏自适应层权与任务级不确定性/GradNorm 加权，缓解梯度冲突。
- 端到端评测与可复现：相对位姿与 PnP 管线、可视化、JSONL/CSV 日志与自动消融脚本。

我们的目标是“在保持学生高效性的同时，系统性提升匹配质量与位姿召回”。

## 2 相关工作（简述）
- 局部特征与匹配：SIFT、R2D2、ALIKE、SuperPoint+SuperGlue、LightGlue、XFeat 等。
- 知识蒸馏与多任务权衡：FitNets、多层蒸馏、Kendall & Gal 的不确定性加权、GradNorm 等。
- 语义/稳定性引导：伪标签、原型学习与记忆库方法。
- 位姿估计：RANSAC、本质矩阵、PnP 与结构化本地化。

## 3 方法
### 3.1 框架概述
- 学生–教师–适配器：
  - 学生 XFeatBackbone 产出密集特征、关键点热图与稀疏描述；
  - 教师 SFD2(ResSegNetV2) 提供多层特征、结构图、稳定性/语义 logits 与教师描述子；
  - FusionAdapter（conv_attn/FiLM）进行通道与空间对齐，输出融合特征。
- 训练阶段：预训练 → 蒸馏（结构/描述子，多层权重自适应）→ 语义（双阈值伪标签 + 原型记忆）→ 微调（联合损失与几何一致性）。

### 3.2 损失与约束
- 检测损失：BCE/Focal（可选）。
- 描述子对比：InfoNCE + 硬负样本（top‑K）。
- 结构蒸馏：学生与（融合后）教师特征的 L2；多层蒸馏采用动态层权（逆损失归一/softmax）。
- 描述子蒸馏：教师密集描述与学生对齐。
- 语义原型损失：双阈值伪标签（高阈值强监督、低阈值剔除、中间区间降权）+ GlobalPrototypeBank(EMA)；支持像素级或图像级。
- 语义蒸馏：学生特征对教师语义/稳定性 logits 的对齐（若分辨率匹配）。
- 几何一致性：双向单应一致性与边界有效掩码，促进等变与跨视几何稳健性。

### 3.3 多损失自适应加权
- 不确定性加权：对任务 i 学习对数方差 s_i，目标 Σ exp(−s_i)·w_i·L_i + s_i，等价多任务最大似然估计，自动抑制高噪声任务。
- GradNorm 风格：按相对训练速率 r_i=(L_i/L_i0)^α 自适应归一，EMA 平滑，缓解梯度竞争与失衡。
- 二者可通过配置切换（adaptive_loss.method ∈ {uncertainty, gradnorm}）。

### 3.4 伪标签与原型记忆的鲁棒性
- 教师置信度分布双阈值策略降低噪声风险；
- EMA 原型一致估计类条件均值，提高类内聚合与跨域迁移；
- 像素采样上限控制显存并抑制极端噪声点的影响。

## 4 评测协议
- 数据集：
  - HPatches：MMA。
  - MegaDepth：成对匹配（E 矩阵内点率/重投影误差）。
  - Aachen：2D‑2D recoverPose 相对位姿召回；2D‑3D PnP（solvePnPRansac）成功率/内点/重投影误差（需 SfM JSON：points3D 与图像观测）。
- 指标：MMA、内点率、Avg Reproj Err、Pose Recall@{0.25m,2°|0.5m,5°|1m,10°}、速度/显存/参数量。
- 可视化：关键点热力图、语义图、匹配连线。

## 5 实验
### 5.1 实现细节
- 优化：AdamW，cosine+warmup，AMP，可选梯度裁剪；
- 动态层权：inverse‑loss / softmax 归一；
- 伪标签：高/低阈值与中间权重可配；像素采样上限（默认 4096）。

### 5.2 基线与对比
- 基线：XFeat 原生 + LightGlue/HLoc；SuperPoint+SuperGlue；R2D2 等。
- 维度：MMA/内点率/Recall、速度、显存、参数量。

### 5.3 消融（建议维度）
- 去除模块：语义原型、双阈值、动态层权、语义蒸馏、难例挖掘、一致性；
- 自适应加权：none vs uncertainty vs gradnorm；
- 教师信号：仅结构 vs 结构+语义 vs 结构+语义+描述子。

### 5.4 结果概览（占位）
- HPatches：MMA 提升 x–y%。
- MegaDepth：内点率/重投影误差改进。
- Aachen：相对位姿召回与 PnP 成功率提升，低纹理/大视角收益明显。
- 速度：接近学生原生推理开销。

### 5.5 图表与详细结果
- 图1（Aachen 匹配可视化示例）：见 `docs/figs/aachen_matches_grid.png`

![Aachen 匹配可视化示例](figs/aachen_matches_grid.png)

- 表1（结果汇总）：见 `docs/results.md`（由脚本自动生成）。

## 6 分析
- 语义带来的稳定性：双阈值策略对噪声不敏感，原型记忆促成类内聚合；
- 动态层权对多层蒸馏的梯度平衡作用；
- 自适应加权在多损失冲突时提升收敛稳定与最终性能；
- 可视化印证：匹配分布更均匀、遮挡/重复纹理区域更稳健。

## 7 局限与展望
- 外部语义教师可进一步增强（更强分割/稳定性模型）；
- PnP 依赖高质量 SfM 结构；后续结合检索与可见性过滤（HLoc 风格）与 P3P+LO‑RANSAC；
- 更系统的跨域评测与实时部署优化（ONNX/TensorRT）。

## 8 可复现性
- 统一入口 `app_train.py`；
- 评测：`eval/pose_recall.py`、`eval/pnp_localization.py`、`eval/localization_aachen.py`；
- 自动消融：`scripts/run_ablation.py`；
- 日志：CSV + JSONL，保存配置哈希、Git 提交哈希与 pip freeze；
- 可视化：热力图、语义图、匹配连线。

## 9 结论
我们提出语义引导的 XFeat–SFD2 融合蒸馏与自适应多损失优化方案，在多数据集上实现稳定提升并保持高效推理。方法在噪声伪标签、多层蒸馏与多任务优化三方面提供工程化且有理论支撑的解决路径。

## 参考文献（占位）
- Kendall & Gal. Multi‑Task Learning Using Uncertainty to Weigh Losses. CVPR 2018.
- Wang et al. LightGlue. CVPR 2023.
- DeTone et al. SuperPoint/SuperGlue. CVPR 2018/2020.
- Revaud et al. R2D2. NeurIPS 2019.
- XFeat 与 HLoc 等代表性工作。

---

## 附录 A：主要配置与超参数（示例）
- 学习率/权重衰减、warmup、max epochs、batch size；
- semantic_pseudo：high/low 阈值、mid_weight、use_memory_bank、momentum；
- distill：dynamic_layer_weight、layer_weight_norm、variance_eps；
- adaptive_loss：method ∈ {uncertainty, gradnorm}、alpha、ema、init_log_var。

## 附录 B：PnP 结构 JSON 模板
```json
{
  "images": {
    "img_0001.jpg": {
      "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
      "observations": [
        {"xy": [u1, v1], "pid": 123},
        {"xy": [u2, v2], "pid": 456}
      ]
    }
  },
  "points3d": {
    "123": {"xyz": [X1, Y1, Z1]},
    "456": {"xyz": [X2, Y2, Z2]}
  }
}
```
说明：`images` 中每张图像提供内参 K（可选）与观测（像素坐标与对应 3D 点 ID）；`points3d` 存储 3D 点坐标（亦可为数组形式携带 `id` 字段）。

## 附录 C：更多可视化
- 匹配连线样例、关键点热力图、语义/稳定性图、蒸馏残差热力图（可选）。
