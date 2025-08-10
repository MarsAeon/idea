# XFeat + SFD2 融合方案 (Draft)

本目录用于撰写与实现一篇将局部特征/匹配框架 XFeat 与结构化特征蒸馏/定位框架 SFD2 融合的新论文原型，强调可复现性。此文档给出：
1. 论文目标与核心贡献设计
2. 模块融合技术路线
3. 训练与评测协议（可复现性保障）
4. 代码结构规划
5. 实验清单 & 结果记录模板

---
## 1. 论文暂定题目与核心想法
**暂定题目**: Semantic-Guided Hybrid Local Feature Framework: Fusing XFeat with Structured Feature Distillation for Robust Visual Localization

**核心问题**: 现有局部特征（如 XFeat）在跨域/光照/尺度变化下的稳健性仍受限；SFD2 的结构化蒸馏能提升定位鲁棒性但未结合新型高效特征抽取器。我们提出一个语义/结构引导的混合特征学习框架，将 XFeat 的高效可重构特征编码与 SFD2 的分层蒸馏 + 语义引导监督联合，提升长距离、跨域定位与匹配性能。

**拟贡献点**:
1. 提出一个双路径 (Appearance + Structural/Semantic) 的特征流水线：XFeat 作为基础特征主干 + 结构蒸馏侧路。
2. 设计语义-结构一致性损失：结合语义掩码/类别中心与几何重投影一致性。
3. 引入跨域自适应蒸馏：教师模型在多域(白天/夜晚/天气)提取结构提示，学生共享XFeat低层特征实现适配。
4. 统一评测协议：Aachen Day-Night / RobotCar Seasons / MegaDepth / HPatches (匹配与定位) + Ablation 全公开脚本。
5. 高复现性基线对齐：逐步替换策略 + 配套 seed / 环境锁定 / 下载脚本。

---
## 2. 模块融合技术路线
```
Input Image
   │
   ├── XFeat Backbone (改造: 输出 {keypoints, descriptors, scores, multi-scale feats})
   │        │
   │        ├─> Local Descriptor Head (保留)
   │        └─> 中层特征缓存 F_mid
   │
   ├── 结构/语义教师 (SFD2 Teacher or Pretrained Net + 语义分割模型)
   │        └─> 结构提示 S_feat, 关键点热力 / dense map / 语义掩码
   │
   ├── 融合适配模块 (Adapter / FiLM / Attention Cross-Gating)
   │        └─> 融合表征 F_fused
   │
   ├── 匹配头 (可接 LighterGlue / SuperGlue 接口)
   │
   └── 损失:
       L = L_desc + L_det + L_struct + L_sem + L_consistency + L_distill
```

### 2.1 关键设计点
- Feature Adapter: 1x1 conv + LayerNorm + 可选 Cross-Attention，将结构提示对齐到 XFeat 特征尺度。
- 语义掩码加权：在关键点检测热力图损失中对语义稳定区域提升权重 (例如建筑/路标 vs 天空/动态物体)。
- 结构蒸馏：对教师 dense feature map 做 channel-wise 分组蒸馏，对学生 F_mid 对应层施加 KL / L2。
- 跨尺度一致性：XFeat 多尺度金字塔特征与重投影 (同一3D点多视图) 一致性损失。

### 2.2 损失项草案
- L_det: 关键点检测 (简单热力图 BCE / Focal 或 Cornerness Ranking)
- L_desc: 正负样本对比 (InfoNCE / Hard Negative Mining)
- L_struct: 教师结构图 distill L2 / SmoothL1
- L_sem: 语义类别中心对齐 (聚合特征 -> 原型, 原型间 margin)
- L_consistency: 多视图重投影特征一致性
- L_distill: 统一蒸馏 (logits 或 feature embedding 级)

---
## 3. 训练与评测协议（可复现性保障）
### 3.1 环境管理
- 提供 `environment.yml` (conda) 与 `requirements.txt`
- 固定主要库版本: torch, torchvision, numpy, opencv, kornia
- 统一随机种子: torch, numpy, random, cudnn deterministic
- 下载脚本: `scripts/download_datasets.sh` / `datasets_prepare.md`

### 3.2 数据集
| 任务 | 数据集 | 用途 | 指标 |
|------|--------|------|------|
| 局部特征匹配 | HPatches / MegaDepth pairs | 训练+验证 | MMA @ 1,3,5 px |
| 定位 | Aachen Day-Night / RobotCar / InLoc | 评测 | Pose Recall @ (0.25m,2°, ... ) |
| 跨域鲁棒 | RobotCar Seasons | 验证 | 成功率 |
| 消融 | 子集抽样 (固定list) | 分析 | 各损失贡献 |

### 3.3 训练阶段划分
1. 预训练阶段: 仅 L_det + L_desc (XFeat 原生)
2. 蒸馏阶段: 加入 L_struct + L_distill
3. 语义融合阶段: 引入 L_sem + 重投影一致性
4. 统一微调: 全部损失权重调和 (自动权重 / GradNorm 可选)

### 3.4 评测脚本
- `eval/hpatches_eval.py`
- `eval/megadepth_matching.py`
- `eval/localization_aachen.py`
- 统一输出 JSON: 指标 + 配置哈希 + git commit
- 保存可视化: top-K 匹配, 失败案例, 定位轨迹

### 3.5 结果追踪
- `results/exp_log.csv` (追加行)
- TensorBoard 或 Weights&Biases 可选 (开关)
- `reproducibility_checklist.md`：列出环境/随机性/数据/脚本/哈希

---
## 4. 代码结构规划
```
fusion_xfeat_sfd2/
  README.md
  requirements.txt (或软链接)
  environment.yml
  configs/
    base.yaml
    pretrain.yaml
    distill.yaml
    semantic.yaml
    finetune.yaml
  models/
    __init__.py
    backbone_xfeat_wrapper.py
    teacher_sfd2_wrapper.py
    fusion_adapter.py
    heads/
      keypoint_head.py
      descriptor_head.py
      matcher_head.py
  losses/
    __init__.py
    detection_loss.py
    descriptor_loss.py
    structural_distill_loss.py
    semantic_loss.py
    consistency_loss.py
  datasets/
    __init__.py
    hpatches.py
    megadepth_pairs.py
    aachen.py
    robotcar.py
    transforms.py
  trainers/
    __init__.py
    base_trainer.py
    stage_pretrain.py
    stage_distill.py
    stage_semantic.py
    stage_finetune.py
  eval/
    hpatches_eval.py
    megadepth_matching.py
    localization_aachen.py
  scripts/
    download_datasets.sh
    prepare_hpatches.py
    prepare_megadepth.py
    run_all_ablation.sh
  experiments/
    logs/
    outputs/
  results/
    exp_log.csv
  utils/
    seed.py
    distributed.py
    config.py
    checkpoint.py
    geometry.py
    visualization.py
  tests/
    test_forward.py
    test_losses.py
```

---
## 5. 实验清单 & 模板
### 5.1 配置哈希
对每个 YAML 配置文件做 md5 -> 记录在日志中。

### 5.2 Ablation 表 (示例)
| ID | 配置 | 去除项 | HPatches MMA@3px | Aachen @0.5m2° | 备注 |
|----|------|--------|------------------|-----------------|------|
| A0 | base | - |  |  | baseline |
| A1 | base | -L_struct |  |  |  |
| A2 | base | -L_sem |  |  |  |
| A3 | base | -Adapter Attention |  |  |  |
| A4 | base | +GradNorm |  |  |  |

### 5.3 Repro Checklist
见 `reproducibility_checklist.md`。

---
## 6. 近期实施里程碑
| 周次 | 目标 | 产出 |
|------|------|------|
| W1 | 结构搭建 + Wrapper | 代码骨架 & 假数据跑通 | 
| W2 | 预训练阶段复现 XFeat | 预训练指标曲线 | 
| W3 | 蒸馏阶段实现 & 训练 | 蒸馏提升报告 | 
| W4 | 语义阶段 + 定位评测 | 中期结果 | 
| W5 | 全量微调 + Ablation | 完整表格 | 
| W6 | 论文初稿撰写 | draft v1 | 

---
## 7. 后续文件待补
- environment.yml
- configs/*.yaml
- 各模块代码与测试

如需继续，下一步建议：生成 `environment.yml` 与基础 `configs/base.yaml`，以及 `utils/seed.py` 等初始文件。

---
## 8. 当前实现进度 (Running Status)
| 模块 | 状态 | 说明 |
|------|------|------|
| XFeat BackBone 包装 | ✅ | 使用真实推理模型, 提供稀疏+密集输出 |
| SFD2 Teacher 占位 | ✅ | 多层特征+结构/语义logits，占位 CNN，可替换真实结构蒸馏网络 |
| Adapter (conv_attn) | ✅ | 支持 conv_attn/film 等扩展接口（暂未全部实现） |
| 数据集 HPatches | ✅ | 真实序列/单应读取，MMA 评测支持 |
| 数据集 MegaDepth pairs | ✅ | 基于 pairs 文件加载，后续需加入几何GT评估 |
| 数据集 Aachen/RobotCar | ✅ | 骨架与元数据解析（姿态/内参若提供） |
| 损失: 检测/描述/结构/一致性 | ✅ | 一致性基于 homography grid sample |
| 语义: 原型+蒸馏 | ✅ | 语义原型随机标签（已接入教师伪标签生成） |
| 多视图一致性 | ✅ | 使用 ref->tgt homography 对齐 dense desc |
| 训练阶段流水 | ✅ | pretrain/distill/semantic/finetune 四阶段脚本 |
| 日志/复现 | ✅ | git hash + config hash + pip freeze 输出 & CSV 指标 |
| 评测 HPatches/MegaDepth | ✅ | MMA / 占位匹配精度；后续补 pose/几何约束 |
| 配置与权重管理 | ✅ | base.yaml 添加语义蒸馏权重与 eval 切换 |
| 学习率调度 | ✅ | cosine + warmup 已接入公共基类 |
| Aachen/RobotCar Pose 评测 | ⏳ | Aachen 占位 PnP，需真实2D-3D映射 |
| 可视化工具 | ⏳ | 待实现匹配线/heatmap/语义mask叠加 |
| Ablation 自动脚本 | ⏳ | 待实现 run_all_ablation.sh & 汇总脚本 |

---
## 9. 使用指南
### 9.1 环境创建
```bash
conda env create -f fusion_xfeat_sfd2/environment.yml
conda activate fusion_xfeat_sfd2
# 或使用 pip 安装 requirements.txt
```

### 9.2 数据准备概述
```
/path/to/datasets/
  hpatches_sequences/  (HPatches 原始解压)
  megadepth/
    images/
    pairs/train_pairs.txt
    pairs/val_pairs.txt
  aachen/
    images/images_upright/... (db, query)
    split.json (可选)
    poses/poses.json (可选)
  robotcar/
    images/route/condition/*.jpg
    poses.json (可选)
```
后续将补充 `scripts/prepare_*.py` 完整自动化。

### 9.3 预训练
```bash
python -m fusion_xfeat_sfd2.app_train --stage pretrain --config fusion_xfeat_sfd2/configs/base.yaml \
  --pretrained accelerated_features/weights/xfeat.pt
```

### 9.4 蒸馏
```bash
python -m fusion_xfeat_sfd2.app_train --stage distill --config fusion_xfeat_sfd2/configs/base.yaml \
  --pretrained accelerated_features/weights/xfeat.pt
```

### 9.5 语义阶段 & 微调
```bash
python -m fusion_xfeat_sfd2.app_train --stage semantic --config fusion_xfeat_sfd2/configs/base.yaml
python -m fusion_xfeat_sfd2.app_train --stage finetune --config fusion_xfeat_sfd2/configs/base.yaml
```

### 9.6 评测
```bash
# HPatches / MegaDepth (依据 val.name)
python -m fusion_xfeat_sfd2.app_train --stage eval --config fusion_xfeat_sfd2/configs/base.yaml
```
输出指标自动写入 CSV: `fusion_xfeat_sfd2/results/exp_log.csv`

### 9.7 指标 CSV 示例
```
datetime,stage,config_hash,git_hash,epoch,metric,value,notes
2025-08-08 12:00:00,pretrain,ab12ef34,deadbeef,1,det_loss,0.543210,
```

---
## 10. 后续计划细化
| 优先级 | 项目 | 动作 | 验证方式 |
|--------|------|------|----------|
| P0 | 真实 SFD2 教师接入 | 导入训练权重/特征接口 | distill loss 曲线下降 & 结构图可视化 |
| P0 | 语义标签接入 | Cityscapes / Mapillary 预训练语义；生成伪标签 (已完成基础 image-level 伪标签) | 语义蒸馏 loss 收敛 |
| P1 | Pose 评测 | 添加 PnP + RANSAC (OpenCV) 求位姿 | Aachen/RobotCar Recall 表 |
| P1 | 几何精评 | MegaDepth 基于深度重投影匹配准确率 | 与占位指标对比提升 |
| P1 | 可视化 | 保存匹配连线 & heatmap & 语义 overlay | README 图示 |
| P2 | Ablation 自动化 | run_all_ablation.sh & collect_results.py | 生成表格一致 |
| P2 | Scheduler 实装 | CosineAnnealing + warmup 已完成 | loss 平滑下降 |
| P2 | GradNorm 权重自适应 | 接入梯度归一化 | 各损失权重动态曲线 |
| P3 | 模型导出 | torchscript / onnx | 推理速度基准 |

---
## 11. 真实 SFD2 接口接入指引 (草案)
1. 将 SFD2 代码加入 `sfd2/`（已存在）并暴露 `build_teacher()` 返回多层特征。  
2. 修改 `teacher_sfd2_wrapper.py`：用真实主干替换轻量 enc1-enc4；保持输出键 `feat` / `struct` / `semantic_logit` / `feat_levels`。  
3. 在 `finetune` 与 `distill` 阶段添加结构分层蒸馏 (例如不同层加权 L2)。  
4. 若语义 logits 尺寸与学生不一致，增加上采样对齐 (bilinear) 再蒸馏。  

---
## 12. 语义标签集成计划
- 方案A (直接监督): 若目标数据有语义标注，替换 `semantic_proto_loss` 中随机标签。  
- 方案B (伪标签): 用教师 `semantic_logit` 最大概率类别作为标签，再加置信度阈值筛选 (-1 忽略)。  
- 方案C (外部分割器): 使用预训练分割模型（如 SegFormer）生成稳定类别 (building / road / vegetation / object) 作为 mask 权重。  

`semantic_proto_loss` 调整：
```
labels = teacher_logits.argmax(1)  # (B,H,W)
conf = teacher_logits.softmax(1).max(1).values
labels[conf < tau] = -1
# 池化 -> token / prototype 前 flatten 有效像素
```

---
## 13. 几何精评扩展
1. MegaDepth：使用深度 + 相机位姿构建 3D 点，经对图投影 -> 真实对应匹配内点率。  
2. Aachen/RobotCar：抽取稀疏匹配 -> 基于内参估算 Essential / PnP -> 位姿误差统计 (Recall @ (0.25m,2°) 等)。  
3. 多视图一致性：引入时间序列 / 不同光照条件图像集，统计特征点重复率与匹配保留率。  

---
## 14. 可视化规划
| 类型 | 内容 | 工具 |
|------|------|------|
| 匹配连线 | top-K 稀疏匹配 | OpenCV linePoly / matplotlib |
| Heatmap | det & struct map 叠加 | matplotlib jet overlay |
| 语义叠加 | semantic argmax mask | color map + alpha |
| 消融对比 | loss & metric 曲线 | tensorboard / seaborn |

---
## 15. 论文写作大纲 (简版)
1. 引言：跨域特征鲁棒性问题 & 贡献点列表  
2. 相关工作：局部特征 (SIFT->现代轻量) / 蒸馏 / 语义引导 / 定位  
3. 方法：体系结构、适配器、损失（结构+语义+一致性）、训练策略  
4. 实验：数据集、实现细节、主结果表、消融、效率分析、可视化  
5. 讨论：局限（语义噪声、动态物体）、未来工作  
6. 结论：总结与扩展方向  

---
## 16. 快速 Todo 列表 (开发视角)
- [ ] 接入真实 SFD2 backbone
- [x] 语义伪标签生成逻辑 & 阈值配置 (image-level)
- [ ] scheduler 实际调用 & warmup 实装
- [ ] MegaDepth 几何 GT 评测
- [ ] PnP / Pose 评测脚本 (Aachen/RobotCar)
- [ ] 可视化工具 `utils/visualization.py`
- [ ] ablation 脚本 & JSON 指标导出
- [ ] 统一 CLI: train / eval / visualize 子命令

---
## 17. License & Citation (占位)
后续补充原始项目 (XFeat / SFD2 / 语义分割模型) License 引用与论文 BibTeX。

---
## 18. 变更日志 (Changelog)
| 日期 | 更新 | 描述 |
|------|------|------|
| 2025-08-08 | 初始骨架 | 构建项目结构与预训练/蒸馏流水线 |
| 2025-08-08 | 数据集扩展 | HPatches/MegaDepth/Aachen/RobotCar 骨架 |
| 2025-08-08 | 语义蒸馏 | 加入 semantic_distill_loss 与权重 |
| 2025-08-08 | 一致性 | Homography 基于 dense desc 一致性损失 |
| 2025-08-08 | 调度器 & Pose 占位 | 添加 cosine 调度 + Aachen PnP 占位 |
| 2025-08-09 | 真实 SFD2 教师接入 | 使用 ResSegNetV2 + 多层 & 描述符蒸馏 |

---
(文档将随实现进度持续更新)

### 追加：Ablation 自动化使用

生成并执行（默认 finetune 阶段）：
```
python fusion_xfeat_sfd2/scripts/run_ablation.py --stage finetune
```
仅生成配置：
```
python fusion_xfeat_sfd2/scripts/run_ablation.py --generate_only
```
干跑显示命令：
```
python fusion_xfeat_sfd2/scripts/run_ablation.py --dry
```
汇总特定指标（已写入 exp_log.csv 后自动打印排序列表）：
```
python fusion_xfeat_sfd2/scripts/run_ablation.py --metric_stage finetune --metric_name struct_loss --generate_only
```
生成的变体位于 `configs/ablation/` 目录，命名格式：`<tag>_<hash>.yaml`。
