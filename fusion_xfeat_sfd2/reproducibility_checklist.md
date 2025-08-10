# Reproducibility Checklist

## 1. Environment
- [x] 提供 `environment.yml` & `requirements.txt`
- [ ] 记录 `pip freeze` 输出
- [ ] 记录 GPU / Driver / CUDA 版本

## 2. Randomness Control
- [x] 统一 `seed` 写入日志 (`utils/seed.py`)
- [x] `cudnn.deterministic=True`, `benchmark=False`
- [ ] DataLoader `worker_init_fn` / `generator` 固定

## 3. Data
- [x] HPatches 加载 (真实单应性缩放)
- [x] MegaDepth pairs 加载 (pairs 文件)
- [x] Aachen 数据管线骨架
- [x] RobotCar 数据管线骨架
- [ ] 数据下载与完整性校验脚本 (sha256)
- [ ] 固定拆分列表存档 (`splits/*.txt`)

## 4. Code Versioning
- [x] 记录 git commit hash (计划: 训练时写入 results)
- [x] 配置文件 md5 (`utils/config.py`)
- [ ] 子模块/外部权重版本说明

## 5. Training Process
- [x] 阶段化 Trainer: pretrain / distill / semantic / finetune
- [x] 多视图一致性损失 (homography warp)
- [ ] 语义原型损失使用真实标签 (当前随机)
- [ ] 蒸馏教师替换为真实 SFD2
- [ ] 学习率调度器 (当前未实现 cosine)
- [ ] 日志记录 JSON 化

## 6. Evaluation
- [x] HPatches MMA @ (1,3,5) px
- [x] MegaDepth 匹配统计 (占位，后续替换真实几何评估)
- [ ] Aachen Pose Recall
- [ ] RobotCar 跨域定位
- [ ] 失败案例可视化 (匹配线, heatmap)

## 7. Ablation
- [ ] -L_struct / -L_sem / -AdapterAttention / +GradNorm
- [ ] 不同 adapter 模式 (conv|film|cross_attn)

## 8. Artifact Archival
- [ ] 训练权重分阶段保存 (epoch_N.pt, model_best.pt)
- [ ] 结果表格 `results/exp_log.csv` 自动追加
- [ ] 评测指标 JSON (含 config hash, git hash)

## 9. 文档
- [x] README 实验框架
- [ ] 数据准备指南 `datasets_prepare.md`
- [ ] FAQ (显存不足 / 导入失败)

## 10. 自动化脚本
- [ ] 一键运行全流程 `scripts/run_all_ablation.sh`
- [ ] 指标汇总脚本 `scripts/collect_results.py`

## 11. 可复现关键哈希
- 配置 md5: `config_hash.txt`
- Git commit: 训练启动时记录
- 环境锁: `environment.yml` + `pip freeze > results/pip_freeze.txt`

## 12. 待办优先级路线
1. 真正 SFD2 教师接入 + 语义标签解析
2. 几何精评 (本质矩阵 / Pose) 替换 MegaDepth 占位指标
3. Ablation 自动脚本 + 指标 JSON 化
4. 数据下载/校验与 split 固定
5. 可视化与论文图生成工具
