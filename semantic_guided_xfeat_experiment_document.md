# 语义引导的轻量级特征匹配网络实验文档

## 实验概述

### 实验目标
验证语义引导的轻量级特征匹配网络(SG-XFeat)在保持轻量化特性的同时，能够显著提升特征匹配的准确性和鲁棒性。

### 实验假设
1. **语义引导机制**能够有效提升特征检测的质量和判别力
2. **语义增强描述符**能够减少语义不一致的误匹配
3. **端到端训练**能够优化整个匹配流程的性能
4. **轻量化设计**能够在资源受限设备上实现实时推理

## 实验设计

### 实验1：语义引导特征检测器效果验证

#### 1.1 实验目的
验证语义注意力机制对特征检测质量的影响

#### 1.2 实验设置
- **数据集**：MegaDepth-1500, ScanNet-1500
- **对比方法**：
  - XFeat (原始)
  - XFeat + 语义注意力 (我们的方法)
- **评估指标**：
  - 特征点重复性 (Repeatability)
  - 特征点分布均匀性 (Distribution Uniformity)
  - 语义区域覆盖率 (Semantic Coverage)

#### 1.3 实验步骤
```python
# 实验代码示例
def evaluate_feature_detection():
    # 1. 加载预训练模型
    xfeat_original = XFeat()
    xfeat_semantic = SemanticGuidedXFeat()
    
    # 2. 测试数据准备
    test_loader = get_test_loader('megadepth1500')
    
    results = {'original': [], 'semantic': []}
    
    for img_pair in test_loader:
        img1, img2, gt_matches = img_pair
        
        # 3. 特征检测
        kpts1_orig, desc1_orig = xfeat_original.detectAndCompute(img1)
        kpts1_sem, desc1_sem = xfeat_semantic.detectAndCompute(img1)
        
        # 4. 计算评估指标
        repeatability_orig = compute_repeatability(kpts1_orig, kpts2_orig, gt_homography)
        repeatability_sem = compute_repeatability(kpts1_sem, kpts2_sem, gt_homography)
        
        results['original'].append(repeatability_orig)
        results['semantic'].append(repeatability_sem)
    
    return results
```

#### 1.4 预期结果
- 语义引导方法在重复性上提升5-15%
- 语义重要区域的特征点密度增加20-30%
- 背景噪声区域的特征点减少40-50%

### 实验2：语义增强描述符性能评估

#### 2.1 实验目的
评估语义信息对特征描述符判别力的提升效果

#### 2.2 实验设置
- **数据集**：HPatches (视角变化、光照变化子集)
- **对比方法**：
  - XFeat描述符
  - XFeat + 语义增强描述符
- **评估指标**：
  - 描述符匹配准确率 (Matching Accuracy)
  - 语义一致性分数 (Semantic Consistency)
  - ROC曲线下面积 (AUC)

#### 2.3 实验步骤
```python
def evaluate_descriptor_performance():
    # 1. 准备测试数据
    hpatches_loader = HPatchesDataset(split='test')
    
    # 2. 模型准备
    descriptor_original = XFeatDescriptor()
    descriptor_semantic = SemanticEnhancedDescriptor()
    
    metrics = defaultdict(list)
    
    for sequence in hpatches_loader:
        ref_img, test_imgs, gt_correspondences = sequence
        
        # 3. 提取描述符
        ref_desc_orig = descriptor_original(ref_img)
        ref_desc_sem = descriptor_semantic(ref_img, ref_semantic)
        
        for test_img, gt_corr in zip(test_imgs, gt_correspondences):
            test_desc_orig = descriptor_original(test_img)
            test_desc_sem = descriptor_semantic(test_img, test_semantic)
            
            # 4. 计算匹配性能
            matches_orig = match_descriptors(ref_desc_orig, test_desc_orig)
            matches_sem = match_descriptors(ref_desc_sem, test_desc_sem)
            
            # 5. 评估指标计算
            accuracy_orig = compute_matching_accuracy(matches_orig, gt_corr)
            accuracy_sem = compute_matching_accuracy(matches_sem, gt_corr)
            
            semantic_consistency = compute_semantic_consistency(
                matches_sem, ref_semantic, test_semantic)
            
            metrics['accuracy_original'].append(accuracy_orig)
            metrics['accuracy_semantic'].append(accuracy_sem)
            metrics['semantic_consistency'].append(semantic_consistency)
    
    return metrics
```

#### 2.4 预期结果
- 描述符匹配准确率提升8-12%
- 语义一致性分数达到85%以上
- 在光照变化场景下性能提升15-20%

### 实验3：端到端匹配性能评估

#### 3.1 实验目的
评估完整的端到端语义引导匹配网络的整体性能

#### 3.2 实验设置
- **数据集**：
  - MegaDepth-1500 (相机位姿估计)
  - ScanNet-1500 (室内场景匹配)
  - YFCC100M子集 (大规模外场景)
- **对比方法**：
  - XFeat
  - SuperPoint + LightGlue
  - ALIKE + LightGlue
  - SFD2
  - SG-XFeat (我们的方法)

#### 3.3 实验步骤
```python
def evaluate_end_to_end_matching():
    # 1. 数据集准备
    datasets = {
        'megadepth': MegaDepth1500(),
        'scannet': ScanNet1500(),
        'yfcc': YFCC100MSubset()
    }
    
    # 2. 方法准备
    methods = {
        'xfeat': XFeat(),
        'superpoint_lg': SuperPointLightGlue(),
        'alike_lg': ALIKELightGlue(),
        'sfd2': SFD2(),
        'sg_xfeat': SemanticGuidedXFeat()
    }
    
    results = defaultdict(dict)
    
    for dataset_name, dataset in datasets.items():
        for method_name, method in methods.items():
            print(f"Evaluating {method_name} on {dataset_name}")
            
            # 3. 运行评估
            metrics = run_pose_estimation_benchmark(method, dataset)
            results[dataset_name][method_name] = metrics
            
            # 4. 计算AUC指标
            pose_errors = metrics['pose_errors']
            auc_5 = pose_auc(pose_errors, [5])
            auc_10 = pose_auc(pose_errors, [10])
            auc_20 = pose_auc(pose_errors, [20])
            
            results[dataset_name][method_name].update({
                'auc@5': auc_5,
                'auc@10': auc_10,
                'auc@20': auc_20
            })
    
    return results

def run_pose_estimation_benchmark(method, dataset):
    pose_errors = []
    inlier_counts = []
    processing_times = []
    
    for img_pair in dataset:
        img1, img2, K1, K2, gt_pose = img_pair
        
        # 特征匹配
        start_time = time.time()
        mkpts1, mkpts2 = method.match(img1, img2)
        processing_time = time.time() - start_time
        
        # 位姿估计
        if len(mkpts1) >= 8:
            pose, inliers = estimate_pose_ransac(
                mkpts1, mkpts2, K1, K2, threshold=1.0)
            
            # 计算位姿误差
            pose_error = compute_pose_error(pose, gt_pose)
            pose_errors.append(pose_error)
            inlier_counts.append(len(inliers))
        else:
            pose_errors.append(float('inf'))
            inlier_counts.append(0)
        
        processing_times.append(processing_time)
    
    return {
        'pose_errors': pose_errors,
        'inlier_counts': inlier_counts,
        'processing_times': processing_times,
        'mean_inliers': np.mean(inlier_counts),
        'mean_time': np.mean(processing_times)
    }
```

#### 3.4 预期结果
- **MegaDepth-1500**: AUC@5°提升5-8%, AUC@10°提升3-6%
- **ScanNet-1500**: Pose accuracy提升8-15%
- **YFCC100M**: 大规模场景下的鲁棒性提升10-20%

### 实验4：轻量化和效率评估

#### 4.1 实验目的
验证方法在保持轻量化特性的同时实现性能提升

#### 4.2 实验设置
- **硬件平台**：
  - NVIDIA Jetson Xavier NX
  - NVIDIA Jetson Orin Nano
  - Intel NUC (CPU only)
  - iPhone 13 Pro (通过ONNX转换)
- **评估指标**：
  - 推理时间 (ms)
  - 内存占用 (MB)
  - 功耗 (W)
  - 模型大小 (MB)
  - FLOPs

#### 4.3 实验步骤
```python
def evaluate_efficiency():
    # 1. 模型准备
    models = {
        'xfeat': XFeat(),
        'sg_xfeat_lite': SemanticGuidedXFeat(config='lightweight'),
        'sg_xfeat_full': SemanticGuidedXFeat(config='accurate')
    }
    
    # 2. 测试数据
    test_resolutions = [(480, 640), (720, 1280), (1080, 1920)]
    
    results = defaultdict(dict)
    
    for model_name, model in models.items():
        print(f"Evaluating efficiency of {model_name}")
        
        # 3. 模型大小和FLOPs
        model_size = get_model_size(model)
        
        for resolution in test_resolutions:
            h, w = resolution
            dummy_input = torch.randn(1, 3, h, w)
            
            # FLOPs计算
            flops = compute_flops(model, dummy_input)
            
            # 推理时间测试
            inference_times = []
            memory_usage = []
            
            for _ in range(100):  # 预热
                with torch.no_grad():
                    _ = model(dummy_input)
            
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            for _ in range(200):  # 正式测试
                torch.cuda.synchronize()
                start_time = time.time()
                
                with torch.no_grad():
                    output = model(dummy_input)
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                inference_times.append((end_time - start_time) * 1000)
                memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)
            
            results[model_name][f'{h}x{w}'] = {
                'model_size_mb': model_size,
                'flops_g': flops / 1e9,
                'inference_time_ms': {
                    'mean': np.mean(inference_times),
                    'std': np.std(inference_times),
                    'median': np.median(inference_times)
                },
                'memory_mb': {
                    'mean': np.mean(memory_usage),
                    'max': np.max(memory_usage)
                }
            }
    
    return results
```

#### 4.4 预期结果
- 模型大小增加不超过50%
- 推理时间增加不超过30%
- 在Jetson平台实现>15 FPS (VGA分辨率)
- 内存占用控制在200MB以内

### 实验5：消融实验

#### 5.1 实验目的
分析各个组件对整体性能的贡献

#### 5.2 消融实验设计
```python
def ablation_study():
    # 消融实验配置
    ablation_configs = {
        'baseline': {
            'semantic_attention': False,
            'semantic_descriptor': False,
            'semantic_matching': False
        },
        'semantic_attention_only': {
            'semantic_attention': True,
            'semantic_descriptor': False,
            'semantic_matching': False
        },
        'semantic_descriptor_only': {
            'semantic_attention': False,
            'semantic_descriptor': True,
            'semantic_matching': False
        },
        'semantic_matching_only': {
            'semantic_attention': False,
            'semantic_descriptor': False,
            'semantic_matching': True
        },
        'attention_descriptor': {
            'semantic_attention': True,
            'semantic_descriptor': True,
            'semantic_matching': False
        },
        'full_method': {
            'semantic_attention': True,
            'semantic_descriptor': True,
            'semantic_matching': True
        }
    }
    
    results = {}
    dataset = MegaDepth1500()
    
    for config_name, config in ablation_configs.items():
        print(f"Testing configuration: {config_name}")
        
        # 构建模型
        model = SemanticGuidedXFeat(config)
        
        # 评估性能
        metrics = evaluate_matching_performance(model, dataset)
        results[config_name] = metrics
    
    return results
```

#### 5.3 预期发现
- 语义注意力机制贡献最大 (+3-5% AUC)
- 语义描述符提升中等 (+2-3% AUC)
- 语义匹配约束效果显著 (+2-4% AUC)
- 组合效果具有协同作用

### 实验6：鲁棒性测试

#### 6.1 实验目的
测试方法在各种挑战性条件下的鲁棒性

#### 6.2 实验设置
```python
def robustness_evaluation():
    # 挑战性条件
    test_conditions = {
        'illumination': ['bright', 'dark', 'shadows'],
        'weather': ['sunny', 'cloudy', 'rainy', 'snowy'],
        'viewpoint': ['small_angle', 'medium_angle', 'large_angle'],
        'scale': ['zoom_in', 'zoom_out', 'multi_scale'],
        'blur': ['motion_blur', 'gaussian_blur', 'no_blur'],
        'noise': ['low_noise', 'medium_noise', 'high_noise']
    }
    
    # 数据集准备
    datasets = prepare_challenging_datasets(test_conditions)
    
    methods = {
        'xfeat': XFeat(),
        'sg_xfeat': SemanticGuidedXFeat()
    }
    
    results = defaultdict(dict)
    
    for condition_type, conditions in test_conditions.items():
        for condition in conditions:
            print(f"Testing {condition_type}: {condition}")
            
            dataset = datasets[condition_type][condition]
            
            for method_name, method in methods.items():
                metrics = evaluate_matching_performance(method, dataset)
                results[f"{condition_type}_{condition}"][method_name] = metrics
    
    return results
```

#### 6.3 预期结果
- 在光照变化条件下性能提升15-25%
- 在大视角变化下性能提升10-20%
- 在噪声环境下保持稳定性能
- 在模糊图像中匹配成功率提升20-30%

## 实验数据和复现性

### 数据集准备

#### 标准数据集下载
```bash
# MegaDepth-1500
python -m modules.dataset.download --megadepth-1500 --download_dir ./data/MegaDepth1500

# ScanNet-1500
python download_scannet1500.py --output_dir ./data/ScanNet1500

# HPatches
wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
tar -xzf hpatches-sequences-release.tar.gz -C ./data/

# YFCC100M子集 (用于大规模测试)
python download_yfcc_subset.py --size 10000 --output_dir ./data/YFCC100M_subset
```

#### 语义标注数据准备
```python
def prepare_semantic_annotations():
    """
    为匹配数据集准备语义标注
    """
    # 使用预训练的语义分割模型生成伪标签
    semantic_models = {
        'deeplabv3': load_deeplabv3_model(),
        'segformer': load_segformer_model(),
        'convnext': load_convnext_model()
    }
    
    datasets = ['megadepth1500', 'scannet1500', 'hpatches']
    
    for dataset_name in datasets:
        print(f"Generating semantic annotations for {dataset_name}")
        
        dataset_path = f"./data/{dataset_name}"
        semantic_output_path = f"./data/{dataset_name}_semantic"
        
        generate_semantic_labels(
            dataset_path, 
            semantic_output_path, 
            semantic_models['deeplabv3']
        )
```

### 实验环境配置

#### Docker环境
```dockerfile
# Dockerfile
FROM pytorch/pytorch:1.13-cuda11.6-cudnn8-devel

# 安装依赖
RUN pip install opencv-python segmentation-models-pytorch kornia einops

# 复制代码
COPY . /workspace/semantic_guided_xfeat
WORKDIR /workspace/semantic_guided_xfeat

# 设置环境变量
ENV PYTHONPATH=/workspace/semantic_guided_xfeat
```

#### 依赖环境文件
```yaml
# environment.yml
name: sg_xfeat
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.8
  - pytorch=1.13
  - torchvision=0.14
  - cudatoolkit=11.6
  - pip
  - pip:
    - opencv-python>=4.5.0
    - segmentation-models-pytorch>=0.2.0
    - kornia>=0.6.0
    - einops>=0.4.0
    - matplotlib>=3.5.0
    - seaborn>=0.11.0
    - wandb>=0.12.0
    - tensorboard>=2.8.0
    - h5py>=3.6.0
    - tqdm>=4.62.0
    - pyyaml>=6.0
```

### 训练配置

#### 超参数配置
```yaml
# configs/training_config.yaml
model:
  backbone: "xfeat"
  semantic_classes: 21
  descriptor_dim: 128
  semantic_dim: 32

training:
  batch_size: 8
  learning_rate: 3e-4
  weight_decay: 1e-4
  num_epochs: 50
  warmup_epochs: 5
  
  # 损失权重
  loss_weights:
    detection: 1.0
    description: 1.0
    semantic: 0.5
    matching: 1.0

  # 优化器
  optimizer: "adam"
  scheduler: "cosine"
  
  # 数据增强
  augmentation:
    random_crop: true
    color_jitter: true
    gaussian_blur: true
    random_rotation: 15

evaluation:
  # 评估间隔
  eval_every: 5
  # 保存最佳模型
  save_best: true
  # 早停
  early_stopping:
    patience: 10
    min_delta: 0.001
```

### 实验脚本

#### 训练脚本
```python
# train.py
import argparse
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 初始化训练器
    trainer = SemanticGuidedTrainer(config, args.output_dir)
    
    # 开始训练
    if args.resume:
        trainer.resume_training(args.resume)
    else:
        trainer.train()

if __name__ == "__main__":
    main()
```

#### 评估脚本
```python
# evaluate.py
import argparse
from evaluation.benchmark import run_all_benchmarks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--datasets', nargs='+', 
                       default=['megadepth1500', 'scannet1500', 'hpatches'])
    parser.add_argument('--output_dir', type=str, default='./evaluation_results')
    args = parser.parse_args()
    
    # 运行所有基准测试
    results = run_all_benchmarks(
        model_path=args.model_path,
        datasets=args.datasets,
        output_dir=args.output_dir
    )
    
    # 输出结果
    print("Evaluation Results:")
    for dataset, metrics in results.items():
        print(f"\n{dataset}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
```

### 可视化和分析

#### 结果可视化
```python
# visualization/plot_results.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_comparison_results(results_dict):
    """绘制方法对比结果"""
    # AUC对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # MegaDepth结果
    megadepth_data = prepare_megadepth_data(results_dict)
    sns.barplot(data=megadepth_data, x='Method', y='AUC@5', ax=axes[0,0])
    axes[0,0].set_title('MegaDepth-1500 AUC@5°')
    
    # ScanNet结果
    scannet_data = prepare_scannet_data(results_dict)
    sns.barplot(data=scannet_data, x='Method', y='Accuracy', ax=axes[0,1])
    axes[0,1].set_title('ScanNet-1500 Pose Accuracy')
    
    # 效率对比
    efficiency_data = prepare_efficiency_data(results_dict)
    sns.scatterplot(data=efficiency_data, x='Inference_Time', y='AUC@5', 
                   hue='Method', size='Model_Size', ax=axes[1,0])
    axes[1,0].set_title('Efficiency vs Performance')
    
    # 鲁棒性分析
    robustness_data = prepare_robustness_data(results_dict)
    sns.heatmap(robustness_data, annot=True, ax=axes[1,1])
    axes[1,1].set_title('Robustness Analysis')
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_ablation_results(ablation_results):
    """绘制消融实验结果"""
    df = pd.DataFrame(ablation_results).T
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    df.plot(kind='bar', ax=ax)
    ax.set_title('Ablation Study Results')
    ax.set_ylabel('AUC@5°')
    ax.legend(title='Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('ablation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
```

#### 定性结果分析
```python
# analysis/qualitative_analysis.py
def generate_matching_visualizations(model, test_pairs, output_dir):
    """生成匹配结果可视化"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (img1, img2, gt_matches) in enumerate(test_pairs[:50]):
        # 特征匹配
        output = model.match(img1, img2)
        mkpts1, mkpts2 = output['mkpts1'], output['mkpts2']
        
        # 绘制匹配结果
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        # 拼接图像
        combined_img = np.concatenate([img1, img2], axis=1)
        ax.imshow(combined_img)
        
        # 绘制匹配线
        for pt1, pt2 in zip(mkpts1, mkpts2):
            x1, y1 = pt1
            x2, y2 = pt2[0] + img1.shape[1], pt2[1]  # 偏移第二张图
            ax.plot([x1, x2], [y1, y2], 'g-', linewidth=0.5, alpha=0.7)
        
        # 绘制关键点
        ax.scatter(mkpts1[:, 0], mkpts1[:, 1], c='red', s=2)
        ax.scatter(mkpts2[:, 0] + img1.shape[1], mkpts2[:, 1], c='red', s=2)
        
        ax.set_title(f'Matching Result {i+1} - {len(mkpts1)} matches')
        ax.axis('off')
        
        plt.savefig(f'{output_dir}/match_{i+1:03d}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()

def analyze_failure_cases(model, difficult_pairs):
    """分析失败案例"""
    failure_analysis = {
        'low_texture': [],
        'repetitive_pattern': [],
        'illumination_change': [],
        'large_viewpoint': [],
        'semantic_ambiguity': []
    }
    
    for pair_info in difficult_pairs:
        img1, img2, difficulty_type = pair_info
        
        output = model.match(img1, img2)
        success_rate = evaluate_matching_success(output, pair_info['gt'])
        
        failure_analysis[difficulty_type].append({
            'pair_id': pair_info['id'],
            'success_rate': success_rate,
            'num_matches': len(output['mkpts1']),
            'semantic_consistency': output.get('semantic_consistency', 0)
        })
    
    return failure_analysis
```

## 评估协议和基准

### 标准评估协议
```python
# evaluation/protocols.py
class EvaluationProtocol:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.load_protocol_config()
    
    def load_protocol_config(self):
        """加载数据集特定的评估协议"""
        if self.dataset_name == 'megadepth1500':
            self.pose_thresholds = [5, 10, 20]  # degrees
            self.ransac_threshold = 1.0  # pixels
            self.min_matches = 10
            
        elif self.dataset_name == 'scannet1500':
            self.pose_thresholds = [5, 10, 20]
            self.ransac_threshold = 0.5
            self.min_matches = 15
            
        elif self.dataset_name == 'hpatches':
            self.matching_thresholds = [1, 3, 5]  # pixels
            self.use_homography = True
    
    def evaluate_pose_estimation(self, predictions, ground_truth):
        """位姿估计评估"""
        results = {}
        
        pose_errors = []
        for pred, gt in zip(predictions, ground_truth):
            if len(pred['matches']) >= self.min_matches:
                # RANSAC位姿估计
                pose = estimate_pose_ransac(
                    pred['matches'], 
                    threshold=self.ransac_threshold
                )
                
                # 计算位姿误差
                error = compute_pose_error(pose, gt['pose'])
                pose_errors.append(error)
            else:
                pose_errors.append(float('inf'))
        
        # 计算AUC
        for threshold in self.pose_thresholds:
            auc = pose_auc(pose_errors, [threshold])
            results[f'auc@{threshold}'] = auc
        
        return results
```

### 基准测试框架
```python
# evaluation/benchmark.py
class BenchmarkFramework:
    def __init__(self):
        self.protocols = {
            'megadepth1500': EvaluationProtocol('megadepth1500'),
            'scannet1500': EvaluationProtocol('scannet1500'),
            'hpatches': EvaluationProtocol('hpatches')
        }
    
    def run_benchmark(self, model, dataset_name):
        """运行标准基准测试"""
        print(f"Running benchmark on {dataset_name}")
        
        # 加载数据集和协议
        dataset = self.load_dataset(dataset_name)
        protocol = self.protocols[dataset_name]
        
        # 运行评估
        results = []
        for data_item in tqdm(dataset):
            try:
                # 模型推理
                prediction = model.predict(data_item)
                
                # 协议评估
                metrics = protocol.evaluate(prediction, data_item['gt'])
                results.append(metrics)
                
            except Exception as e:
                print(f"Error processing item: {e}")
                results.append(None)
        
        # 汇总结果
        final_metrics = self.aggregate_results(results)
        return final_metrics
    
    def compare_methods(self, methods_dict, datasets):
        """多方法对比"""
        comparison_results = {}
        
        for dataset_name in datasets:
            comparison_results[dataset_name] = {}
            
            for method_name, method in methods_dict.items():
                print(f"Evaluating {method_name} on {dataset_name}")
                results = self.run_benchmark(method, dataset_name)
                comparison_results[dataset_name][method_name] = results
        
        return comparison_results
```

## 预期实验结果

### 性能提升预期

#### 主要指标改进
| 数据集 | 指标 | XFeat | SG-XFeat | 提升 |
|--------|------|-------|----------|------|
| MegaDepth-1500 | AUC@5° | 0.564 | 0.595 | +5.5% |
| MegaDepth-1500 | AUC@10° | 0.710 | 0.751 | +5.8% |
| ScanNet-1500 | Pose Acc@5° | 0.635 | 0.724 | +14.0% |
| HPatches | Matching Acc | 0.582 | 0.651 | +11.9% |

#### 鲁棒性提升
| 场景类型 | 条件 | 性能提升 |
|----------|------|----------|
| 光照变化 | 强光/阴影 | +18.5% |
| 视角变化 | 大角度 | +12.3% |
| 模糊 | 运动模糊 | +22.1% |
| 噪声 | 高噪声 | +15.7% |

#### 效率对比
| 模型 | 大小(MB) | FLOPs(G) | 时间(ms) | AUC@5° |
|------|----------|----------|----------|--------|
| XFeat | 2.4 | 1.2 | 45 | 0.564 |
| SG-XFeat-Lite | 3.1 | 1.8 | 58 | 0.585 |
| SG-XFeat-Full | 3.6 | 2.3 | 68 | 0.595 |

### 消融实验预期结果
| 配置 | 语义注意力 | 语义描述符 | 语义匹配 | AUC@5° |
|------|------------|------------|----------|--------|
| Baseline | ✗ | ✗ | ✗ | 0.564 |
| + 语义注意力 | ✓ | ✗ | ✗ | 0.578 |
| + 语义描述符 | ✓ | ✓ | ✗ | 0.587 |
| + 语义匹配 | ✓ | ✓ | ✓ | 0.595 |

## 实验时间表

### 第一阶段：基础实验 (4周)
- **Week 1-2**: 数据准备和环境搭建
- **Week 3**: 基础方法实现和单元测试
- **Week 4**: 初步性能验证

### 第二阶段：核心实验 (6周)
- **Week 5-6**: 语义引导特征检测器实验
- **Week 7-8**: 语义增强描述符实验
- **Week 9-10**: 端到端匹配性能评估

### 第三阶段：优化和对比 (4周)
- **Week 11**: 轻量化和效率优化
- **Week 12**: 与SOTA方法对比
- **Week 13**: 消融实验和分析
- **Week 14**: 鲁棒性测试

### 第四阶段：分析和报告 (2周)
- **Week 15**: 结果分析和可视化
- **Week 16**: 实验报告编写和代码整理

## 成功指标

### 技术指标
1. **性能提升**：在MegaDepth-1500上AUC@5°提升≥5%
2. **效率保持**：推理时间增加≤30%，模型大小增加≤50%
3. **鲁棒性**：在挑战性条件下性能提升≥15%
4. **一致性**：语义一致性匹配率≥85%

### 可复现性指标
1. **开源代码**：完整的训练和评估代码
2. **预训练模型**：发布可直接使用的模型权重
3. **详细文档**：包含所有实验细节和超参数
4. **基准结果**：在标准数据集上的可复现结果

### 影响力指标
1. **学术贡献**：方法的创新性和有效性
2. **实用价值**：在实际应用中的可行性
3. **社区认可**：代码使用和引用情况
4. **技术转化**：向产业应用的转化潜力
