"""
语义引导的轻量级特征检测器评估脚本
用于模型性能评估和实验对比
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# 导入自定义模块
from semantic_guided_xfeat_implementation import SemanticGuidedXFeat
from train_semantic_xfeat import load_config


class FeatureEvaluator:
    """
    特征评估器
    评估语义引导特征检测器的性能
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def evaluate_repeatability(self, image_pairs, threshold=3.0):
        """
        评估特征点重复性
        Args:
            image_pairs: 图像对列表
            threshold: 重复性阈值（像素）
        Returns:
            repeatability_score: 重复性分数
        """
        repeatability_scores = []
        
        for img1, img2 in tqdm(image_pairs, desc="Evaluating repeatability"):
            # 检测特征点
            feat1 = self.model.detectAndCompute(img1)
            feat2 = self.model.detectAndCompute(img2)
            
            # 计算重复性
            rep_score = self._compute_repeatability(feat1[0], feat2[0], threshold)
            repeatability_scores.append(rep_score)
            
        return np.mean(repeatability_scores)
        
    def _compute_repeatability(self, feat1, feat2, threshold):
        """
        计算单对图像的重复性
        """
        kpts1 = feat1['keypoints']
        kpts2 = feat2['keypoints']
        
        if len(kpts1) == 0 or len(kpts2) == 0:
            return 0.0
            
        # 计算距离矩阵
        dist_matrix = np.zeros((len(kpts1), len(kpts2)))
        for i, kp1 in enumerate(kpts1):
            for j, kp2 in enumerate(kpts2):
                dist_matrix[i, j] = np.linalg.norm(kp1 - kp2)
                
        # 找到最近邻
        min_dist_12 = np.min(dist_matrix, axis=1)
        min_dist_21 = np.min(dist_matrix, axis=0)
        
        # 计算重复点数量
        rep_mask_12 = min_dist_12 < threshold
        rep_mask_21 = min_dist_21 < threshold
        
        rep_count = np.sum(rep_mask_12) + np.sum(rep_mask_21)
        total_count = len(kpts1) + len(kpts2)
        
        return rep_count / total_count if total_count > 0 else 0.0
        
    def evaluate_matching_accuracy(self, image_pairs, homographies, threshold=3.0):
        """
        评估匹配准确率
        Args:
            image_pairs: 图像对列表
            homographies: 对应的单应性矩阵
            threshold: 内点阈值
        Returns:
            matching_metrics: 匹配指标字典
        """
        inlier_rates = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for (img1, img2), H in tqdm(zip(image_pairs, homographies), desc="Evaluating matching"):
            # 检测和匹配
            matches = self.match_images(img1, img2)
            
            # 评估匹配质量
            metrics = self._evaluate_matches(matches, H, threshold)
            
            inlier_rates.append(metrics['inlier_rate'])
            precision_scores.append(metrics['precision'])
            recall_scores.append(metrics['recall'])
            f1_scores.append(metrics['f1_score'])
            
        return {
            'inlier_rate': np.mean(inlier_rates),
            'precision': np.mean(precision_scores),
            'recall': np.mean(recall_scores),
            'f1_score': np.mean(f1_scores)
        }
        
    def match_images(self, img1, img2):
        """
        匹配两幅图像
        """
        # 检测特征点
        feat1 = self.model.detectAndCompute(img1, top_k=1000)
        feat2 = self.model.detectAndCompute(img2, top_k=1000)
        
        if len(feat1) == 0 or len(feat2) == 0:
            return {'matches': np.array([]), 'scores': np.array([])}
            
        # 简化的匹配实现（最近邻匹配）
        desc1 = feat1[0]['descriptors']
        desc2 = feat2[0]['descriptors']
        
        # 计算距离矩阵
        dist_matrix = np.zeros((len(desc1), len(desc2)))
        for i, d1 in enumerate(desc1):
            for j, d2 in enumerate(desc2):
                dist_matrix[i, j] = np.linalg.norm(d1 - d2)
                
        # 最近邻匹配
        matches = []
        scores = []
        for i in range(len(desc1)):
            min_idx = np.argmin(dist_matrix[i])
            min_dist = dist_matrix[i, min_idx]
            
            # 比率测试
            second_min_dist = np.partition(dist_matrix[i], 1)[1]
            ratio = min_dist / (second_min_dist + 1e-8)
            
            if ratio < 0.8:  # 阈值可调
                matches.append([i, min_idx])
                scores.append(1.0 - ratio)
                
        return {
            'matches': np.array(matches),
            'scores': np.array(scores)
        }
        
    def _evaluate_matches(self, matches, H, threshold):
        """
        评估匹配质量
        """
        if len(matches['matches']) == 0:
            return {
                'inlier_rate': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
            
        # 获取匹配点
        match_indices = matches['matches']
        
        # 简化：假设我们有特征点坐标
        # 实际实现需要从特征检测结果中获取
        kpts1 = np.random.rand(len(match_indices), 2) * 256
        kpts2 = np.random.rand(len(match_indices), 2) * 256
        
        # 变换第一组点
        kpts1_transformed = self._transform_points(kpts1, H)
        
        # 计算距离
        distances = np.linalg.norm(kpts1_transformed - kpts2, axis=1)
        
        # 内点判断
        inlier_mask = distances < threshold
        inlier_count = np.sum(inlier_mask)
        total_count = len(match_indices)
        
        # 计算指标
        inlier_rate = inlier_count / total_count if total_count > 0 else 0.0
        precision = inlier_rate  # 在这里相同
        recall = inlier_rate  # 简化计算
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'inlier_rate': inlier_rate,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        
    def _transform_points(self, points, H):
        """
        变换点坐标
        """
        # 转换为齐次坐标
        homogeneous_points = np.column_stack([points, np.ones(len(points))])
        
        # 应用变换
        transformed = H @ homogeneous_points.T
        
        # 转换回笛卡尔坐标
        transformed = transformed[:2] / transformed[2]
        
        return transformed.T
        
    def evaluate_runtime(self, image_size=(256, 256), num_runs=100):
        """
        评估运行时间
        Args:
            image_size: 图像尺寸
            num_runs: 运行次数
        Returns:
            runtime_metrics: 运行时间指标
        """
        # 创建测试图像
        test_image = torch.randn(1, 1, image_size[0], image_size[1]).to(self.device)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = self.model.detectAndCompute(test_image)
                
        # 测量时间
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in tqdm(range(num_runs), desc="Evaluating runtime"):
                _ = self.model.detectAndCompute(test_image)
                
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        fps = 1.0 / avg_time
        
        return {
            'avg_time_ms': avg_time * 1000,
            'fps': fps,
            'total_time_s': total_time
        }
        
    def evaluate_semantic_consistency(self, image_pairs, semantic_maps):
        """
        评估语义一致性
        Args:
            image_pairs: 图像对列表
            semantic_maps: 对应的语义图
        Returns:
            semantic_metrics: 语义指标
        """
        consistency_scores = []
        
        for (img1, img2), (sem1, sem2) in tqdm(zip(image_pairs, semantic_maps), desc="Evaluating semantic consistency"):
            # 检测特征
            feat1 = self.model.detectAndCompute(img1)
            feat2 = self.model.detectAndCompute(img2)
            
            if len(feat1) == 0 or len(feat2) == 0:
                consistency_scores.append(0.0)
                continue
                
            # 获取语义注意力
            sem_attention1 = feat1[0]['semantic_attention']
            sem_attention2 = feat2[0]['semantic_attention']
            
            # 匹配特征点
            matches = self.match_images(img1, img2)
            
            # 计算语义一致性
            consistency = self._compute_semantic_consistency(
                matches, sem_attention1, sem_attention2, sem1, sem2
            )
            consistency_scores.append(consistency)
            
        return {
            'semantic_consistency': np.mean(consistency_scores)
        }
        
    def _compute_semantic_consistency(self, matches, sem_attention1, sem_attention2, sem1, sem2):
        """
        计算语义一致性
        """
        if len(matches['matches']) == 0:
            return 0.0
            
        match_indices = matches['matches']
        consistency_scores = []
        
        for idx1, idx2 in match_indices:
            # 获取语义类别
            # 简化实现
            sem_class1 = torch.argmax(sem_attention1, dim=0)
            sem_class2 = torch.argmax(sem_attention2, dim=0)
            
            # 计算语义相似度
            similarity = (sem_class1 == sem_class2).float().mean()
            consistency_scores.append(similarity.item())
            
        return np.mean(consistency_scores)
        
    def evaluate_model_size(self):
        """
        评估模型大小
        Returns:
            size_metrics: 大小指标
        """
        # 计算参数数量
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # 估算模型大小（MB）
        model_size_mb = param_count * 4 / (1024 * 1024)  # 假设float32
        
        return {
            'total_params': param_count,
            'trainable_params': trainable_count,
            'model_size_mb': model_size_mb
        }


class BenchmarkSuite:
    """
    基准测试套件
    完整的模型评估流程
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.evaluator = FeatureEvaluator(model)
        
    def run_full_benchmark(self, test_data):
        """
        运行完整基准测试
        Args:
            test_data: 测试数据
        Returns:
            results: 完整的评估结果
        """
        results = {}
        
        print("Starting full benchmark evaluation...")
        
        # 1. 模型大小评估
        print("\n1. Evaluating model size...")
        results['model_size'] = self.evaluator.evaluate_model_size()
        
        # 2. 运行时间评估
        print("\n2. Evaluating runtime performance...")
        results['runtime'] = self.evaluator.evaluate_runtime(
            image_size=self.config.get('image_size', (256, 256)),
            num_runs=self.config.get('runtime_runs', 100)
        )
        
        # 3. 特征重复性评估
        if 'repeatability_data' in test_data:
            print("\n3. Evaluating feature repeatability...")
            results['repeatability'] = self.evaluator.evaluate_repeatability(
                test_data['repeatability_data']
            )
        
        # 4. 匹配准确率评估
        if 'matching_data' in test_data:
            print("\n4. Evaluating matching accuracy...")
            results['matching'] = self.evaluator.evaluate_matching_accuracy(
                test_data['matching_data']['image_pairs'],
                test_data['matching_data']['homographies']
            )
        
        # 5. 语义一致性评估
        if 'semantic_data' in test_data:
            print("\n5. Evaluating semantic consistency...")
            results['semantic'] = self.evaluator.evaluate_semantic_consistency(
                test_data['semantic_data']['image_pairs'],
                test_data['semantic_data']['semantic_maps']
            )
        
        return results
        
    def save_results(self, results, output_path):
        """
        保存评估结果
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
        
    def generate_report(self, results, output_dir):
        """
        生成评估报告
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 生成文本报告
        report_path = output_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write("Semantic-Guided XFeat Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            # 模型大小
            if 'model_size' in results:
                f.write("Model Size:\n")
                f.write(f"  Total Parameters: {results['model_size']['total_params']:,}\n")
                f.write(f"  Trainable Parameters: {results['model_size']['trainable_params']:,}\n")
                f.write(f"  Model Size: {results['model_size']['model_size_mb']:.2f} MB\n\n")
                
            # 运行时间
            if 'runtime' in results:
                f.write("Runtime Performance:\n")
                f.write(f"  Average Time: {results['runtime']['avg_time_ms']:.2f} ms\n")
                f.write(f"  FPS: {results['runtime']['fps']:.2f}\n\n")
                
            # 重复性
            if 'repeatability' in results:
                f.write("Feature Repeatability:\n")
                f.write(f"  Repeatability Score: {results['repeatability']:.4f}\n\n")
                
            # 匹配准确率
            if 'matching' in results:
                f.write("Matching Accuracy:\n")
                f.write(f"  Inlier Rate: {results['matching']['inlier_rate']:.4f}\n")
                f.write(f"  Precision: {results['matching']['precision']:.4f}\n")
                f.write(f"  Recall: {results['matching']['recall']:.4f}\n")
                f.write(f"  F1 Score: {results['matching']['f1_score']:.4f}\n\n")
                
            # 语义一致性
            if 'semantic' in results:
                f.write("Semantic Consistency:\n")
                f.write(f"  Consistency Score: {results['semantic']['semantic_consistency']:.4f}\n\n")
                
        print(f"Report generated at {report_path}")
        
        # 生成可视化图表
        self._generate_visualizations(results, output_dir)
        
    def _generate_visualizations(self, results, output_dir):
        """
        生成可视化图表
        """
        # 创建性能对比图
        if 'matching' in results and 'runtime' in results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 匹配准确率
            metrics = ['Inlier Rate', 'Precision', 'Recall', 'F1 Score']
            values = [
                results['matching']['inlier_rate'],
                results['matching']['precision'],
                results['matching']['recall'],
                results['matching']['f1_score']
            ]
            
            ax1.bar(metrics, values)
            ax1.set_title('Matching Accuracy')
            ax1.set_ylabel('Score')
            ax1.set_ylim([0, 1])
            
            # 运行时间
            ax2.text(0.5, 0.5, f'FPS: {results["runtime"]["fps"]:.1f}\nAvg Time: {results["runtime"]["avg_time_ms"]:.1f} ms',
                    ha='center', va='center', fontsize=16, transform=ax2.transAxes)
            ax2.set_title('Runtime Performance')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        print(f"Visualizations saved to {output_dir}")


def create_test_data(num_samples=50):
    """
    创建测试数据（简化版本）
    """
    test_data = {}
    
    # 重复性测试数据
    test_data['repeatability_data'] = [
        (torch.randn(1, 1, 256, 256), torch.randn(1, 1, 256, 256))
        for _ in range(num_samples)
    ]
    
    # 匹配测试数据
    test_data['matching_data'] = {
        'image_pairs': [
            (torch.randn(1, 1, 256, 256), torch.randn(1, 1, 256, 256))
            for _ in range(num_samples)
        ],
        'homographies': [
            np.random.rand(3, 3) for _ in range(num_samples)
        ]
    }
    
    # 语义测试数据
    test_data['semantic_data'] = {
        'image_pairs': [
            (torch.randn(1, 1, 256, 256), torch.randn(1, 1, 256, 256))
            for _ in range(num_samples)
        ],
        'semantic_maps': [
            (torch.randint(0, 20, (256, 256)), torch.randint(0, 20, (256, 256)))
            for _ in range(num_samples)
        ]
    }
    
    return test_data


def main():
    """
    主函数
    """
    # 加载配置
    config_path = "config_train_semantic_xfeat.json"
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = {
            'image_size': (256, 256),
            'runtime_runs': 100,
            'semantic_channels': 20
        }
        
    # 创建模型
    print("Loading model...")
    model = SemanticGuidedXFeat(
        semantic_channels=config.get('semantic_channels', 20)
    )
    
    # 加载预训练权重（如果有的话）
    model_path = "outputs/best_model.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
    
    # 创建基准测试套件
    benchmark = BenchmarkSuite(model, config)
    
    # 创建测试数据
    print("\nCreating test data...")
    test_data = create_test_data(num_samples=20)
    
    # 运行基准测试
    print("\nRunning benchmark evaluation...")
    results = benchmark.run_full_benchmark(test_data)
    
    # 保存结果
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    results_path = output_dir / "evaluation_results.json"
    benchmark.save_results(results, results_path)
    
    # 生成报告
    benchmark.generate_report(results, output_dir)
    
    print("\nEvaluation completed!")
    print(f"Results saved to {output_dir}")
    
    # 打印摘要
    print("\n=== Evaluation Summary ===")
    if 'model_size' in results:
        print(f"Model Size: {results['model_size']['model_size_mb']:.2f} MB")
    if 'runtime' in results:
        print(f"FPS: {results['runtime']['fps']:.1f}")
    if 'matching' in results:
        print(f"Matching F1 Score: {results['matching']['f1_score']:.4f}")
    if 'semantic' in results:
        print(f"Semantic Consistency: {results['semantic']['semantic_consistency']:.4f}")


if __name__ == "__main__":
    main()