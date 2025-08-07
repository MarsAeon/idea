"""
评估示例
演示如何使用评估脚本来评估语义引导的轻量级特征检测器
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import os
import sys
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入项目模块
from semantic_guided_xfeat_implementation import SemanticGuidedXFeat
from evaluate_semantic_xfeat import BenchmarkSuite, FeatureEvaluator
from dataset_processing import DataModule


class EvaluationExample:
    """
    评估示例类
    """
    
    def __init__(self, model_path=None, config_path=None):
        """
        初始化评估示例
        
        Args:
            model_path: 模型文件路径
            config_path: 配置文件路径
        """
        self.model_path = model_path
        self.config_path = config_path
        
        # 加载配置
        self.config = self._load_config()
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型
        self.model = self._load_model()
        
        # 初始化评估器
        self.evaluator = FeatureEvaluator(self.model, self.device)
        
        # 初始化基准测试套件
        self.benchmark_suite = BenchmarkSuite(self.model, self.device)
        
    def _load_config(self):
        """
        加载配置
        
        Returns:
            配置字典
        """
        if self.config_path is None:
            # 使用默认配置
            config = {
                "model": {
                    "input_channels": 1,
                    "feature_dim": 128,
                    "hidden_dim": 256,
                    "num_keypoints": 500,
                    "backbone": "resnet18",
                    "use_semantic": True,
                    "semantic_channels": 64
                },
                "evaluation": {
                    "batch_size": 8,
                    "num_test_samples": 100,
                    "test_datasets": ["synthetic"],
                    "metrics": [
                        "repeatability",
                        "matching_accuracy",
                        "runtime",
                        "model_size",
                        "semantic_consistency"
                    ],
                    "visualization": True,
                    "save_results": True,
                    "generate_report": True
                },
                "benchmark": {
                    "image_pairs": 50,
                    "homography_threshold": 3.0,
                    "matching_threshold": 0.8,
                    "runtime_iterations": 10,
                    "memory_profiling": True
                }
            }
        else:
            # 从文件加载配置
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
        return config
        
    def _load_model(self):
        """
        加载模型
        
        Returns:
            加载的模型
        """
        model_config = self.config['model']
        
        # 创建模型
        model = SemanticGuidedXFeat(
            input_channels=model_config['input_channels'],
            feature_dim=model_config['feature_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_keypoints=model_config['num_keypoints'],
            backbone_type=model_config['backbone'],
            use_semantic=model_config['use_semantic'],
            semantic_channels=model_config['semantic_channels']
        )
        
        # 加载预训练权重
        if self.model_path and Path(self.model_path).exists():
            print(f"加载模型权重: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            print("模型权重加载成功")
        else:
            print("使用随机初始化的模型")
            
        # 将模型移动到设备
        model = model.to(self.device)
        model.eval()
        
        return model
        
    def create_test_data(self, num_samples=50):
        """
        创建测试数据
        
        Args:
            num_samples: 样本数量
            
        Returns:
            测试数据列表
        """
        print(f"\n创建测试数据 ({num_samples}个样本)...")
        test_data = []
        
        for i in range(num_samples):
            # 创建基础图像
            image = np.zeros((256, 256), dtype=np.uint8)
            
            # 添加随机几何形状
            shape_type = np.random.choice(['circle', 'rectangle', 'ellipse'])
            center = (np.random.randint(64, 192), np.random.randint(64, 192))
            color = np.random.randint(100, 255)
            
            if shape_type == 'circle':
                radius = np.random.randint(30, 80)
                cv2.circle(image, center, radius, color, -1)
            elif shape_type == 'rectangle':
                width = np.random.randint(60, 120)
                height = np.random.randint(60, 120)
                x1 = max(0, center[0] - width // 2)
                y1 = max(0, center[1] - height // 2)
                x2 = min(256, x1 + width)
                y2 = min(256, y1 + height)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
            else:  # ellipse
                axes = (np.random.randint(40, 80), np.random.randint(30, 60))
                angle = np.random.randint(0, 180)
                cv2.ellipse(image, center, axes, angle, 0, 360, color, -1)
                
            # 添加噪声
            noise = np.random.normal(0, 10, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # 创建语义图（简单的语义分割）
            semantic_map = np.zeros((256, 256), dtype=np.uint8)
            semantic_map[image > 50] = 1  # 前景
            
            test_data.append({
                'image': image,
                'semantic_map': semantic_map,
                'image_path': f'test_image_{i}.png'
            })
            
        print(f"创建了 {len(test_data)} 个测试样本")
        return test_data
        
    def evaluate_repeatability(self, test_data):
        """
        评估特征重复性
        
        Args:
            test_data: 测试数据
            
        Returns:
            重复性评估结果
        """
        print("\n评估特征重复性...")
        
        # 创建图像对
        image_pairs = []
        for i in range(0, len(test_data), 2):
            if i + 1 < len(test_data):
                image_pairs.append((test_data[i]['image'], test_data[i + 1]['image']))
                
        # 评估重复性
        results = self.evaluator.evaluate_repeatability(image_pairs)
        
        print(f"重复性评估结果:")
        print(f"平均重复性: {results['mean_repeatability']:.4f}")
        print(f"最小重复性: {results['min_repeatability']:.4f}")
        
        return results
        
    def evaluate_matching_accuracy(self, test_data):
        """
        评估匹配准确率
        
        Args:
            test_data: 测试数据
            
        Returns:
            匹配准确率评估结果
        """
        print("\n评估匹配准确率...")
        
        # 创建图像对
        image_pairs = []
        for i in range(0, len(test_data), 2):
            if i + 1 < len(test_data):
                image_pairs.append((test_data[i]['image'], test_data[i + 1]['image']))
                
        # 评估匹配准确率
        results = self.evaluator.evaluate_matching_accuracy(image_pairs)
        
        print(f"匹配准确率评估结果:")
        print(f"内点率: {results['inlier_ratio']:.4f}")
        print(f"精确率: {results['precision']:.4f}")
        print(f"召回率: {results['recall']:.4f}")
        print(f"F1分数: {results['f1_score']:.4f}")
        
        return results
        
    def evaluate_runtime(self, test_data):
        """
        评估运行时间
        
        Args:
            test_data: 测试数据
            
        Returns:
            运行时间评估结果
        """
        print("\n评估运行时间...")
        
        # 使用部分数据进行运行时间测试
        test_images = [data['image'] for data in test_data[:20]]
        
        # 评估运行时间
        results = self.evaluator.evaluate_runtime(test_images)
        
        print(f"运行时间评估结果:")
        print(f"平均检测时间: {results['mean_detection_time']:.4f} 秒")
        print(f"FPS: {results['fps']:.2f}")
        print(f"估计内存使用 (MB): {results['estimated_memory_mb']:.2f}")
        
        return results
        
    def evaluate_semantic_consistency(self, test_data):
        """
        评估语义一致性
        
        Args:
            test_data: 测试数据
            
        Returns:
            语义一致性评估结果
        """
        print("\n评估语义一致性...")
        
        # 准备数据
        test_images = [data['image'] for data in test_data]
        semantic_maps = [data['semantic_map'] for data in test_data]
        
        # 评估语义一致性
        results = self.evaluator.evaluate_semantic_consistency(test_images, semantic_maps)
        
        print(f"语义一致性评估结果:")
        print(f"平均一致性: {results['mean_consistency']:.4f}")
        
        return results
        
    def evaluate_model_size(self):
        """
        评估模型大小
        
        Returns:
            模型大小评估结果
        """
        print("\n评估模型大小...")
        
        # 评估模型大小
        results = self.evaluator.evaluate_model_size()
        
        print(f"模型大小评估结果:")
        print(f"参数数量: {results['param_count']:,}")
        print(f"模型大小 (MB): {results['model_size_mb']:.2f}")
        print(f"估计内存使用 (MB): {results['estimated_memory_mb']:.2f}")
        
        return results
        
    def run_full_benchmark(self, test_data):
        """
        运行完整基准测试
        
        Args:
            test_data: 测试数据
            
        Returns:
            完整基准测试结果
        """
        print("\n运行完整基准测试...")
        
        # 准备基准测试数据
        benchmark_data = {
            'images': [data['image'] for data in test_data],
            'semantic_maps': [data['semantic_map'] for data in test_data]
        }
        
        # 运行基准测试
        results = self.benchmark_suite.run_full_benchmark(benchmark_data)
        
        print(f"完整基准测试结果:")
        for metric, value in results.items():
            if isinstance(value, dict):
                print(f"  {metric}:")
                for sub_metric, sub_value in value.items():
                    print(f"    {sub_metric}: {sub_value:.4f}")
            else:
                print(f"  {metric}: {value:.4f}")
                
        return results
        
    def visualize_results(self, results, output_dir):
        """
        可视化评估结果
        
        Args:
            results: 评估结果字典
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建综合可视化图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 重复性分布
        if 'repeatability' in results:
            repeatability_results = results['repeatability']
            if 'repeatability_scores' in repeatability_results:
                axes[0, 0].hist(repeatability_results['repeatability_scores'], bins=20, alpha=0.7)
                axes[0, 0].set_title('特征重复性分布')
                axes[0, 0].set_xlabel('重复性分数')
                axes[0, 0].set_ylabel('频次')
                axes[0, 0].grid(True, alpha=0.3)
                
        # 2. 匹配准确率
        if 'matching_accuracy' in results:
            matching_results = results['matching_accuracy']
            metrics = ['inlier_ratio', 'precision', 'recall', 'f1_score']
            values = [matching_results.get(metric, 0) for metric in metrics]
            
            bars = axes[0, 1].bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
            axes[0, 1].set_title('匹配准确率指标')
            axes[0, 1].set_ylabel('分数')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.3f}', ha='center', va='bottom')
                               
        # 3. 运行时间分析
        if 'runtime' in results:
            runtime_results = results['runtime']
            if 'detection_times' in runtime_results:
                axes[0, 2].plot(runtime_results['detection_times'], 'b-o', markersize=4)
                axes[0, 2].axhline(y=runtime_results.get('mean_detection_time', 0), 
                                  color='r', linestyle='--', label='平均值')
                axes[0, 2].set_title('检测时间分析')
                axes[0, 2].set_xlabel('样本序号')
                axes[0, 2].set_ylabel('检测时间 (秒)')
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)
                
        # 4. 语义一致性
        if 'semantic_consistency' in results:
            semantic_results = results['semantic_consistency']
            if 'consistency_scores' in semantic_results:
                axes[1, 0].hist(semantic_results['consistency_scores'], bins=20, alpha=0.7, color='green')
                axes[1, 0].set_title('语义一致性分布')
                axes[1, 0].set_xlabel('一致性分数')
                axes[1, 0].set_ylabel('频次')
                axes[1, 0].grid(True, alpha=0.3)
                
        # 5. 模型复杂度
        if 'model_size' in results:
            model_results = results['model_size']
            metrics = ['参数数量 (千)', '模型大小 (MB)', '内存使用 (MB)']
            values = [
                model_results.get('param_count', 0) / 1000,
                model_results.get('model_size_mb', 0),
                model_results.get('estimated_memory_mb', 0)
            ]
            
            bars = axes[1, 1].bar(metrics, values, color=['purple', 'cyan', 'magenta'])
            axes[1, 1].set_title('模型复杂度指标')
            axes[1, 1].set_ylabel('数值')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.1f}', ha='center', va='bottom')
                               
        # 6. 综合性能雷达图
        self._create_performance_radar(results, axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'evaluation_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"评估结果可视化保存到: {output_dir / 'evaluation_results.png'}")
        
    def _create_performance_radar(self, results, ax):
        """
        创建性能雷达图
        
        Args:
            results: 评估结果
            ax: matplotlib轴对象
        """
        # 准备雷达图数据
        categories = ['重复性', '匹配准确率', '运行效率', '语义一致性', '模型轻量性']
        values = [
            results.get('repeatability', {}).get('mean_repeatability', 0),
            results.get('matching_accuracy', {}).get('f1_score', 0),
            min(results.get('runtime', {}).get('fps', 0) / 30, 1.0),  # 归一化FPS
            results.get('semantic_consistency', {}).get('mean_consistency', 0),
            max(0, 1 - results.get('model_size', {}).get('model_size_mb', 0) / 100)  # 轻量性分数
        ]
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]
        
        # 创建雷达图
        ax.plot(angles, values, 'o-', linewidth=2, label='模型性能')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('综合性能雷达图', size=12, color='black', y=1.1)
        ax.grid(True)
        
    def create_evaluation_report(self, results, output_dir):
        """
        创建评估报告
        
        Args:
            results: 评估结果字典
            output_dir: 输出目录
            
        Returns:
            评估报告字典
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建报告
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_info": {
                "model_path": str(self.model_path) if self.model_path else "N/A",
                "initialization": "pretrained" if self.model_path else "random",
                "model_config": self.config['model']
            },
            "evaluation_config": self.config['evaluation'],
            "results": results,
            "summary": self._create_summary(results)
        }
        
        # 保存报告
        report_path = output_dir / 'evaluation_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"评估报告保存到: {report_path}")
        
        return report
        
    def _create_summary(self, results):
        """
        创建评估摘要
        
        Args:
            results: 评估结果字典
            
        Returns:
            摘要字典
        """
        summary = {
            "overall_score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        # 计算总体分数
        scores = []
        if 'repeatability' in results:
            scores.append(results['repeatability'].get('mean_repeatability', 0))
        if 'matching_accuracy' in results:
            scores.append(results['matching_accuracy'].get('f1_score', 0))
        if 'semantic_consistency' in results:
            scores.append(results['semantic_consistency'].get('mean_consistency', 0))
            
        if scores:
            summary['overall_score'] = np.mean(scores)
            
        # 分析优势和劣势
        if summary['overall_score'] > 0.8:
            summary['strengths'].append("整体性能优秀")
        elif summary['overall_score'] > 0.6:
            summary['strengths'].append("整体性能良好")
        else:
            summary['weaknesses'].append("整体性能需要改进")
            
        # 具体指标分析
        if 'repeatability' in results:
            rep_score = results['repeatability'].get('mean_repeatability', 0)
            if rep_score > 0.7:
                summary['strengths'].append("特征重复性优秀")
            else:
                summary['weaknesses'].append("特征重复性需要改进")
                
        if 'matching_accuracy' in results:
            match_score = results['matching_accuracy'].get('f1_score', 0)
            if match_score > 0.7:
                summary['strengths'].append("匹配准确率优秀")
            else:
                summary['weaknesses'].append("匹配准确率需要改进")
                
        if 'runtime' in results:
            fps = results['runtime'].get('fps', 0)
            if fps > 20:
                summary['strengths'].append("运行速度快")
            else:
                summary['weaknesses'].append("运行速度需要优化")
                
        # 生成建议
        if summary['overall_score'] < 0.6:
            summary['recommendations'].append("建议重新训练模型或调整超参数")
        if 'runtime' in results and results['runtime'].get('fps', 0) < 10:
            summary['recommendations'].append("建议优化模型结构或使用模型量化")
        if 'model_size' in results and results['model_size'].get('model_size_mb', 0) > 50:
            summary['recommendations'].append("建议使用模型压缩技术减小模型大小")
            
        return summary
        
    def quick_evaluation(self, num_samples=20):
        """
        快速评估（用于演示）
        
        Args:
            num_samples: 样本数量
            
        Returns:
            评估结果
        """
        print("\n开始快速评估演示...")
        
        # 创建测试数据
        test_data = self.create_test_data(num_samples)
        
        # 运行各项评估
        results = {}
        
        try:
            results['repeatability'] = self.evaluate_repeatability(test_data)
        except Exception as e:
            print(f"重复性评估失败: {e}")
            
        try:
            results['matching_accuracy'] = self.evaluate_matching_accuracy(test_data)
        except Exception as e:
            print(f"匹配准确率评估失败: {e}")
            
        try:
            results['runtime'] = self.evaluate_runtime(test_data)
        except Exception as e:
            print(f"运行时间评估失败: {e}")
            
        try:
            results['semantic_consistency'] = self.evaluate_semantic_consistency(test_data)
        except Exception as e:
            print(f"语义一致性评估失败: {e}")
            
        try:
            results['model_size'] = self.evaluate_model_size()
        except Exception as e:
            print(f"模型大小评估失败: {e}")
            
        # 创建输出目录
        output_dir = Path("./examples/output/evaluation_demo")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 可视化结果
        self.visualize_results(results, output_dir)
        
        # 创建评估报告
        self.create_evaluation_report(results, output_dir)
        
        print("\n快速评估演示完成！")
        
        return results


def main():
    """
    主函数
    """
    print("语义引导的轻量级特征检测器 - 评估示例")
    print("="*60)
    
    # 配置参数
    config = {
        'model_path': None,  # 模型路径（None表示使用随机初始化）
        'config_path': None,  # 配置文件路径
        'quick_evaluation': True,  # 使用快速评估模式
        'num_samples': 20,  # 测试样本数量
        'full_benchmark': False,  # 是否运行完整基准测试
        'create_visualization': True,  # 创建可视化
        'output_dir': './examples/output'
    }
    
    # 创建评估示例
    evaluation_example = EvaluationExample(
        model_path=config['model_path'],
        config_path=config['config_path']
    )
    
    try:
        if config['quick_evaluation']:
            # 快速评估演示
            print("\n使用快速评估模式...")
            results = evaluation_example.quick_evaluation(
                num_samples=config['num_samples']
            )
        else:
            # 完整评估流程
            print("\n使用完整评估流程...")
            
            # 创建测试数据
            test_data = evaluation_example.create_test_data(config['num_samples'])
            
            # 运行各项评估
            results = {}
            
            results['repeatability'] = evaluation_example.evaluate_repeatability(test_data)
            results['matching_accuracy'] = evaluation_example.evaluate_matching_accuracy(test_data)
            results['runtime'] = evaluation_example.evaluate_runtime(test_data)
            results['semantic_consistency'] = evaluation_example.evaluate_semantic_consistency(test_data)
            results['model_size'] = evaluation_example.evaluate_model_size()
            
            # 运行完整基准测试
            if config['full_benchmark']:
                benchmark_results = evaluation_example.run_full_benchmark(test_data)
                results.update(benchmark_results)
                
            # 创建输出目录
            output_dir = Path(config['output_dir']) / 'evaluation_results'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 可视化结果
            if config['create_visualization']:
                evaluation_example.visualize_results(results, output_dir)
                
            # 创建评估报告
            evaluation_example.create_evaluation_report(results, output_dir)
            
        print("\n" + "="*60)
        print("评估示例完成！")
        print(f"结果保存在: {config['output_dir']}")
        print("="*60)
        
        return results
        
    except KeyboardInterrupt:
        print("\n评估被用户中断")
    except Exception as e:
        print(f"\n评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行主函数
    results = main()
    
    # 可选：返回结果供其他示例使用
    if results:
        print("\n评估成功完成！")
    else:
        print("\n评估失败！")