"""
训练示例
演示如何训练语义引导的轻量级特征检测器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
from dataset_processing import DataModule
from train_semantic_xfeat import SemanticXFeatTrainer


class TrainingExample:
    """
    训练示例类
    """
    
    def __init__(self, config_path=None):
        """
        初始化训练示例
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = self._create_model()
        
        # 初始化数据模块
        self.data_module = DataModule(self.config['data'])
        
        # 初始化训练器
        self.trainer = SemanticXFeatTrainer(self.model, self.config)
        
    def _load_config(self, config_path):
        """
        加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        if config_path is None:
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
                "training": {
                    "batch_size": 16,
                    "learning_rate": 0.001,
                    "weight_decay": 1e-4,
                    "epochs": 10,
                    "warmup_epochs": 2,
                    "scheduler_type": "cosine",
                    "gradient_clip": 1.0,
                    "mixed_precision": True
                },
                "data": {
                    "dataset_type": "synthetic",
                    "image_size": [256, 256],
                    "num_train_samples": 1000,
                    "num_val_samples": 200,
                    "data_augmentation": True,
                    "augmentation_params": {
                        "rotation_range": 30,
                        "brightness_range": 0.2,
                        "contrast_range": 0.2,
                        "noise_std": 0.1
                    }
                },
                "loss": {
                    "detection_weight": 1.0,
                    "description_weight": 1.0,
                    "semantic_weight": 0.5,
                    "repeatability_weight": 0.3,
                    "matching_weight": 0.2
                },
                "logging": {
                    "log_dir": "./logs",
                    "save_dir": "./checkpoints",
                    "save_frequency": 5,
                    "log_frequency": 100,
                    "visualize_frequency": 500
                }
            }
        else:
            # 从文件加载配置
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
        return config
        
    def _create_model(self):
        """
        创建模型
        
        Returns:
            语义引导的特征检测器模型
        """
        model_config = self.config['model']
        
        model = SemanticGuidedXFeat(
            input_channels=model_config['input_channels'],
            feature_dim=model_config['feature_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_keypoints=model_config['num_keypoints'],
            backbone_type=model_config['backbone'],
            use_semantic=model_config['use_semantic'],
            semantic_channels=model_config['semantic_channels']
        )
        
        # 将模型移动到设备
        model = model.to(self.device)
        
        # 如果使用多GPU，使用DataParallel
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU")
            model = nn.DataParallel(model)
            
        return model
        
    def prepare_data(self):
        """
        准备数据
        
        Returns:
            训练和验证数据加载器
        """
        print("\n准备数据...")
        
        # 创建数据加载器
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        
        print(f"训练数据加载器: {len(train_loader)} 个批次")
        print(f"验证数据加载器: {len(val_loader)} 个批次")
        
        return train_loader, val_loader
        
    def train_model(self, train_loader, val_loader):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            训练历史
        """
        print("\n开始训练模型...")
        print(f"训练配置:")
        print(f"- 批次大小: {self.config['training']['batch_size']}")
        print(f"- 学习率: {self.config['training']['learning_rate']}")
        print(f"- 训练轮数: {self.config['training']['epochs']}")
        print(f"- 混合精度: {self.config['training']['mixed_precision']}")
        
        # 训练模型
        history = self.trainer.train(train_loader, val_loader)
        
        print("\n训练完成！")
        
        return history
        
    def evaluate_model(self, val_loader):
        """
        评估模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            评估结果
        """
        print("\n评估模型...")
        
        # 评估模型
        results = self.trainer.evaluate(val_loader)
        
        print("\n评估结果:")
        for metric, value in results.items():
            print(f"- {metric}: {value:.4f}")
            
        return results
        
    def save_model(self, save_path=None):
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        if save_path is None:
            save_dir = Path(self.config['logging']['save_dir'])
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "semantic_xfeat_trained.pth"
            
        print(f"\n保存模型到: {save_path}")
        
        # 保存模型状态
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'trainer_state': self.trainer.get_state()
        }, save_path)
        
        print(f"模型已保存到: {save_path}")
        
    def visualize_training(self, history, output_dir):
        """
        可视化训练过程
        
        Args:
            history: 训练历史
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建训练损失图表
        plt.figure(figsize=(12, 8))
        
        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.title('训练和验证损失')
        plt.xlabel('轮数')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        
        # 学习率曲线
        plt.subplot(2, 2, 2)
        plt.plot(history['learning_rate'])
        plt.title('学习率变化')
        plt.xlabel('轮数')
        plt.ylabel('学习率')
        plt.grid(True)
        
        # 检测准确率
        plt.subplot(2, 2, 3)
        if 'detection_accuracy' in history:
            plt.plot(history['detection_accuracy'], label='检测准确率')
            plt.title('检测准确率')
            plt.xlabel('轮数')
            plt.ylabel('准确率')
            plt.legend()
            plt.grid(True)
            
        # 匹配准确率
        plt.subplot(2, 2, 4)
        if 'matching_accuracy' in history:
            plt.plot(history['matching_accuracy'], label='匹配准确率')
            plt.title('匹配准确率')
            plt.xlabel('轮数')
            plt.ylabel('准确率')
            plt.legend()
            plt.grid(True)
            
        plt.tight_layout()
        plt.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"训练历史可视化保存到: {output_dir / 'training_history.png'}")
        
        # 创建指标雷达图
        if 'val_metrics' in history and history['val_metrics']:
            self._create_metrics_radar(history['val_metrics'][-1], output_dir)
            
    def _create_metrics_radar(self, metrics, output_dir):
        """
        创建指标雷达图
        
        Args:
            metrics: 指标字典
            output_dir: 输出目录
        """
        # 准备雷达图数据
        categories = ['检测准确率', '描述符质量', '语义一致性', '重复性', '匹配准确率']
        values = [
            metrics.get('detection_accuracy', 0),
            metrics.get('descriptor_quality', 0),
            metrics.get('semantic_consistency', 0),
            metrics.get('repeatability', 0),
            metrics.get('matching_accuracy', 0)
        ]
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]
        
        # 创建雷达图
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, values, 'o-', linewidth=2, label='模型性能')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('模型性能雷达图', size=16, color='black', y=1.1)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_radar.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"指标雷达图保存到: {output_dir / 'metrics_radar.png'}")
        
    def create_training_report(self, history, results, output_dir):
        """
        创建训练报告
        
        Args:
            history: 训练历史
            results: 评估结果
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.config,
            "training_summary": {
                "total_epochs": len(history['train_loss']),
                "final_train_loss": history['train_loss'][-1],
                "final_val_loss": history['val_loss'][-1],
                "best_val_loss": min(history['val_loss']),
                "best_epoch": np.argmin(history['val_loss']) + 1
            },
            "final_metrics": results,
            "performance_analysis": self._analyze_performance(history, results)
        }
        
        # 保存报告
        report_path = output_dir / 'training_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"训练报告保存到: {report_path}")
        
        return report
        
    def _analyze_performance(self, history, results):
        """
        分析性能
        
        Args:
            history: 训练历史
            results: 评估结果
            
        Returns:
            性能分析结果
        """
        analysis = {
            "convergence_analysis": {
                "loss_converged": self._check_convergence(history['val_loss']),
                "convergence_epoch": self._find_convergence_epoch(history['val_loss']),
                "stability_score": self._calculate_stability(history['val_loss'][-10:])
            },
            "performance_scores": {
                "overall_score": np.mean(list(results.values())),
                "detection_score": results.get('detection_accuracy', 0),
                "matching_score": results.get('matching_accuracy', 0),
                "semantic_score": results.get('semantic_consistency', 0)
            },
            "recommendations": self._generate_recommendations(history, results)
        }
        
        return analysis
        
    def _check_convergence(self, losses, threshold=0.01, window=5):
        """
        检查损失是否收敛
        
        Args:
            losses: 损失列表
            threshold: 收敛阈值
            window: 窗口大小
            
        Returns:
            是否收敛
        """
        if len(losses) < window:
            return False
            
        recent_losses = losses[-window:]
        variance = np.var(recent_losses)
        
        return variance < threshold
        
    def _find_convergence_epoch(self, losses, threshold=0.01, window=5):
        """
        找到收敛轮数
        
        Args:
            losses: 损失列表
            threshold: 收敛阈值
            window: 窗口大小
            
        Returns:
            收敛轮数
        """
        for i in range(window, len(losses)):
            recent_losses = losses[i-window:i]
            variance = np.var(recent_losses)
            
            if variance < threshold:
                return i
                
        return len(losses)
        
    def _calculate_stability(self, losses):
        """
        计算稳定性分数
        
        Args:
            losses: 损失列表
            
        Returns:
            稳定性分数
        """
        if len(losses) < 2:
            return 0.0
            
        # 计算损失变化的标准差
        changes = np.diff(losses)
        stability = 1.0 / (1.0 + np.std(changes))
        
        return stability
        
    def _generate_recommendations(self, history, results):
        """
        生成改进建议
        
        Args:
            history: 训练历史
            results: 评估结果
            
        Returns:
            建议列表
        """
        recommendations = []
        
        # 分析损失收敛
        if not self._check_convergence(history['val_loss']):
            recommendations.append("考虑增加训练轮数或调整学习率")
            
        # 分析性能指标
        if results.get('detection_accuracy', 0) < 0.8:
            recommendations.append("检测准确率较低，建议调整检测损失权重或增强数据增强")
            
        if results.get('matching_accuracy', 0) < 0.7:
            recommendations.append("匹配准确率较低，建议改进描述符网络或增加匹配损失权重")
            
        if results.get('semantic_consistency', 0) < 0.6:
            recommendations.append("语义一致性较低，建议增强语义引导模块或改进语义损失")
            
        # 分析过拟合
        train_loss = history['train_loss'][-1]
        val_loss = history['val_loss'][-1]
        
        if val_loss > train_loss * 1.2:
            recommendations.append("可能存在过拟合，建议增加正则化或减少模型复杂度")
            
        if not recommendations:
            recommendations.append("模型性能良好，可以继续优化或进行部署")
            
        return recommendations
        
    def quick_train(self, epochs=5, batch_size=8):
        """
        快速训练（用于演示）
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
        """
        print("\n开始快速训练演示...")
        
        # 更新配置
        self.config['training']['epochs'] = epochs
        self.config['training']['batch_size'] = batch_size
        self.config['data']['num_train_samples'] = 100
        self.config['data']['num_val_samples'] = 20
        
        # 重新初始化数据模块和训练器
        self.data_module = DataModule(self.config['data'])
        self.trainer = SemanticXFeatTrainer(self.model, self.config)
        
        # 准备数据
        train_loader, val_loader = self.prepare_data()
        
        # 训练模型
        history = self.train_model(train_loader, val_loader)
        
        # 评估模型
        results = self.evaluate_model(val_loader)
        
        # 保存模型
        self.save_model()
        
        # 创建输出目录
        output_dir = Path("./examples/output/training_demo")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 可视化训练过程
        self.visualize_training(history, output_dir)
        
        # 创建训练报告
        self.create_training_report(history, results, output_dir)
        
        print("\n快速训练演示完成！")
        
        return history, results


def main():
    """
    主函数
    """
    print("语义引导的轻量级特征检测器 - 训练示例")
    print("="*60)
    
    # 配置参数
    config = {
        'config_path': None,  # 使用默认配置
        'quick_train': True,  # 使用快速训练模式
        'epochs': 5,          # 快速训练轮数
        'batch_size': 8,      # 批次大小
        'save_model': True,   # 保存模型
        'create_visualization': True,  # 创建可视化
        'output_dir': './examples/output'
    }
    
    # 创建训练示例
    training_example = TrainingExample(config['config_path'])
    
    try:
        if config['quick_train']:
            # 快速训练演示
            print("\n使用快速训练模式...")
            history, results = training_example.quick_train(
                epochs=config['epochs'],
                batch_size=config['batch_size']
            )
        else:
            # 完整训练流程
            print("\n使用完整训练流程...")
            
            # 准备数据
            train_loader, val_loader = training_example.prepare_data()
            
            # 训练模型
            history = training_example.train_model(train_loader, val_loader)
            
            # 评估模型
            results = training_example.evaluate_model(val_loader)
            
            # 保存模型
            if config['save_model']:
                training_example.save_model()
                
            # 创建输出目录
            output_dir = Path(config['output_dir']) / 'training_results'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 可视化训练过程
            if config['create_visualization']:
                training_example.visualize_training(history, output_dir)
                
            # 创建训练报告
            training_example.create_training_report(history, results, output_dir)
            
        print("\n" + "="*60)
        print("训练示例完成！")
        print(f"结果保存在: {config['output_dir']}")
        print("="*60)
        
        return history, results
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行主函数
    history, results = main()
    
    # 可选：返回结果供其他示例使用
    if history and results:
        print("\n训练成功完成！")
    else:
        print("\n训练失败！")