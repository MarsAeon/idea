"""
语义引导的轻量级特征检测器训练脚本
基于SFD2和XFeat的结合实现
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import wandb
from datetime import datetime

# 导入自定义模块
from semantic_guided_xfeat_implementation import (
    SemanticGuidedXFeat, 
    SemanticGuidedLoss,
    ModelQuantization
)


class SemanticFeatureDataset(Dataset):
    """
    语义特征数据集
    结合图像对和语义标注
    """
    def __init__(self, image_pairs, semantic_labels, transform=None):
        self.image_pairs = image_pairs
        self.semantic_labels = semantic_labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_pairs)
        
    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        sem1_path, sem2_path = self.semantic_labels[idx]
        
        # 加载图像
        img1 = self.load_image(img1_path)
        img2 = self.load_image(img2_path)
        
        # 加载语义标签
        sem1 = self.load_semantic(sem1_path)
        sem2 = self.load_semantic(sem2_path)
        
        # 生成匹配标签（简化版本）
        matches = self.generate_matches(img1, img2)
        
        return {
            'image1': img1,
            'image2': img2,
            'semantic1': sem1,
            'semantic2': sem2,
            'matches': matches
        }
        
    def load_image(self, path):
        """加载图像"""
        # 简化实现，实际使用时需要完整的图像加载逻辑
        return torch.randn(3, 256, 256)
        
    def load_semantic(self, path):
        """加载语义标签"""
        # 简化实现，返回20类语义分割图
        return torch.randint(0, 20, (256, 256)).long()
        
    def generate_matches(self, img1, img2):
        """生成匹配标签（简化版本）"""
        # 实际实现需要使用SFM或其他方法生成真实匹配
        return torch.randint(0, 100, (50, 2))


class SemanticFeatureTrainer:
    """
    语义特征训练器
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = SemanticGuidedXFeat(
            semantic_channels=config['semantic_channels'],
            top_k=config['top_k'],
            detection_threshold=config['detection_threshold']
        )
        self.model.to(self.device)
        
        # 损失函数
        self.criterion = SemanticGuidedLoss(config)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['min_lr']
        )
        
        # 数据加载器
        self.train_loader = self.create_data_loader('train')
        self.val_loader = self.create_data_loader('val')
        
        # 日志记录
        self.setup_logging()
        
    def create_data_loader(self, split):
        """创建数据加载器"""
        # 简化实现
        image_pairs = [('img1.jpg', 'img2.jpg')] * 100
        semantic_labels = [('sem1.png', 'sem2.png')] * 100
        
        dataset = SemanticFeatureDataset(image_pairs, semantic_labels)
        
        batch_size = self.config['batch_size'] if split == 'train' else 1
        shuffle = split == 'train'
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
    def setup_logging(self):
        """设置日志记录"""
        if self.config.get('use_wandb', False):
            wandb.init(
                project="semantic-guided-xfeat",
                config=self.config,
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
        # 创建输出目录
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 数据预处理
            img1 = batch['image1'].to(self.device)
            img2 = batch['image2'].to(self.device)
            sem1 = batch['semantic1'].to(self.device)
            sem2 = batch['semantic2'].to(self.device)
            matches = batch['matches'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # 处理第一张图像
            outputs1 = self.model(img1)
            features1, keypoint_logits1, heatmap1, descriptors1, semantic_attention1 = outputs1
            
            # 处理第二张图像
            outputs2 = self.model(img2)
            features2, keypoint_logits2, heatmap2, descriptors2, semantic_attention2 = outputs2
            
            # 构建目标
            targets = {
                'keypoints_target': self.generate_keypoint_targets(keypoint_logits1),
                'matches': matches,
                'semantic_target': sem1
            }
            
            # 构建模型输出
            model_outputs = {
                'keypoints': keypoint_logits1,
                'descriptors': descriptors1,
                'semantic_attention': semantic_attention1
            }
            
            # 计算损失
            losses = self.criterion(model_outputs, targets)
            loss = losses['total_loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # 日志记录
            if batch_idx % self.config['log_interval'] == 0:
                self.log_step(epoch, batch_idx, losses)
                
        return total_loss / len(self.train_loader)
        
    def generate_keypoint_targets(self, keypoint_logits):
        """生成关键点目标（简化版本）"""
        # 实际实现需要使用真实的关键点标注
        B, C, H, W = keypoint_logits.shape
        return torch.randint(0, C, (B, H, W), device=keypoint_logits.device)
        
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                img1 = batch['image1'].to(self.device)
                img2 = batch['image2'].to(self.device)
                sem1 = batch['semantic1'].to(self.device)
                sem2 = batch['semantic2'].to(self.device)
                matches = batch['matches'].to(self.device)
                
                # 前向传播
                outputs1 = self.model(img1)
                outputs2 = self.model(img2)
                
                # 构建目标和输出
                targets = {
                    'keypoints_target': self.generate_keypoint_targets(outputs1[1]),
                    'matches': matches,
                    'semantic_target': sem1
                }
                
                model_outputs = {
                    'keypoints': outputs1[1],
                    'descriptors': outputs1[3],
                    'semantic_attention': outputs1[4]
                }
                
                # 计算损失
                losses = self.criterion(model_outputs, targets)
                total_loss += losses['total_loss'].item()
                
        return total_loss / len(self.val_loader)
        
    def log_step(self, epoch, batch_idx, losses):
        """记录训练步骤"""
        if self.config.get('use_wandb', False):
            wandb.log({
                'epoch': epoch,
                'batch': batch_idx,
                'total_loss': losses['total_loss'],
                'det_loss': losses.get('det_loss', 0),
                'desc_loss': losses.get('desc_loss', 0),
                'seg_loss': losses.get('seg_loss', 0),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
    def save_checkpoint(self, epoch, best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        if best:
            checkpoint_path = self.output_dir / 'best_model.pth'
        else:
            checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
            
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
    def train(self):
        """完整训练流程"""
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 打印结果
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, best=True)
                
            # 定期保存检查点
            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch)
                
        print("Training completed!")
        
    def quantize_model(self):
        """量化模型"""
        quantizer = ModelQuantization(self.model)
        quantized_model = quantizer.quantize_model()
        
        # 保存量化模型
        quantized_path = self.output_dir / 'quantized_model.pth'
        torch.save(quantized_model.state_dict(), quantized_path)
        print(f"Quantized model saved to {quantized_path}")
        
        return quantized_model


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_default_config():
    """创建默认配置"""
    config = {
        # 模型参数
        'semantic_channels': 20,
        'top_k': 4096,
        'detection_threshold': 0.05,
        
        # 训练参数
        'batch_size': 16,
        'epochs': 100,
        'learning_rate': 1e-4,
        'min_lr': 1e-6,
        'weight_decay': 1e-4,
        
        # 数据参数
        'num_workers': 4,
        'image_size': 256,
        
        # 日志参数
        'log_interval': 10,
        'save_interval': 10,
        'use_wandb': False,
        
        # 输出目录
        'output_dir': './outputs',
        
        # 损失权重
        'det_weight': 1.0,
        'desc_weight': 1.0,
        'seg_weight': 0.5,
    }
    return config


def main():
    """主函数"""
    # 加载配置
    config_path = "config_train_semantic_xfeat.json"
    
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = create_default_config()
        # 保存默认配置
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
    print("Training configuration:")
    print(json.dumps(config, indent=2))
    
    # 创建训练器
    trainer = SemanticFeatureTrainer(config)
    
    # 开始训练
    trainer.train()
    
    # 模型量化
    if config.get('quantize', False):
        print("\nQuantizing model...")
        quantized_model = trainer.quantize_model()
        
    print("\nTraining and quantization completed!")


if __name__ == "__main__":
    main()