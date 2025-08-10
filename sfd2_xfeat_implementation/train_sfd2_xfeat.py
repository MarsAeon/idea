"""
SFD2 + XFeat混合网络训练脚本
用XFeat的轻量级骨干网络替换SFD2中的ResSegNetV2
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse

# 导入模型
from sfd2_xfeat_hybrid import XFeatSFD2Model

# 导入SFD2的训练相关模块
sys.path.append('./sfd2')
from nets.losses import SegLoss
from datasets.dataset import ImagePairDataset
from tools.dataloader import collate_fn


class XFeatSFD2Trainer:
    """
    XFeat + SFD2混合网络训练器
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = XFeatSFD2Model(
            outdim=config.get('outdim', 128),
            semantic_channels=config.get('semantic_channels', 20),
            require_stability=config.get('require_stability', False)
        ).to(self.device)
        
        # 创建损失函数 - 使用SFD2的SegLoss
        self.criterion = SegLoss(
            desc_loss_fn='wap',
            weights={'wdet': config.get('det_weight', 1.0), 
                     'wdesc': config.get('desc_weight', 1.0)},
            det_loss='bce',
            seg_desc_loss_fn='wap',
            seg_desc=config.get('seg_desc', True),
            seg_feat=config.get('seg_feat', False),
            seg_det=config.get('seg_det', True),
            seg_cls=config.get('seg_cls', False),
            margin=config.get('margin', 1.0)
        ).to(self.device)
        
        # 创建优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-6)
        )
        
        # 创建学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get('lr_step_size', 30),
            gamma=config.get('lr_gamma', 0.1)
        )
        
        # 创建数据加载器
        self.train_loader = self._create_data_loader('train')
        self.val_loader = self._create_data_loader('val')
        
        # 创建日志目录
        self.log_dir = Path(config.get('log_dir', './logs/sfd2_xfeat'))
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建检查点目录
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints/sfd2_xfeat'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def _create_data_loader(self, split):
        """创建数据加载器"""
        dataset = ImagePairDataset(
            data_root=self.config.get('data_root', './data'),
            split=split,
            config=self.config
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 4),
            shuffle=(split == 'train'),
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        det_loss = 0.0
        desc_loss = 0.0
        sem_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 准备数据
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            
            # 准备损失函数输入
            loss_input = {
                'desc': outputs[2],  # 描述符
                'gt_score': labels,   # 真实分数
                'score_th': 0.01,     # 分数阈值
                'seg_mask': torch.ones_like(labels),  # 分割掩码
                'seg': torch.zeros_like(labels).long(),  # 语义标签
                'seg_confidence': outputs[4]  # 语义置信度
            }
            
            # 计算损失
            loss_dict = self.criterion(loss_input)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss_dict['total_loss'].item()
            det_loss += loss_dict.get('det_loss', 0).item()
            desc_loss += loss_dict.get('desc_loss', 0).item()
            sem_loss += loss_dict.get('sem_loss', 0).item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss_dict['total_loss'].item(),
                'det': loss_dict.get('det_loss', 0).item(),
                'desc': loss_dict.get('desc_loss', 0).item(),
                'sem': loss_dict.get('sem_loss', 0).item()
            })
        
        # 计算平均损失
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_det_loss = det_loss / num_batches
        avg_desc_loss = desc_loss / num_batches
        avg_sem_loss = sem_loss / num_batches
        
        return {
            'total_loss': avg_loss,
            'det_loss': avg_det_loss,
            'desc_loss': avg_desc_loss,
            'sem_loss': avg_sem_loss
        }
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        det_loss = 0.0
        desc_loss = 0.0
        sem_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # 准备数据
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                
                # 准备损失函数输入
                loss_input = {
                    'desc': outputs[2],  # 描述符
                    'gt_score': labels,   # 真实分数
                    'score_th': 0.01,     # 分数阈值
                    'seg_mask': torch.ones_like(labels),  # 分割掩码
                    'seg': torch.zeros_like(labels).long(),  # 语义标签
                    'seg_confidence': outputs[4]  # 语义置信度
                }
                
                # 计算损失
                loss_dict = self.criterion(loss_input)
                
                # 累计损失
                total_loss += loss_dict['total_loss'].item()
                det_loss += loss_dict.get('det_loss', 0).item()
                desc_loss += loss_dict.get('desc_loss', 0).item()
                sem_loss += loss_dict.get('sem_loss', 0).item()
        
        # 计算平均损失
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        avg_det_loss = det_loss / num_batches
        avg_desc_loss = desc_loss / num_batches
        avg_sem_loss = sem_loss / num_batches
        
        return {
            'total_loss': avg_loss,
            'det_loss': avg_det_loss,
            'desc_loss': avg_desc_loss,
            'sem_loss': avg_sem_loss
        }
    
    def train(self):
        """训练模型"""
        best_val_loss = float('inf')
        
        for epoch in range(self.config.get('epochs', 40)):
            print(f'\nEpoch {epoch + 1}/{self.config["epochs"]}')
            
            # 训练
            train_losses = self.train_epoch(epoch)
            
            # 验证
            val_losses = self.validate_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 打印损失
            print(f'Train Loss: {train_losses["total_loss"]:.4f} '
                  f'(Det: {train_losses["det_loss"]:.4f}, '
                  f'Desc: {train_losses["desc_loss"]:.4f}, '
                  f'Sem: {train_losses["sem_loss"]:.4f})')
            print(f'Val Loss: {val_losses["total_loss"]:.4f} '
                  f'(Det: {val_losses["det_loss"]:.4f}, '
                  f'Desc: {val_losses["desc_loss"]:.4f}, '
                  f'Sem: {val_losses["sem_loss"]:.4f})')
            
            # 保存最佳模型
            if val_losses['total_loss'] < best_val_loss:
                best_val_loss = val_losses['total_loss']
                self.save_checkpoint(epoch, is_best=True)
            
            # 定期保存检查点
            if epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch)
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch']


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train SFD2 + XFeat Hybrid Model')
    parser.add_argument('--config', type=str, default='config_sfd2_xfeat.json',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 更新配置
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['learning_rate'] = args.lr
    
    # 创建训练器
    trainer = XFeatSFD2Trainer(config)
    
    # 恢复训练（如果指定）
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f'Resumed training from epoch {start_epoch}')
    else:
        start_epoch = 0
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()