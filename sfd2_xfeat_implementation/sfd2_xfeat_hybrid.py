"""
sfd2 + xfeat混合网络实现
用xfeat的轻量级骨干网络替换sfd2中的ressegnetv2
保留sfd2的语义引导机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from typing import dict, list, tuple, optional


class basiclayer(nn.module):
    """
    基础卷积层：conv2d -> batchnorm -> relu
    基于xfeat的实现
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, dilation=1, bias=false):
        super().__init__()
        self.layer = nn.sequential(
            nn.conv2d(in_channels, out_channels, kernel_size, 
                     padding=padding, stride=stride, dilation=dilation, bias=bias),
            nn.batchnorm2d(out_channels, affine=false),
            nn.relu(inplace=true),
        )

    def forward(self, x):
        return self.layer(x)


class semanticguidancemodule(nn.module):
    """
    sfd2语义引导模块
    基于sfd2的语义分割思想
    """
    def __init__(self, in_channels, semantic_channels=20):
        super().__init__()
        
        # 语义分割分支
        self.semantic_branch = nn.sequential(
            nn.conv2d(in_channels, 64, 3, padding=1),
            nn.batchnorm2d(64),
            nn.relu(inplace=true),
            nn.conv2d(64, semantic_channels, 1),
        )
        
        # 语义置信度生成
        self.confidence_branch = nn.sequential(
            nn.conv2d(semantic_channels, 32, 1),
            nn.relu(inplace=true),
            nn.conv2d(32, 1, 1),
            nn.sigmoid()
        )
        
        # 特征调制分支
        self.modulation_branch = nn.sequential(
            nn.conv2d(in_channels + semantic_channels, 64, 3, padding=1),
            nn.batchnorm2d(64),
            nn.relu(inplace=true),
            nn.conv2d(64, in_channels, 1),
            nn.sigmoid()
        )
        
    def forward(self, x):
        # 生成语义图
        semantic_map = self.semantic_branch(x)
        semantic_probs = f.softmax(semantic_map, dim=1)
        
        # 生成语义置信度
        semantic_confidence = self.confidence_branch(semantic_probs)
        
        # 特征调制
        concat_feat = torch.cat([x, semantic_probs], dim=1)
        modulation_map = self.modulation_branch(concat_feat)
        
        # 应用语义调制
        enhanced_feat = x * modulation_map
        
        return enhanced_feat, semantic_probs, semantic_confidence


class xfeatsfd2backbone(nn.module):
    """
    xfeat骨干网络 + sfd2语义引导
    结合xfeat的轻量级架构和sfd2的语义引导机制
    """
    def __init__(self, outdim=128, semantic_channels=20, require_stability=false):
        super().__init__()
        self.outdim = outdim
        self.semantic_channels = semantic_channels
        self.require_stability = require_stability
        
        # xfeat预处理
        self.norm = nn.instancenorm2d(1)
        self.skip1 = nn.sequential(
            nn.avgpool2d(4, stride=4),
            nn.conv2d(1, 24, 1, stride=1, padding=0)
        )
        
        # xfeat block 1
        self.block1 = nn.sequential(
            basiclayer(1, 4, stride=1),
            basiclayer(4, 8, stride=2),
            basiclayer(8, 8, stride=1),
            basiclayer(8, 24, stride=2),
        )
        
        # sfd2语义引导模块
        self.semantic_guidance = semanticguidancemodule(24, semantic_channels)
        
        # xfeat block 2
        self.block2 = nn.sequential(
            basiclayer(24, 24, stride=1),
            basiclayer(24, 24, stride=1),
        )
        
        # xfeat block 3
        self.block3 = nn.sequential(
            basiclayer(24, 64, stride=2),
            basiclayer(64, 64, stride=1),
            basiclayer(64, 64, 1, padding=0),
        )
        
        # xfeat block 4
        self.block4 = nn.sequential(
            basiclayer(64, 64, stride=2),
            basiclayer(64, 64, stride=1),
            basiclayer(64, 64, stride=1),
        )
        
        # xfeat block 5
        self.block5 = nn.sequential(
            basiclayer(64, 128, stride=2),
            basiclayer(128, 128, stride=1),
            basiclayer(128, 128, stride=1),
            basiclayer(128, 64, 1, padding=0),
        )
        
        # 特征融合
        self.block_fusion = nn.sequential(
            basiclayer(64, 64, stride=1),
            basiclayer(64, 64, stride=1),
            nn.conv2d(64, 64, 1, padding=0)
        )
        
        # sfd2检测头
        self.convpa = nn.sequential(
            nn.conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.batchnorm2d(64),
            nn.relu(inplace=true),
            nn.conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )
        
        self.convda = nn.sequential(
            nn.conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.batchnorm2d(64),
            nn.relu(inplace=true),
            nn.conv2d(64, outdim, kernel_size=3, stride=1, padding=1)
        )
        
        self.convpb = nn.conv2d(64, 65, kernel_size=1, stride=1, padding=0)
        self.convdb = nn.conv2d(64, outdim, kernel_size=1, stride=1, padding=0)
        
        if self.require_stability:
            self.convsta = nn.conv2d(64, 1, kernel_size=1)
        
    def det(self, x):
        # 预处理
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=true)
            x = self.norm(x)
        
        # xfeat主干
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        
        # sfd2语义引导
        x2_enhanced, semantic_attention, semantic_confidence = self.semantic_guidance(x2)
        
        # 继续xfeat处理
        x3 = self.block3(x2_enhanced)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        
        # 特征融合
        x4 = f.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = f.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        fused = self.block_fusion(x3 + x4 + x5)
        
        # sfd2检测头
        cpa = self.convpa(fused)
        semi = self.convpb(cpa)
        semi = torch.exp(semi)
        semi_norm = semi / (torch.sum(semi, dim=1, keepdim=true) + .00001)
        score = semi_norm[:, :-1, :, :]
        hc, wc = score.size(2), score.size(3)
        score = score.permute([0, 2, 3, 1])
        score = score.view(score.size(0), hc, wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), 1, hc * 8, wc * 8)
        
        # 描述符头
        cda = self.convda(fused)
        desc = self.convdb(cda)
        desc = f.normalize(desc, dim=1)
        
        # 稳定性头（如果需要）
        if self.require_stability:
            stability = torch.sigmoid(self.convsta(fused))
            stability = f.interpolate(stability, size=(x.shape[2], x.shape[3]), mode='bilinear')
        else:
            stability = none
        
        return score, stability, desc, semantic_attention, semantic_confidence
    
    def det_train(self, x):
        # 训练时的前向传播
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=true)
            x = self.norm(x)
        
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        
        # sfd2语义引导
        x2_enhanced, semantic_attention, semantic_confidence = self.semantic_guidance(x2)
        
        x3 = self.block3(x2_enhanced)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        
        # 特征融合
        x4 = f.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = f.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        fused = self.block_fusion(x3 + x4 + x5)
        
        return fused, semantic_attention, semantic_confidence


class xfeatsfd2model(nn.module):
    """
    完整的xfeat + sfd2语义引导模型
    """
    def __init__(self, outdim=128, semantic_channels=20, require_stability=false):
        super().__init__()
        self.backbone = xfeatsfd2backbone(
            outdim=outdim,
            semantic_channels=semantic_channels,
            require_stability=require_stability
        )
        
    def forward(self, x):
        return self.backbone.det(x)
    
    def detect_and_compute(self, x, threshold=0.5):
        """
        检测和计算特征点
        """
        score, stability, desc, semantic_attention, semantic_confidence = self.backbone.det(x)
        
        # 获取关键点
        b, _, h, w = score.shape
        score_flat = score.view(b, -1)
        
        # 选择超过阈值的点
        keypoints = []
        scores = []
        descriptors = []
        
        for i in range(b):
            # 找到超过阈值的点
            valid_mask = score_flat[i] > threshold
            valid_indices = torch.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                # 转换为坐标
                y_coords = valid_indices // w
                x_coords = valid_indices % w
                
                keypoints.append(torch.stack([x_coords, y_coords], dim=1))
                scores.append(score_flat[i][valid_indices])
                
                # 获取描述符
                valid_desc = desc[i, :, y_coords, x_coords].t()
                descriptors.append(valid_desc)
            else:
                keypoints.append(torch.empty(0, 2))
                scores.append(torch.empty(0))
                descriptors.append(torch.empty(0, desc.shape[1]))
        
        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
            'semantic_attention': semantic_attention,
            'semantic_confidence': semantic_confidence,
            'stability': stability
        }


# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = xfeatsfd2model(outdim=128, semantic_channels=20)
    model.eval()
    
    # 测试输入
    batch_size = 2
    height, width = 480, 640
    x = torch.randn(batch_size, 3, height, width)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(x)
        
    print("模型输出:")
    print(f"score shape: {outputs[0].shape}")
    print(f"stability: {outputs[1] is not none}")
    print(f"desc shape: {outputs[2].shape}")
    print(f"semantic_attention shape: {outputs[3].shape}")
    print(f"semantic_confidence shape: {outputs[4].shape}")
    
    # 测试特征检测
    features = model.detect_and_compute(x, threshold=0.5)
    print("\n检测到的特征:")
    print(f"关键点数量: {[len(kp) for kp in features['keypoints']]}")
    print(f"描述符形状: {[desc.shape for desc in features['descriptors']]}")