"""
数据集处理模块
为语义引导的轻量级特征检测器提供数据加载和预处理功能
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImageTransforms:
    """
    图像变换类
    提供训练和推理时的图像预处理
    """
    def __init__(self, image_size=256, is_training=True):
        self.image_size = image_size
        self.is_training = is_training
        
        if is_training:
            # 训练时的数据增强
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.RandomRotate90(),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.1),
                A.OneOf([
                    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
                ], p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            # 推理时的预处理
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __call__(self, image):
        """
        应用图像变换
        """
        if isinstance(image, str):
            image = np.array(Image.open(image))
        elif isinstance(image, Image.Image):
            image = np.array(image)
            
        return self.transform(image=image)['image']


class SemanticTransforms:
    """
    语义标注变换类
    处理语义分割标签的变换
    """
    def __init__(self, image_size=256, num_classes=20, is_training=True):
        self.image_size = image_size
        self.num_classes = num_classes
        self.is_training = is_training
        
        if is_training:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.RandomRotate90(),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
            ])
    
    def __call__(self, semantic_map):
        """
        应用语义标注变换
        """
        if isinstance(semantic_map, str):
            semantic_map = np.array(Image.open(semantic_map))
        elif isinstance(semantic_map, Image.Image):
            semantic_map = np.array(semantic_map)
            
        transformed = self.transform(image=semantic_map)
        return transformed['image']


class MegaDepthDataset(Dataset):
    """
    MegaDepth数据集加载器
    适用于特征检测和匹配任务
    """
    def __init__(self, root_dir, scene_list=None, image_size=256, 
                 max_pairs=1000, is_training=True):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.is_training = is_training
        self.max_pairs = max_pairs
        
        # 图像变换
        self.image_transform = ImageTransforms(image_size, is_training)
        
        # 加载场景列表
        if scene_list is None:
            self.scene_list = self._load_scene_list()
        else:
            self.scene_list = scene_list
            
        # 加载图像对
        self.image_pairs = self._load_image_pairs()
        
    def _load_scene_list(self):
        """
        加载场景列表
        """
        scenes_dir = self.root_dir / "scenes"
        if not scenes_dir.exists():
            raise FileNotFoundError(f"Scenes directory not found: {scenes_dir}")
            
        scene_list = []
        for scene_dir in scenes_dir.iterdir():
            if scene_dir.is_dir():
                scene_list.append(scene_dir.name)
                
        return scene_list
    
    def _load_image_pairs(self):
        """
        加载图像对
        """
        image_pairs = []
        
        for scene_name in self.scene_list:
            scene_dir = self.root_dir / "scenes" / scene_name
            
            # 查找图像文件
            images_dir = scene_dir / "images"
            if not images_dir.exists():
                continue
                
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            image_files = sorted([str(f) for f in image_files])
            
            # 生成图像对
            for i in range(len(image_files) - 1):
                if len(image_pairs) >= self.max_pairs:
                    break
                image_pairs.append((image_files[i], image_files[i + 1]))
                    
            if len(image_pairs) >= self.max_pairs:
                break
                
        return image_pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        
        # 加载图像
        img1 = self.image_transform(img1_path)
        img2 = self.image_transform(img2_path)
        
        # 生成伪语义标签（MegaDepth没有真实语义标签）
        semantic1 = torch.randint(0, 20, (self.image_size, self.image_size))
        semantic2 = torch.randint(0, 20, (self.image_size, self.image_size))
        
        # 生成伪匹配标签
        matches = torch.randint(0, 100, (50, 2))
        
        return {
            'image1': img1,
            'image2': img2,
            'semantic1': semantic1,
            'semantic2': semantic2,
            'matches': matches,
            'image1_path': img1_path,
            'image2_path': img2_path
        }


class HPatchesDataset(Dataset):
    """
    HPatches数据集加载器
    适用于特征检测和匹配评估
    """
    def __init__(self, root_dir, image_size=256, is_training=True):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.is_training = is_training
        
        # 图像变换
        self.image_transform = ImageTransforms(image_size, is_training)
        
        # 加载序列
        self.sequences = self._load_sequences()
        
    def _load_sequences(self):
        """
        加载HPatches序列
        """
        sequences = []
        
        for seq_dir in self.root_dir.iterdir():
            if seq_dir.is_dir():
                sequence = {
                    'name': seq_dir.name,
                    'images': [],
                    'homographies': []
                }
                
                # 加载图像
                for i in range(1, 6):  # HPatches通常有5张图像
                    img_path = seq_dir / f"{i}.ppm"
                    if img_path.exists():
                        sequence['images'].append(str(img_path))
                        
                # 加载单应性矩阵
                for i in range(2, 6):
                    H_path = seq_dir / f"H_1_{i}"
                    if H_path.exists():
                        H = np.loadtxt(H_path)
                        sequence['homographies'].append(H)
                        
                if len(sequence['images']) > 0:
                    sequences.append(sequence)
                    
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # 加载第一张图像作为参考
        ref_img = self.image_transform(sequence['images'][0])
        
        # 随机选择一张图像进行匹配
        if len(sequence['images']) > 1:
            img_idx = np.random.randint(1, len(sequence['images']))
            tgt_img = self.image_transform(sequence['images'][img_idx])
            
            # 获取对应的单应性矩阵
            if img_idx - 1 < len(sequence['homographies']):
                homography = sequence['homographies'][img_idx - 1]
            else:
                homography = np.eye(3)
        else:
            tgt_img = ref_img
            homography = np.eye(3)
            
        # 生成伪语义标签
        semantic1 = torch.randint(0, 20, (self.image_size, self.image_size))
        semantic2 = torch.randint(0, 20, (self.image_size, self.image_size))
        
        return {
            'image1': ref_img,
            'image2': tgt_img,
            'semantic1': semantic1,
            'semantic2': semantic2,
            'homography': torch.from_numpy(homography).float(),
            'sequence_name': sequence['name']
        }


class CityscapesDataset(Dataset):
    """
    Cityscapes数据集加载器
    提供真实的语义分割标签
    """
    def __init__(self, root_dir, split='train', image_size=256, 
                 max_samples=1000, is_training=True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.max_samples = max_samples
        self.is_training = is_training
        
        # 图像和语义变换
        self.image_transform = ImageTransforms(image_size, is_training)
        self.semantic_transform = SemanticTransforms(image_size, num_classes=19, is_training=is_training)
        
        # 加载图像-语义对
        self.image_semantic_pairs = self._load_image_semantic_pairs()
        
    def _load_image_semantic_pairs(self):
        """
        加载图像和语义标注对
        """
        split_dir = self.root_dir / "leftImg8bit" / self.split
        semantic_dir = self.root_dir / "gtFine" / self.split
        
        if not split_dir.exists() or not semantic_dir.exists():
            raise FileNotFoundError(f"Cityscapes split directory not found: {split_dir}")
            
        pairs = []
        
        for city_dir in split_dir.iterdir():
            if not city_dir.is_dir():
                continue
                
            city_name = city_dir.name
            semantic_city_dir = semantic_dir / city_name
            
            if not semantic_city_dir.exists():
                continue
                
            # 查找图像文件
            for img_file in city_dir.glob("*_leftImg8bit.png"):
                # 对应的语义标注文件
                base_name = img_file.stem.replace("_leftImg8bit", "")
                semantic_file = semantic_city_dir / f"{base_name}_gtFine_labelIds.png"
                
                if semantic_file.exists():
                    pairs.append((str(img_file), str(semantic_file)))
                    
                    if len(pairs) >= self.max_samples:
                        break
                        
            if len(pairs) >= self.max_samples:
                break
                
        return pairs
    
    def __len__(self):
        return len(self.image_semantic_pairs)
    
    def __getitem__(self, idx):
        img_path, sem_path = self.image_semantic_pairs[idx]
        
        # 加载图像
        image = self.image_transform(img_path)
        
        # 加载语义标注
        semantic = self.semantic_transform(sem_path)
        semantic = torch.from_numpy(semantic).long()
        
        return {
            'image': image,
            'semantic': semantic,
            'image_path': img_path,
            'semantic_path': sem_path
        }


class SemanticFeatureDataset(Dataset):
    """
    语义特征数据集
    结合多个数据源，为语义引导特征检测器提供训练数据
    """
    def __init__(self, datasets, weights=None, is_training=True):
        self.datasets = datasets
        self.is_training = is_training
        
        # 计算数据集权重
        if weights is None:
            self.weights = [1.0 / len(datasets)] * len(datasets)
        else:
            self.weights = weights
            
        # 计算累积权重
        self.cum_weights = np.cumsum(self.weights)
        
    def __len__(self):
        return max(len(dataset) for dataset in self.datasets)
    
    def __getitem__(self, idx):
        # 根据权重随机选择数据集
        rand_val = np.random.random()
        dataset_idx = np.searchsorted(self.cum_weights, rand_val)
        dataset_idx = min(dataset_idx, len(self.datasets) - 1)
        
        # 从选定的数据集中获取样本
        dataset = self.datasets[dataset_idx]
        sample_idx = idx % len(dataset)
        
        sample = dataset[sample_idx]
        
        # 确保返回的数据格式一致
        if 'image1' not in sample:
            # 如果是单图像数据集，创建图像对
            sample['image1'] = sample['image']
            sample['image2'] = sample['image']
            sample['semantic1'] = sample['semantic']
            sample['semantic2'] = sample['semantic']
            
        if 'matches' not in sample:
            # 如果没有匹配标签，生成伪匹配
            sample['matches'] = torch.randint(0, 100, (50, 2))
            
        return sample


class DataModule:
    """
    数据模块
    统一管理数据加载器的创建和配置
    """
    def __init__(self, config):
        self.config = config
        self.batch_size = config.get('batch_size', 16)
        self.num_workers = config.get('num_workers', 4)
        self.image_size = config.get('image_size', 256)
        
    def create_train_dataloader(self):
        """
        创建训练数据加载器
        """
        datasets = []
        weights = []
        
        # 添加MegaDepth数据集
        if 'megadepth_path' in self.config:
            megadepth_dataset = MegaDepthDataset(
                root_dir=self.config['megadepth_path'],
                image_size=self.image_size,
                is_training=True
            )
            datasets.append(megadepth_dataset)
            weights.append(0.6)  # 60%权重
            
        # 添加Cityscapes数据集
        if 'cityscapes_path' in self.config:
            cityscapes_dataset = CityscapesDataset(
                root_dir=self.config['cityscapes_path'],
                split='train',
                image_size=self.image_size,
                is_training=True
            )
            datasets.append(cityscapes_dataset)
            weights.append(0.4)  # 40%权重
            
        if len(datasets) == 0:
            raise ValueError("No training datasets specified in config")
            
        # 创建组合数据集
        combined_dataset = SemanticFeatureDataset(
            datasets=datasets,
            weights=weights,
            is_training=True
        )
        
        return DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def create_val_dataloader(self):
        """
        创建验证数据加载器
        """
        datasets = []
        
        # 添加HPatches数据集用于验证
        if 'hpatches_path' in self.config:
            hpatches_dataset = HPatchesDataset(
                root_dir=self.config['hpatches_path'],
                image_size=self.image_size,
                is_training=False
            )
            datasets.append(hpatches_dataset)
            
        if len(datasets) == 0:
            # 如果没有验证数据集，使用训练数据集的一部分
            train_datasets = []
            if 'megadepth_path' in self.config:
                train_datasets.append(MegaDepthDataset(
                    root_dir=self.config['megadepth_path'],
                    image_size=self.image_size,
                    max_pairs=100,  # 减少验证数据量
                    is_training=False
                ))
                
            if len(train_datasets) == 0:
                raise ValueError("No validation datasets available")
                
            combined_dataset = SemanticFeatureDataset(
                datasets=train_datasets,
                is_training=False
            )
        else:
            combined_dataset = SemanticFeatureDataset(
                datasets=datasets,
                is_training=False
            )
            
        return DataLoader(
            combined_dataset,
            batch_size=1,  # 验证时使用batch_size=1
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def create_test_dataloader(self):
        """
        创建测试数据加载器
        """
        return self.create_val_dataloader()  # 测试和验证使用相同的数据加载器


def create_data_module(config):
    """
    创建数据模块的工厂函数
    """
    return DataModule(config)


def test_dataset_loading():
    """
    测试数据集加载功能
    """
    print("Testing dataset loading...")
    
    # 创建测试配置
    test_config = {
        'batch_size': 2,
        'num_workers': 0,
        'image_size': 256,
        'megadepth_path': 'data/megadepth',  # 假设的路径
        'cityscapes_path': 'data/cityscapes',  # 假设的路径
        'hpatches_path': 'data/hpatches',  # 假设的路径
    }
    
    try:
        # 创建数据模块
        data_module = create_data_module(test_config)
        
        # 测试训练数据加载器
        print("Creating train dataloader...")
        train_loader = data_module.create_train_dataloader()
        print(f"Train dataloader created with {len(train_loader)} batches")
        
        # 测试验证数据加载器
        print("Creating validation dataloader...")
        val_loader = data_module.create_val_dataloader()
        print(f"Validation dataloader created with {len(val_loader)} batches")
        
        # 尝试加载一个批次
        print("Loading a batch from train dataloader...")
        try:
            batch = next(iter(train_loader))
            print(f"Batch keys: {list(batch.keys())}")
            print(f"Image1 shape: {batch['image1'].shape}")
            print(f"Image2 shape: {batch['image2'].shape}")
            print(f"Semantic1 shape: {batch['semantic1'].shape}")
            print(f"Semantic2 shape: {batch['semantic2'].shape}")
            print(f"Matches shape: {batch['matches'].shape}")
            print("Dataset loading test passed!")
        except Exception as e:
            print(f"Error loading batch: {e}")
            print("This is expected if the dataset paths don't exist.")
            
    except Exception as e:
        print(f"Error creating data module: {e}")
        print("This is expected if the dataset paths don't exist.")


if __name__ == "__main__":
    test_dataset_loading()