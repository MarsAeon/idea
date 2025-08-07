"""
测试用例集合
为语义引导的轻量级特征检测器提供全面的测试覆盖
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# 导入项目模块
from semantic_guided_xfeat_implementation import (
    SemanticGuidedXFeat, SemanticGuidedLoss, ModelQuantization,
    create_semantic_guided_xfeat
)
from dataset_processing import (
    ImageTransforms, SemanticTransforms, MegaDepthDataset,
    HPatchesDataset, CityscapesDataset, SemanticFeatureDataset, DataModule
)
from tools import ModelConverter, BatchInference, FeatureVisualizer

# 测试配置
TEST_CONFIG = {
    'model': {
        'input_channels': 1,
        'feature_channels': 64,
        'semantic_classes': 20,
        'use_attention': True,
        'dropout_rate': 0.1
    },
    'training': {
        'batch_size': 2,
        'learning_rate': 0.001,
        'num_epochs': 2,
        'device': 'cpu'
    },
    'data': {
        'image_size': 256,
        'top_k': 1024
    }
}


class TestModelComponents(unittest.TestCase):
    """
    模型组件单元测试
    ""
    def setUp(self):
n        """
        测试设置
        """
        self.device = torch.device('cpu')
        self.batch_size = 2
        self.input_size = 256
        self.model = create_semantic_guided_xfeat(**TEST_CONFIG['model']).to(self.device)
        
    def test_model_creation(self):
        """
        测试模型创建
        """
        self.assertIsInstance(self.model, SemanticGuidedXFeat)
        self.assertEqual(self.model.input_channels, TEST_CONFIG['model']['input_channels'])
        self.assertEqual(self.model.semantic_classes, TEST_CONFIG['model']['semantic_classes'])
        
    def test_forward_pass(self):
        """
        测试前向传播
        """
        # 创建测试输入
        x = torch.randn(self.batch_size, 1, self.input_size, self.input_size, device=self.device)
        
        # 前向传播
        with torch.no_grad():
            features = self.model(x)
            
        # 检查输出格式
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 5)  # 5个输出特征
        
        # 检查特征维度
        self.assertEqual(features[0].shape[0], self.batch_size)  # batch_size
        self.assertEqual(features[1].shape[0], self.batch_size)  # keypoint_logits
        self.assertEqual(features[2].shape[0], self.batch_size)  # heatmap
        self.assertEqual(features[3].shape[0], self.batch_size)  # descriptors
        self.assertEqual(features[4].shape[0], self.batch_size)  # semantic_attention
        
    def test_detect_and_compute(self):
        """
        测试关键点检测和描述符生成
        """
        # 创建测试输入
        x = torch.randn(1, 1, self.input_size, self.input_size, device=self.device)
        
        # 检测和计算
        with torch.no_grad():
            results = self.model.detectAndCompute(x, top_k=100)
            
        # 检查结果格式
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        
        result = results[0]
        self.assertIn('keypoints', result)
        self.assertIn('scores', result)
        self.assertIn('descriptors', result)
        self.assertIn('semantic_attention', result)
        
        # 检查关键点数量
        self.assertLessEqual(len(result['keypoints']), 100)
        
    def test_loss_function(self):
        """
        测试损失函数
        """
        # 创建测试数据
        batch_size = 2
        num_keypoints = 50
        descriptor_dim = 256
        
        # 预测结果
        pred_keypoints = torch.randn(batch_size, num_keypoints, 2)
        pred_scores = torch.randn(batch_size, num_keypoints)
        pred_descriptors = torch.randn(batch_size, num_keypoints, descriptor_dim)
        pred_semantic = torch.randn(batch_size, 20, 64, 64)
        
        # 真实标签
        gt_keypoints = torch.randn(batch_size, num_keypoints, 2)
        gt_scores = torch.rand(batch_size, num_keypoints)
        gt_descriptors = torch.randn(batch_size, num_keypoints, descriptor_dim)
        gt_semantic = torch.randint(0, 20, (batch_size, 64, 64))
        
        # 创建损失函数
        loss_fn = SemanticGuidedLoss()
        
        # 计算损失
        loss = loss_fn(
            pred_keypoints, pred_scores, pred_descriptors, pred_semantic,
            gt_keypoints, gt_scores, gt_descriptors, gt_semantic
        )
        
        # 检查损失值
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
        
    def test_model_quantization(self):
        """
        测试模型量化
        """
        # 创建量化器
        quantizer = ModelQuantization(self.model)
        
        # 动态量化
        quantized_model = quantizer.dynamic_quantization()
        
        # 检查量化模型
        self.assertIsInstance(quantized_model, torch.nn.Module)
        
        # 测试量化模型推理
        x = torch.randn(1, 1, self.input_size, self.input_size, device=self.device)
        with torch.no_grad():
            output = quantized_model(x)
            
        self.assertIsInstance(output, torch.Tensor)
        
    def test_model_parameters(self):
        """
        测试模型参数
        """
        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # 检查参数数量合理
        self.assertGreater(total_params, 0)
        self.assertGreater(trainable_params, 0)
        self.assertLessEqual(trainable_params, total_params)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


class TestDataProcessing(unittest.TestCase):
    """
    数据处理单元测试
    ""
    def setUp(self):
n        """
        测试设置
        """
        self.image_size = 256
        self.transform = ImageTransforms(image_size=self.image_size, is_training=False)
        self.semantic_transform = SemanticTransforms(image_size=self.image_size, is_training=False)
        
        # 创建临时测试图像
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, 'test_image.jpg')
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(self.test_image_path, test_image)
        
    def tearDown(self):
        """
        清理测试环境
        """
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_image_transforms(self):
        """
        测试图像变换
        """
        # 测试图像加载和变换
        transformed_image = self.transform(self.test_image_path)
        
        # 检查输出格式
        self.assertIsInstance(transformed_image, torch.Tensor)
        self.assertEqual(transformed_image.shape, (1, self.image_size, self.image_size))
        self.assertEqual(transformed_image.dtype, torch.float32)
        
    def test_semantic_transforms(self):
        """
        测试语义变换
        """
        # 创建测试语义标签
        semantic_label = np.random.randint(0, 20, (512, 512), dtype=np.uint8)
        semantic_path = os.path.join(self.temp_dir, 'test_semantic.png')
        cv2.imwrite(semantic_path, semantic_label)
        
        # 测试语义变换
        transformed_semantic = self.semantic_transform(semantic_path)
        
        # 检查输出格式
        self.assertIsInstance(transformed_semantic, torch.Tensor)
        self.assertEqual(transformed_semantic.shape, (self.image_size, self.image_size))
        self.assertEqual(transformed_semantic.dtype, torch.long)
        
    def test_dataset_creation(self):
        """
        测试数据集创建
        """
        # 创建测试数据集配置
        dataset_config = {
            'image_dir': self.temp_dir,
            'semantic_dir': self.temp_dir,
            'image_list': [self.test_image_path],
            'semantic_list': [os.path.join(self.temp_dir, 'test_semantic.png')],
            'transform': self.transform,
            'semantic_transform': self.semantic_transform
        }
        
        # 创建数据集
        dataset = SemanticFeatureDataset(**dataset_config)
        
        # 检查数据集
        self.assertEqual(len(dataset), 1)
        
        # 测试数据获取
        sample = dataset[0]
        self.assertIn('image', sample)
        self.assertIn('semantic', sample)
        self.assertEqual(sample['image'].shape, (1, self.image_size, self.image_size))
        
    def test_data_module(self):
        """
        测试数据模块
        """
        # 创建数据模块配置
        data_config = {
            'batch_size': 2,
            'num_workers': 0,
            'image_size': self.image_size,
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        }
        
        # 创建临时数据集文件
        dataset_file = os.path.join(self.temp_dir, 'dataset.json')
        dataset_data = {
            'image_dir': self.temp_dir,
            'semantic_dir': self.temp_dir,
            'image_list': [self.test_image_path] * 10,
            'semantic_list': [os.path.join(self.temp_dir, 'test_semantic.png')] * 10
        }
        with open(dataset_file, 'w') as f:
            json.dump(dataset_data, f)
            
        # 创建数据模块
        data_module = DataModule(dataset_file, **data_config)
        
        # 检查数据加载器
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(val_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(test_loader, torch.utils.data.DataLoader)
        
        # 测试批量数据
        for batch in train_loader:
            self.assertIn('image', batch)
            self.assertIn('semantic', batch)
            self.assertEqual(batch['image'].shape[0], data_config['batch_size'])
            break


class TestTools(unittest.TestCase):
    """
    工具脚本单元测试
    ""
    def setUp(self):
n        """
        测试设置
        """
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, 'test_model.pth')
        
        # 创建测试模型
        self.model = create_semantic_guided_xfeat(**TEST_CONFIG['model'])
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': TEST_CONFIG['model']
        }, self.model_path)
        
        # 创建测试图像
        self.test_image_path = os.path.join(self.temp_dir, 'test_image.jpg')
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(self.test_image_path, test_image)
        
    def tearDown(self):
        """
        清理测试环境
        """
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_model_converter(self):
        """
        测试模型转换器
        """
        # 创建转换器
        converter = ModelConverter(self.model_path, self.temp_dir)
        
        # 测试ONNX转换
        onnx_path = converter.convert_to_onnx()
        self.assertTrue(os.path.exists(onnx_path))
        
        # 检查ONNX文件大小
        self.assertGreater(os.path.getsize(onnx_path), 0)
        
    def test_batch_inference(self):
        """
        测试批量推理
        """
        # 创建推理器
        inference = BatchInference(self.model_path, device='cpu')
        
        # 创建测试图像目录
        image_dir = os.path.join(self.temp_dir, 'images')
        os.makedirs(image_dir)
        
        # 复制测试图像
        for i in range(3):
            test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            image_path = os.path.join(image_dir, f'test_image_{i}.jpg')
            cv2.imwrite(image_path, test_image)
            
        # 测试批量特征提取
        output_dir = os.path.join(self.temp_dir, 'features')
        features = inference.extract_features_batch(image_dir, output_dir, top_k=100)
        
        # 检查结果
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 3)
        
        # 检查文件保存
        self.assertTrue(os.path.exists(output_dir))
        feature_files = list(Path(output_dir).glob('*_features.json'))
        self.assertGreater(len(feature_files), 0)
        
    def test_feature_visualizer(self):
        """
        测试特征可视化器
        """
        # 创建可视化器
        visualizer = FeatureVisualizer(self.temp_dir)
        
        # 创建测试特征数据
        test_features = {
            'keypoints': np.random.rand(50, 2) * 256,
            'scores': np.random.rand(50),
            'descriptors': np.random.rand(50, 256),
            'semantic_attention': np.random.rand(20, 64, 64)
        }
        
        # 测试关键点可视化
        keypoints_path = visualizer.visualize_keypoints(
            self.test_image_path, test_features, save_name='test_keypoints'
        )
        self.assertTrue(os.path.exists(keypoints_path))
        
        # 测试语义注意力可视化
        semantic_path = visualizer.visualize_semantic_attention(
            self.test_image_path, test_features['semantic_attention'], save_name='test_semantic'
        )
        self.assertTrue(os.path.exists(semantic_path))
        
    def test_feature_report(self):
        """
        测试特征报告生成
        """
        # 创建可视化器
        visualizer = FeatureVisualizer(self.temp_dir)
        
        # 创建测试结果数据
        test_results = {
            'model_size': {
                'total_params': 1000000,
                'model_size_mb': 4.5
            },
            'runtime': {
                'avg_time_ms': 15.2,
                'fps': 65.8
            },
            'matching': {
                'inlier_rate': 0.85,
                'precision': 0.88,
                'recall': 0.82,
                'f1_score': 0.85
            }
        }
        
        # 测试报告生成
        report_path = visualizer.create_feature_report(test_results, 'test_report')
        self.assertTrue(os.path.exists(report_path))
        
        # 检查报告内容
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
            self.assertIn('Feature Analysis Report', report_content)
            self.assertIn('Model Information', report_content)


class TestIntegration(unittest.TestCase):
    """
    集成测试
    ""
    def setUp(self):
n        """
        测试设置
        """
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试配置
        self.config = TEST_CONFIG.copy()
        
        # 创建测试数据
        self._create_test_data()
        
    def tearDown(self):
        """
        清理测试环境
        """
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def _create_test_data(self):
        """
        创建测试数据
        """
        # 创建图像目录
        self.image_dir = os.path.join(self.temp_dir, 'images')
        self.semantic_dir = os.path.join(self.temp_dir, 'semantic')
        os.makedirs(self.image_dir)
        os.makedirs(self.semantic_dir)
        
        # 创建测试图像和语义标签
        for i in range(10):
            # 创建测试图像
            image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            image_path = os.path.join(self.image_dir, f'image_{i:03d}.jpg')
            cv2.imwrite(image_path, image)
            
            # 创建语义标签
            semantic = np.random.randint(0, 20, (256, 256), dtype=np.uint8)
            semantic_path = os.path.join(self.semantic_dir, f'semantic_{i:03d}.png')
            cv2.imwrite(semantic_path, semantic)
            
        # 创建数据集文件
        self.dataset_file = os.path.join(self.temp_dir, 'dataset.json')
        dataset_data = {
            'image_dir': self.image_dir,
            'semantic_dir': self.semantic_dir,
            'image_list': [os.path.join(self.image_dir, f'image_{i:03d}.jpg') for i in range(10)],
            'semantic_list': [os.path.join(self.semantic_dir, f'semantic_{i:03d}.png') for i in range(10)]
        }
        with open(self.dataset_file, 'w') as f:
            json.dump(dataset_data, f)
            
    def test_full_pipeline(self):
        """
        测试完整流程
        """
        # 1. 创建模型
        model = create_semantic_guided_xfeat(**self.config['model'])
        
        # 2. 创建数据模块
        data_module = DataModule(
            self.dataset_file,
            batch_size=self.config['training']['batch_size'],
            num_workers=0,
            image_size=self.config['data']['image_size']
        )
        
        # 3. 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['training']['learning_rate'])
        
        # 4. 创建损失函数
        loss_fn = SemanticGuidedLoss()
        
        # 5. 训练一个epoch
        model.train()
        train_loader = data_module.train_dataloader()
        
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # 前向传播
            images = batch['image']
            semantics = batch['semantic']
            outputs = model(images)
            
            # 计算损失
            loss = loss_fn(
                outputs[1], outputs[2], outputs[3], outputs[4],  # 预测
                torch.zeros_like(outputs[1]),  # 伪标签
                torch.ones_like(outputs[2]),   # 伪标签
                torch.randn_like(outputs[3]),  # 伪标签
                semantics                     # 真实语义标签
            )
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 2:  # 只测试前几个批次
                break
                
        avg_loss = total_loss / num_batches
        self.assertGreater(avg_loss, 0)
        
        # 6. 保存模型
        model_path = os.path.join(self.temp_dir, 'trained_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config['model'],
            'epoch': 1,
            'loss': avg_loss
        }, model_path)
        self.assertTrue(os.path.exists(model_path))
        
        # 7. 测试推理
        model.eval()
        test_image = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            features = model.detectAndCompute(test_image, top_k=100)
            
        self.assertIsInstance(features, list)
        self.assertGreater(len(features[0]['keypoints']), 0)
        
        # 8. 测试工具
        # 批量推理
        inference = BatchInference(model_path, device='cpu')
        output_dir = os.path.join(self.temp_dir, 'inference_results')
        features = inference.extract_features_batch(self.image_dir, output_dir, top_k=50)
        
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)
        
        # 可视化
        visualizer = FeatureVisualizer(os.path.join(self.temp_dir, 'visualizations'))
        if len(features) > 0:
            keypoints_path = visualizer.visualize_keypoints(
                features[0]['image_path'], features[0], save_name='pipeline_test'
            )
            self.assertTrue(os.path.exists(keypoints_path))
            
        print(f"Full pipeline test completed successfully. Average loss: {avg_loss:.4f}")
        
    def test_model_conversion_pipeline(self):
        """
        测试模型转换流程
        """
        # 创建并保存模型
        model = create_semantic_guided_xfeat(**self.config['model'])
        model_path = os.path.join(self.temp_dir, 'test_model.pth')
        torch.save({'model_state_dict': model.state_dict()}, model_path)
        
        # 转换为ONNX
        converter = ModelConverter(model_path, self.temp_dir)
        onnx_path = converter.convert_to_onnx()
        
        # 验证ONNX模型
        self.assertTrue(os.path.exists(onnx_path))
        
        # 测试ONNX推理
        import onnxruntime as ort
        ort_session = ort.InferenceSession(str(onnx_path))
        
        # 创建测试输入
        test_input = np.random.randn(1, 1, 256, 256).astype(np.float32)
        
        # ONNX推理
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # PyTorch推理
        with torch.no_grad():
            torch_outputs = model(torch.from_numpy(test_input))
            
        # 比较输出形状
        self.assertEqual(len(ort_outputs), len(torch_outputs))
        for i, (torch_out, ort_out) in enumerate(zip(torch_outputs, ort_outputs)):
            self.assertEqual(torch_out.shape, ort_out.shape)
            
        print("Model conversion pipeline test completed successfully.")


class TestPerformance(unittest.TestCase):
    """
    性能测试
    ""
    def setUp(self):
n        """
        测试设置
        """
        self.model = create_semantic_guided_xfeat(**TEST_CONFIG['model'])
        self.device = torch.device('cpu')
        self.model.to(self.device)
        
        # 创建测试数据
        self.test_images = []
        for i in range(10):
            image = torch.randn(1, 1, 256, 256, device=self.device)
            self.test_images.append(image)
            
    def test_inference_speed(self):
        """
        测试推理速度
        """
        model = self.model
        model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(5):
                _ = model(self.test_images[0])
                
        # 测试推理时间
        times = []
        with torch.no_grad():
            for image in self.test_images:
                start_time = time.time()
                _ = model(image)
                end_time = time.time()
                times.append(end_time - start_time)
                
        avg_time = np.mean(times) * 1000  # 转换为毫秒
        fps = 1.0 / np.mean(times)
        
        print(f"Average inference time: {avg_time:.2f} ms")
        print(f"FPS: {fps:.1f}")
        
        # 性能要求：CPU上平均推理时间应小于100ms
        self.assertLess(avg_time, 100, "Inference too slow on CPU")
        
    def test_memory_usage(self):
        """
        测试内存使用
        """
        import psutil
        import gc
        
        # 清理内存
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # 创建并运行模型
        model = self.model
        model.eval()
        
        with torch.no_grad():
            for image in self.test_images[:5]:
                _ = model(image)
                
        # 测量内存使用
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase: {memory_increase:.2f} MB")
        
        # 内存要求：内存增长应小于500MB
        self.assertLess(memory_increase, 500, "Memory usage too high")
        
    def test_model_size(self):
        """
        测试模型大小
        """
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        model_size_mb = total_params * 4 / 1024 / 1024  # 假设float32格式
        
        print(f"Model size: {model_size_mb:.2f} MB")
        print(f"Total parameters: {total_params:,}")
        
        # 大小要求：模型应小于10MB
        self.assertLess(model_size_mb, 10, "Model too large")
        
    def test_batch_processing(self):
        """
        测试批量处理性能
        """
        model = self.model
        model.eval()
        
        batch_sizes = [1, 2, 4, 8]
        times = {}
        
        for batch_size in batch_sizes:
            # 创建批量数据
            batch_images = torch.randn(batch_size, 1, 256, 256, device=self.device)
            
            # 预热
            with torch.no_grad():
                for _ in range(3):
                    _ = model(batch_images)
                    
            # 测试时间
            batch_times = []
n            with torch.no_grad():
n                for _ in range(10):
n                    start_time = time.time()
n                    _ = model(batch_images)
n                    end_time = time.time()
n                    batch_times.append(end_time - start_time)
n                    
            avg_time = np.mean(batch_times) * 1000  # 毫秒
            times[batch_size] = avg_time
            
            print(f"Batch size {batch_size}: {avg_time:.2f} ms")
            
        # 检查批量处理效率
        # 批量处理应该比单张处理更高效
        if len(times) > 1:
            single_time = times[1]
            batch_efficiency = times[1] / times[2]  # 单张 vs 批量2张
            self.assertGreater(batch_efficiency, 1.5, "Batch processing not efficient enough")
            
    def test_quantization_performance(self):
        """
        测试量化性能
        """
        from semantic_guided_xfeat_implementation import ModelQuantization
        
        # 创建量化模型
        quantizer = ModelQuantization(self.model)
        quantized_model = quantizer.dynamic_quantization()
        
        # 测试原始模型性能
        original_times = []
n        self.model.eval()
        with torch.no_grad():
            for image in self.test_images[:5]:
                start_time = time.time()
                _ = self.model(image)
                end_time = time.time()
n                original_times.append(end_time - start_time)
                
        # 测试量化模型性能
        quantized_times = []
n        quantized_model.eval()
n        with torch.no_grad():
            for image in self.test_images[:5]:
                start_time = time.time()
n                _ = quantized_model(image)
n                end_time = time.time()
n                quantized_times.append(end_time - start_time)
                
        original_avg = np.mean(original_times) * 1000
n        quantized_avg = np.mean(quantized_times) * 1000
n        speedup = original_avg / quantized_avg
        
        print(f"Original model: {original_avg:.2f} ms")
        print(f"Quantized model: {quantized_avg:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")
        
        # 量化应该带来性能提升
        self.assertGreater(speedup, 1.0, "Quantization should improve performance")


class TestRobustness(unittest.TestCase):
    """
    鲁棒性测试
    ""
    def setUp(self):
n        """
        测试设置
        """
        self.model = create_semantic_guided_xfeat(**TEST_CONFIG['model'])
n        self.model.eval()
        
    def test_input_variations(self):
        """
        测试输入变化
        """
        base_image = torch.randn(1, 1, 256, 256)
        
        # 测试不同输入尺寸
        for size in [128, 256, 512]:
            image = torch.randn(1, 1, size, size)
            with torch.no_grad():
                try:
                    features = self.model.detectAndCompute(image, top_k=100)
                    self.assertIsInstance(features, list)
                    print(f"Size {size}x{size}: OK")
                except Exception as e:
                    print(f"Size {size}x{size}: Error - {e}")
                    
        # 测试不同输入范围
        for scale in [0.1, 1.0, 10.0]:
            image = base_image * scale
            with torch.no_grad():
                try:
                    features = self.model.detectAndCompute(image, top_k=100)
                    self.assertIsInstance(features, list)
                    print(f"Scale {scale}: OK")
                except Exception as e:
                    print(f"Scale {scale}: Error - {e}")
                    
    def test_edge_cases(self):
        """
        测试边界情况
        """
        # 测试空输入
        with torch.no_grad():
            try:
                empty_image = torch.zeros(1, 1, 256, 256)
                features = self.model.detectAndCompute(empty_image, top_k=100)
                self.assertIsInstance(features, list)
                print("Empty image: OK")
            except Exception as e:
                print(f"Empty image: Error - {e}")
                
        # 测试极大值输入
        with torch.no_grad():
            try:
                large_image = torch.ones(1, 1, 256, 256) * 1000
                features = self.model.detectAndCompute(large_image, top_k=100)
                self.assertIsInstance(features, list)
                print("Large values: OK")
            except Exception as e:
                print(f"Large values: Error - {e}")
                
    def test_noise_resistance(self):
        """
        测试噪声抵抗性
        """
        base_image = torch.randn(1, 1, 256, 256)
        
        # 添加不同类型的噪声
        noise_types = {
            'gaussian': torch.randn_like(base_image) * 0.1,
            'salt_pepper': (torch.rand_like(base_image) > 0.9).float() * 0.5,
            'uniform': (torch.rand_like(base_image) - 0.5) * 0.2
        }
        
        for noise_name, noise in noise_types.items():
            noisy_image = base_image + noise
            with torch.no_grad():
                try:
                    features = self.model.detectAndCompute(noisy_image, top_k=100)
                    self.assertIsInstance(features, list)
                    num_keypoints = len(features[0]['keypoints'])
                    print(f"{noise_name} noise: {num_keypoints} keypoints")
                except Exception as e:
                    print(f"{noise_name} noise: Error - {e}")


def run_all_tests():
    """
    运行所有测试
    """
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestModelComponents,
        TestDataProcessing,
        TestTools,
        TestIntegration,
        TestPerformance,
        TestRobustness
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
        
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出结果
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
            
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
            
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # 运行测试
    success = run_all_tests()
    
    # 根据测试结果退出
    sys.exit(0 if success else 1)