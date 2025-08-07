"""
批量处理示例
演示如何使用语义引导的轻量级特征检测器进行批量图像处理
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入项目模块
from semantic_guided_xfeat_implementation import create_semantic_guided_xfeat
from tools import BatchInference
from basic_feature_detection import load_image, preprocess_image, detect_features
from feature_matching import match_features_semantic, filter_matches_by_homography


class BatchFeatureProcessor:
    """
    批量特征处理器
    """
    
    def __init__(self, model, config=None):
        """
        初始化批量特征处理器
        
        Args:
            model: 语义引导的特征检测器模型
            config: 配置字典
        """
        self.model = model
        self.config = config or {
            'target_size': 256,
            'top_k': 1000,
            'threshold': 0.01,
            'semantic_weight': 0.3,
            'ratio_threshold': 0.8,
            'ransac_threshold': 3.0,
            'batch_size': 8,
            'num_workers': 4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        self.device = torch.device(self.config['device'])
        self.model.to(self.device)
        self.model.eval()
        
        # 创建批量推理器
        self.batch_inference = BatchInference(
            model, 
            batch_size=self.config['batch_size'],
            device=self.device
        )
        
        # 结果存储
        self.results = {}
        self.statistics = {}
        
    def process_single_image(self, image_path, save_features=True):
        """
        处理单张图像
        
        Args:
            image_path: 图像文件路径
            save_features: 是否保存特征数据
            
        Returns:
            特征字典
        """
        try:
            # 加载图像
            image = load_image(image_path)
            
            # 预处理
            image_tensor, _ = preprocess_image(image, self.config['target_size'])
            image_tensor = image_tensor.to(self.device)
            
            # 特征检测
            features = detect_features(
                self.model,
                image_tensor,
                top_k=self.config['top_k'],
                threshold=self.config['threshold']
            )
            
            # 添加元数据
            features['image_path'] = str(image_path)
            features['image_size'] = list(image.shape)
            features['processing_time'] = time.time()
            
            # 保存特征数据
            if save_features:
                self._save_features(features, image_path)
                
            return features
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
            
    def process_image_batch(self, image_paths, use_multiprocessing=True):
        """
        批量处理图像
        
        Args:
            image_paths: 图像路径列表
            use_multiprocessing: 是否使用多进程处理
            
        Returns:
            特征结果字典
        """
        results = {}
        
        if use_multiprocessing and len(image_paths) > 1:
            # 使用多进程处理
            with ThreadPoolExecutor(max_workers=self.config['num_workers']) as executor:
                # 提交任务
                future_to_path = {
                    executor.submit(self.process_single_image, path): path 
                    for path in image_paths
                }
                
                # 收集结果
                for future in tqdm(as_completed(future_to_path), total=len(image_paths), desc="Processing images"):
                    path = future_to_path[future]
                    try:
                        features = future.result()
                        if features is not None:
                            results[str(path)] = features
                    except Exception as e:
                        print(f"Error processing {path}: {e}")
                        
        else:
            # 单进程处理
            for path in tqdm(image_paths, desc="Processing images"):
                features = self.process_single_image(path)
                if features is not None:
                    results[str(path)] = features
                    
        self.results.update(results)
        return results
        
    def process_image_pairs(self, image_paths, use_multiprocessing=True):
        """
        处理图像对（用于匹配）
        
        Args:
            image_paths: 图像路径列表
            use_multiprocessing: 是否使用多进程处理
            
        Returns:
            匹配结果字典
        """
        # 生成图像对
        image_pairs = []
        for i in range(len(image_paths)):
            for j in range(i + 1, len(image_paths)):
                image_pairs.append((image_paths[i], image_paths[j]))
                
        print(f"Generated {len(image_pairs)} image pairs")
        
        matching_results = {}
        
        if use_multiprocessing and len(image_pairs) > 1:
            # 使用多进程处理
            with ThreadPoolExecutor(max_workers=self.config['num_workers']) as executor:
                # 提交任务
                future_to_pair = {
                    executor.submit(self._process_image_pair, pair): pair 
                    for pair in image_pairs
                }
                
                # 收集结果
                for future in tqdm(as_completed(future_to_pair), total=len(image_pairs), desc="Processing pairs"):
                    pair = future_to_pair[future]
                    try:
                        match_result = future.result()
                        if match_result is not None:
                            pair_key = f"{Path(pair[0]).stem}_{Path(pair[1]).stem}"
                            matching_results[pair_key] = match_result
                    except Exception as e:
                        print(f"Error processing pair {pair}: {e}")
                        
        else:
            # 单进程处理
            for pair in tqdm(image_pairs, desc="Processing pairs"):
                match_result = self._process_image_pair(pair)
                if match_result is not None:
                    pair_key = f"{Path(pair[0]).stem}_{Path(pair[1]).stem}"
                    matching_results[pair_key] = match_result
                    
        return matching_results
        
    def _process_image_pair(self, image_pair):
        """
        处理单个图像对
        
        Args:
            image_pair: (image_path1, image_path2)
            
        Returns:
            匹配结果字典
        """
        try:
            path1, path2 = image_pair
            
            # 检查是否已有特征结果
            if str(path1) not in self.results:
                features1 = self.process_single_image(path1, save_features=False)
                self.results[str(path1)] = features1
            else:
                features1 = self.results[str(path1)]
                
            if str(path2) not in self.results:
                features2 = self.process_single_image(path2, save_features=False)
                self.results[str(path2)] = features2
            else:
                features2 = self.results[str(path2)]
                
            if features1 is None or features2 is None:
                return None
                
            # 特征匹配
            matches = match_features_semantic(
                features1['descriptors'], features2['descriptors'],
                features1['semantic_attention'], features2['semantic_attention'],
                semantic_weight=self.config['semantic_weight'],
                ratio_threshold=self.config['ratio_threshold'], cross_check=True
            )
            
            # 过滤匹配
            filtered_matches = filter_matches_by_homography(
                features1['keypoints'], features2['keypoints'],
                matches[0], matches[1],
                ransac_threshold=self.config['ransac_threshold']
            )
            
            # 计算匹配统计
            match_stats = {
                'total_matches': len(matches[0]),
                'filtered_matches': len(filtered_matches[0]),
                'inlier_ratio': len(filtered_matches[0]) / max(len(matches[0]), 1),
                'match_scores': matches[2].tolist() if len(matches[2]) > 0 else [],
                'image1_path': str(path1),
                'image2_path': str(path2)
            }
            
            return match_stats
            
        except Exception as e:
            print(f"Error processing pair {image_pair}: {e}")
            return None
            
    def _save_features(self, features, image_path):
        """
        保存特征数据
        
        Args:
            features: 特征字典
            image_path: 图像路径
        """
        # 创建输出目录
        output_dir = Path(self.config.get('output_dir', './examples/output'))
        features_dir = output_dir / 'features'
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成输出文件名
        image_name = Path(image_path).stem
        output_path = features_dir / f"{image_name}_features.json"
        
        # 准备保存的数据
        save_data = {
            'image_path': str(image_path),
            'image_size': features['image_size'],
            'num_keypoints': features['num_keypoints'],
            'keypoints': features['keypoints'].tolist(),
            'scores': features['scores'].tolist(),
            'processing_time': features['processing_time'],
            'config': self.config
        }
        
        # 保存为JSON文件
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
            
    def save_results(self, output_dir):
        """
        保存批量处理结果
        
        Args:
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存特征结果
        features_file = output_dir / 'batch_features.json'
        with open(features_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # 保存统计信息
        stats_file = output_dir / 'batch_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(self.statistics, f, indent=2)
            
        # 生成CSV报告
        self._generate_csv_report(output_dir)
        
        # 生成可视化报告
        self._generate_visualization_report(output_dir)
        
    def _generate_csv_report(self, output_dir):
        """
        生成CSV报告
        
        Args:
            output_dir: 输出目录
        """
        # 准备数据
        rows = []
        for image_path, features in self.results.items():
            row = {
                'image_path': image_path,
                'image_size': f"{features['image_size'][1]}x{features['image_size'][0]}",
                'num_keypoints': features['num_keypoints'],
                'avg_score': np.mean(features['scores']) if len(features['scores']) > 0 else 0,
                'max_score': np.max(features['scores']) if len(features['scores']) > 0 else 0,
                'min_score': np.min(features['scores']) if len(features['scores']) > 0 else 0,
                'processing_time': features['processing_time']
            }
            rows.append(row)
            
        # 创建DataFrame
        df = pd.DataFrame(rows)
        
        # 保存CSV
        csv_file = output_dir / 'batch_report.csv'
        df.to_csv(csv_file, index=False)
        
    def _generate_visualization_report(self, output_dir):
        """
        生成可视化报告
        
        Args:
            output_dir: 输出目录
        """
        # 创建可视化目录
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 提取统计数据
        num_keypoints = [f['num_keypoints'] for f in self.results.values()]
        avg_scores = [np.mean(f['scores']) if len(f['scores']) > 0 else 0 for f in self.results.values()]
        
        # 创建统计图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 关键点数量分布
        axes[0, 0].hist(num_keypoints, bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('Distribution of Keypoint Counts')
        axes[0, 0].set_xlabel('Number of Keypoints')
        axes[0, 0].set_ylabel('Frequency')
        
        # 平均分数分布
        axes[0, 1].hist(avg_scores, bins=20, alpha=0.7, color='green')
        axes[0, 1].set_title('Distribution of Average Scores')
        axes[0, 1].set_xlabel('Average Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # 关键点数量 vs 平均分数
        axes[1, 0].scatter(num_keypoints, avg_scores, alpha=0.6, color='red')
        axes[1, 0].set_title('Keypoint Count vs Average Score')
        axes[1, 0].set_xlabel('Number of Keypoints')
        axes[1, 0].set_ylabel('Average Score')
        
        # 处理时间统计
        processing_times = [f['processing_time'] for f in self.results.values()]
        if processing_times:
            time_diffs = [processing_times[i] - processing_times[i-1] for i in range(1, len(processing_times))]
            axes[1, 1].plot(time_diffs, color='purple')
            axes[1, 1].set_title('Processing Time per Image')
            axes[1, 1].set_xlabel('Image Index')
            axes[1, 1].set_ylabel('Processing Time (s)')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'batch_statistics.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 生成最佳和最差样本的可视化
        self._generate_sample_visualizations(viz_dir)
        
    def _generate_sample_visualizations(self, viz_dir):
        """
        生成样本可视化
        
        Args:
            viz_dir: 可视化目录
        """
        # 找到关键点最多和最少的样本
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['num_keypoints'])
        
        # 最佳样本（关键点最多）
        if len(sorted_results) > 0:
            best_path, best_features = sorted_results[-1]
            worst_path, worst_features = sorted_results[0]
            
            # 加载图像
            best_image = load_image(best_path)
            worst_image = load_image(worst_path)
            
            # 创建对比图
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # 最佳样本
            axes[0].imshow(cv2.cvtColor(best_image, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f'Best Sample: {best_features["num_keypoints"]} keypoints')
            axes[0].axis('off')
            
            # 最差样本
            axes[1].imshow(cv2.cvtColor(worst_image, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f'Worst Sample: {worst_features["num_keypoints"]} keypoints')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'best_worst_samples.png', dpi=150, bbox_inches='tight')
            plt.close()
            
    def compute_statistics(self):
        """
        计算统计信息
        
        Returns:
            统计信息字典
        """
        if not self.results:
            return {}
            
        stats = {
            'total_images': len(self.results),
            'total_keypoints': sum(f['num_keypoints'] for f in self.results.values()),
            'avg_keypoints_per_image': np.mean([f['num_keypoints'] for f in self.results.values()]),
            'std_keypoints_per_image': np.std([f['num_keypoints'] for f in self.results.values()]),
            'max_keypoints': max(f['num_keypoints'] for f in self.results.values()),
            'min_keypoints': min(f['num_keypoints'] for f in self.results.values()),
            'avg_score': np.mean([np.mean(f['scores']) for f in self.results.values() if len(f['scores']) > 0]),
            'processing_times': [f['processing_time'] for f in self.results.values()]
        }
        
        # 计算处理时间统计
        if len(stats['processing_times']) > 1:
            time_diffs = [stats['processing_times'][i] - stats['processing_times'][i-1] for i in range(1, len(stats['processing_times']))]
            stats.update({
                'avg_processing_time': np.mean(time_diffs),
                'std_processing_time': np.std(time_diffs),
                'total_processing_time': stats['processing_times'][-1] - stats['processing_times'][0]
            })
            
        self.statistics = stats
        return stats
        
    def print_statistics(self):
        """
        打印统计信息
        """
        stats = self.compute_statistics()
        
        print("\n" + "="*60)
        print("批量处理统计信息")
        print("="*60)
        print(f"总图像数量: {stats['total_images']}")
        print(f"总关键点数量: {stats['total_keypoints']:,}")
        print(f"平均每张图像关键点数量: {stats['avg_keypoints_per_image']:.2f} ± {stats['std_keypoints_per_image']:.2f}")
        print(f"关键点数量范围: {stats['min_keypoints']} - {stats['max_keypoints']}")
        print(f"平均分数: {stats['avg_score']:.4f}")
        
        if 'avg_processing_time' in stats:
            print(f"平均处理时间: {stats['avg_processing_time']:.4f} 秒/图像")
            print(f"总处理时间: {stats['total_processing_time']:.2f} 秒")
            print(f"处理速度: {stats['total_images'] / stats['total_processing_time']:.2f} 图像/秒")
            
        print("="*60 + "\n")


def create_test_images(output_dir, num_images=10):
    """
    创建测试图像集
    
    Args:
        output_dir: 输出目录
        num_images: 图像数量
        
    Returns:
        图像路径列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    
    for i in range(num_images):
        # 创建随机图像
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # 添加随机几何形状
        num_shapes = np.random.randint(3, 8)
        for _ in range(num_shapes):
            shape_type = np.random.choice(['circle', 'rectangle', 'ellipse'])
            color = np.random.randint(0, 255, 3).tolist()
            
            if shape_type == 'circle':
                center = (np.random.randint(50, 462), np.random.randint(50, 462))
                radius = np.random.randint(20, 80)
                cv2.circle(image, center, radius, color, -1)
            elif shape_type == 'rectangle':
                x1, y1 = np.random.randint(0, 400, 2)
                x2, y2 = x1 + np.random.randint(50, 150), y1 + np.random.randint(50, 150)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
            else:  # ellipse
                center = (np.random.randint(50, 462), np.random.randint(50, 462))
                axes = (np.random.randint(20, 80), np.random.randint(20, 80))
                angle = np.random.randint(0, 360)
                cv2.ellipse(image, center, axes, angle, 0, 360, color, -1)
                
        # 添加噪声
        noise = np.random.normal(0, 10, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 保存图像
        image_path = output_dir / f'test_image_{i:03d}.jpg'
        cv2.imwrite(str(image_path), image)
        image_paths.append(str(image_path))
        
    return image_paths


def main():
    """
    主函数
    """
    # 配置参数
    config = {
        'model_path': None,
        'input_dir': './examples/test_images',
        'output_dir': './examples/output',
        'target_size': 256,
        'top_k': 1000,
        'threshold': 0.01,
        'semantic_weight': 0.3,
        'ratio_threshold': 0.8,
        'ransac_threshold': 3.0,
        'batch_size': 4,
        'num_workers': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_test_images': 10,
        'process_pairs': True,
        'use_multiprocessing': True
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print("语义引导的轻量级特征检测器 - 批量处理示例")
    print("="*60)
    
    # 1. 创建模型
    print("1. 创建模型...")
    model = create_semantic_guided_xfeat()
    print(f"   模型创建成功")
    
    # 2. 创建批量处理器
    print("\n2. 创建批量处理器...")
    processor = BatchFeatureProcessor(model, config)
    print(f"   批量处理器创建成功")
    print(f"   设备: {processor.device}")
    print(f"   批量大小: {config['batch_size']}")
    print(f"   工作进程数: {config['num_workers']}")
    
    # 3. 创建或加载测试图像
    print("\n3. 准备测试图像...")
    input_dir = Path(config['input_dir'])
    
    if not input_dir.exists():
        print("   创建测试图像集...")
        image_paths = create_test_images(input_dir, config['num_test_images'])
    else:
        # 加载现有图像
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(input_dir.glob(f'*{ext}')))
            image_paths.extend(list(input_dir.glob(f'*{ext.upper()}')))
        image_paths = [str(p) for p in image_paths]
        
    print(f"   找到 {len(image_paths)} 张图像")
    
    if len(image_paths) == 0:
        print("   没有找到图像，创建测试图像集...")
        image_paths = create_test_images(input_dir, config['num_test_images'])
        
    # 4. 批量处理图像
    print("\n4. 批量处理图像...")
    start_time = time.time()
    
    results = processor.process_image_batch(
        image_paths, 
        use_multiprocessing=config['use_multiprocessing']
    )
    
    processing_time = time.time() - start_time
    print(f"   批量处理完成，耗时: {processing_time:.2f} 秒")
    print(f"   处理速度: {len(results) / processing_time:.2f} 图像/秒")
    
    # 5. 处理图像对（可选）
    matching_results = None
    if config['process_pairs'] and len(image_paths) > 1:
        print("\n5. 处理图像对...")
        start_time = time.time()
        
        matching_results = processor.process_image_pairs(
            image_paths[:5],  # 限制处理数量以避免过多计算
            use_multiprocessing=config['use_multiprocessing']
        )
        
        matching_time = time.time() - start_time
        print(f"   图像对处理完成，耗时: {matching_time:.2f} 秒")
        print(f"   处理了 {len(matching_results)} 个图像对")
        
    # 6. 计算统计信息
    print("\n6. 计算统计信息...")
    stats = processor.compute_statistics()
    processor.print_statistics()
    
    # 7. 保存结果
    print("\n7. 保存结果...")
    processor.save_results(config['output_dir'])
    print(f"   结果保存到: {config['output_dir']}")
    
    # 8. 显示摘要
    print("\n8. 处理摘要:")
    print(f"   - 成功处理 {len(results)} 张图像")
    print(f"   - 总共检测到 {stats['total_keypoints']:,} 个关键点")
    print(f"   - 平均每张图像 {stats['avg_keypoints_per_image']:.2f} 个关键点")
    
    if matching_results:
        avg_inlier_ratio = np.mean([r['inlier_ratio'] for r in matching_results.values()])
        print(f"   - 平均内点率: {avg_inlier_ratio:.4f}")
        
    print("\n" + "="*60)
    print("批量处理示例完成！")
    print(f"结果保存在: {config['output_dir']}")
    print("="*60)
    
    return {
        'results': results,
        'matching_results': matching_results,
        'statistics': stats
    }


if __name__ == "__main__":
    # 运行主函数
    results = main()
    
    # 可选：返回结果供其他示例使用
    if results:
        print("\n批量处理成功完成！")
    else:
        print("\n批量处理失败！")