"""
可视化示例
演示如何使用语义引导的轻量级特征检测器进行各种可视化分析
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import json
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入项目模块
from semantic_guided_xfeat_implementation import create_semantic_guided_xfeat
from tools import FeatureVisualizer
from basic_feature_detection import load_image, preprocess_image, detect_features
from feature_matching import match_features_semantic, filter_matches_by_homography


class FeatureVisualizerAdvanced:
    """
    高级特征可视化器
    """
    
    def __init__(self, model, config=None):
        """
        初始化可视化器
        
        Args:
            model: 语义引导的特征检测器模型
            config: 配置字典
        """
        self.model = model
        self.config = config or {
            'target_size': 256,
            'top_k': 1000,
            'threshold': 0.01,
            'output_dir': './examples/output',
            'dpi': 150,
            'figsize': (12, 8)
        }
        
        # 创建输出目录
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建可视化子目录
        self.viz_dir = self.output_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def visualize_keypoint_distribution(self, image, features, output_path=None):
        """
        可视化关键点分布
        
        Args:
            image: 原始图像
            features: 特征字典
            output_path: 输出路径
            
        Returns:
            可视化图像
        """
        fig, axes = plt.subplots(2, 2, figsize=self.config['figsize'])
        
        # 原始图像
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 关键点热力图
        h, w = image.shape[:2]
        heatmap = np.zeros((h, w))
        
        keypoints = features['keypoints']
        scores = features['scores']
        
        for kp, score in zip(keypoints, scores):
            x, y = int(kp[0]), int(kp[1])
            if 0 <= x < w and 0 <= y < h:
                heatmap[y, x] += score
                
        # 高斯模糊平滑热力图
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        im = axes[0, 1].imshow(heatmap, cmap='hot', interpolation='bilinear')
        axes[0, 1].set_title('Keypoint Heatmap')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1])
        
        # 关键点分数分布
        axes[1, 0].hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Keypoint Score Distribution')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 关键点空间分布
        x_coords = keypoints[:, 0]
        y_coords = keypoints[:, 1]
        
        scatter = axes[1, 1].scatter(x_coords, y_coords, c=scores, cmap='viridis', s=50, alpha=0.6)
        axes[1, 1].set_title('Keypoint Spatial Distribution')
        axes[1, 1].set_xlabel('X Coordinate')
        axes[1, 1].set_ylabel('Y Coordinate')
        axes[1, 1].invert_yaxis()  # 图像坐标系
        plt.colorbar(scatter, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            print(f"Keypoint distribution visualization saved to: {output_path}")
            
        return fig
        
    def visualize_semantic_attention(self, image, features, output_path=None):
        """
        可视化语义注意力
        
        Args:
            image: 原始图像
            features: 特征字典
            output_path: 输出路径
            
        Returns:
            可视化图像
        """
        semantic_attention = features['semantic_attention']
        num_classes = semantic_attention.shape[0]
        
        # 创建网格布局
        grid_size = int(np.ceil(np.sqrt(num_classes + 1)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()
        
        # 原始图像
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 每个语义类别的注意力图
        for i in range(num_classes):
            attention_map = semantic_attention[i]
            
            # 调整尺寸以匹配图像
            if attention_map.shape != image.shape[:2]:
                attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
                
            # 显示注意力图
            im = axes[i + 1].imshow(attention_map, cmap='hot', interpolation='bilinear')
            axes[i + 1].set_title(f'Semantic Class {i}')
            axes[i + 1].axis('off')
            plt.colorbar(im, ax=axes[i + 1])
            
        # 隐藏多余的子图
        for i in range(num_classes + 1, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            print(f"Semantic attention visualization saved to: {output_path}")
            
        return fig
        
    def visualize_descriptor_analysis(self, features, output_path=None):
        """
        可视化描述符分析
        
        Args:
            features: 特征字典
            output_path: 输出路径
            
        Returns:
            可视化图像
        """
        descriptors = features['descriptors']
        scores = features['scores']
        
        if len(descriptors) == 0:
            print("No descriptors found")
            return None
            
        fig, axes = plt.subplots(2, 3, figsize=self.config['figsize'])
        
        # 描述符统计
        desc_mean = np.mean(descriptors, axis=1)
        desc_std = np.std(descriptors, axis=1)
        desc_max = np.max(descriptors, axis=1)
        desc_min = np.min(descriptors, axis=1)
        
        # 描述符均值分布
        axes[0, 0].hist(desc_mean, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 0].set_title('Descriptor Mean Distribution')
        axes[0, 0].set_xlabel('Mean Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 描述符标准差分布
        axes[0, 1].hist(desc_std, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Descriptor Std Distribution')
        axes[0, 1].set_xlabel('Std Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 描述符范围分布
        desc_range = desc_max - desc_min
        axes[0, 2].hist(desc_range, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 2].set_title('Descriptor Range Distribution')
        axes[0, 2].set_xlabel('Range Value')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 描述符维度分析
        desc_dim = descriptors.shape[1]
        dim_means = np.mean(descriptors, axis=0)
        dim_stds = np.std(descriptors, axis=0)
        
        axes[1, 0].plot(dim_means, color='blue', alpha=0.7)
        axes[1, 0].set_title('Descriptor Dimension Means')
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Mean Value')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(dim_stds, color='red', alpha=0.7)
        axes[1, 1].set_title('Descriptor Dimension Stds')
        axes[1, 1].set_xlabel('Dimension')
        axes[1, 1].set_ylabel('Std Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 分数 vs 描述符统计
        axes[1, 2].scatter(scores, desc_mean, alpha=0.6, color='purple')
        axes[1, 2].set_title('Score vs Descriptor Mean')
        axes[1, 2].set_xlabel('Keypoint Score')
        axes[1, 2].set_ylabel('Descriptor Mean')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            print(f"Descriptor analysis visualization saved to: {output_path}")
            
        return fig
        
    def visualize_descriptor_embedding(self, features, method='tsne', output_path=None):
        """
        可视化描述符嵌入
        
        Args:
            features: 特征字典
            method: 降维方法 ('tsne' 或 'pca')
            output_path: 输出路径
            
        Returns:
            可视化图像
        """
        descriptors = features['descriptors']
        scores = features['scores']
        
        if len(descriptors) < 10:
            print("Not enough descriptors for embedding visualization")
            return None
            
        # 降维
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(descriptors)-1))
        else:
            reducer = PCA(n_components=2, random_state=42)
            
        embedding = reducer.fit_transform(descriptors)
        
        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 按分数着色
        scatter1 = axes[0].scatter(embedding[:, 0], embedding[:, 1], c=scores, cmap='viridis', s=50, alpha=0.7)
        axes[0].set_title(f'Descriptor Embedding ({method.upper()}) - Colored by Score')
        axes[0].set_xlabel('Component 1')
        axes[0].set_ylabel('Component 2')
        plt.colorbar(scatter1, ax=axes[0])
        
        # 按空间位置着色
        keypoints = features['keypoints']
        spatial_colors = keypoints[:, 0] + keypoints[:, 1]  # 简单的空间编码
        
        scatter2 = axes[1].scatter(embedding[:, 0], embedding[:, 1], c=spatial_colors, cmap='plasma', s=50, alpha=0.7)
        axes[1].set_title(f'Descriptor Embedding ({method.upper()}) - Colored by Position')
        axes[1].set_xlabel('Component 1')
        axes[1].set_ylabel('Component 2')
        plt.colorbar(scatter2, ax=axes[1])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            print(f"Descriptor embedding visualization saved to: {output_path}")
            
        return fig
        
    def visualize_matching_analysis(self, features1, features2, matches, output_path=None):
        """
        可视化匹配分析
        
        Args:
            features1: 第一张图像的特征
            features2: 第二张图像的特征
            matches: 匹配结果
            output_path: 输出路径
            
        Returns:
            可视化图像
        """
        if len(matches[0]) == 0:
            print("No matches found")
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=self.config['figsize'])
        
        # 匹配分数分布
        match_scores = matches[2]
        axes[0, 0].hist(match_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Match Score Distribution')
        axes[0, 0].set_xlabel('Match Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 匹配距离分布
        match_distances = matches[3]
        axes[0, 1].hist(match_distances, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Match Distance Distribution')
        axes[0, 1].set_xlabel('Match Distance')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 关键点分数 vs 匹配分数
        kp1_scores = features1['scores'][matches[0]]
        axes[1, 0].scatter(kp1_scores, match_scores, alpha=0.6, color='orange')
        axes[1, 0].set_title('Keypoint Score vs Match Score')
        axes[1, 0].set_xlabel('Keypoint Score')
        axes[1, 0].set_ylabel('Match Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 匹配空间一致性
        kp1_positions = features1['keypoints'][matches[0]]
        kp2_positions = features2['keypoints'][matches[1]]
        
        # 计算位移向量
        displacements = kp2_positions - kp1_positions
        displacement_magnitudes = np.linalg.norm(displacements, axis=1)
        
        axes[1, 1].scatter(displacements[:, 0], displacements[:, 1], c=match_scores, cmap='viridis', s=50, alpha=0.7)
        axes[1, 1].set_title('Match Displacement Vectors')
        axes[1, 1].set_xlabel('X Displacement')
        axes[1, 1].set_ylabel('Y Displacement')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加颜色条
        scatter = axes[1, 1].scatter(displacements[:, 0], displacements[:, 1], c=match_scores, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(scatter, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            print(f"Matching analysis visualization saved to: {output_path}")
            
        return fig
        
    def create_comprehensive_report(self, image, features, matches_info=None, output_path=None):
        """
        创建综合可视化报告
        
        Args:
            image: 原始图像
            features: 特征字典
            matches_info: 匹配信息字典 (可选)
            output_path: 输出路径
            
        Returns:
            报告数据字典
        """
        report_data = {
            'image_info': {
                'size': image.shape,
                'num_keypoints': features['num_keypoints'],
                'avg_score': np.mean(features['scores']) if len(features['scores']) > 0 else 0
            },
            'visualizations': {}
        }
        
        # 1. 关键点分布可视化
        kp_dist_path = self.viz_dir / 'keypoint_distribution.png'
        fig1 = self.visualize_keypoint_distribution(image, features, str(kp_dist_path))
        if fig1:
            plt.close(fig1)
            report_data['visualizations']['keypoint_distribution'] = str(kp_dist_path)
            
        # 2. 语义注意力可视化
        semantic_path = self.viz_dir / 'semantic_attention.png'
        fig2 = self.visualize_semantic_attention(image, features, str(semantic_path))
        if fig2:
            plt.close(fig2)
            report_data['visualizations']['semantic_attention'] = str(semantic_path)
            
        # 3. 描述符分析可视化
        desc_path = self.viz_dir / 'descriptor_analysis.png'
        fig3 = self.visualize_descriptor_analysis(features, str(desc_path))
        if fig3:
            plt.close(fig3)
            report_data['visualizations']['descriptor_analysis'] = str(desc_path)
            
        # 4. 描述符嵌入可视化
        embed_path = self.viz_dir / 'descriptor_embedding_tsne.png'
        fig4 = self.visualize_descriptor_embedding(features, method='tsne', output_path=str(embed_path))
        if fig4:
            plt.close(fig4)
            report_data['visualizations']['descriptor_embedding'] = str(embed_path)
            
        # 5. 匹配分析可视化（如果有匹配信息）
        if matches_info:
            match_path = self.viz_dir / 'matching_analysis.png'
            fig5 = self.visualize_matching_analysis(
                matches_info['features1'], matches_info['features2'], 
                matches_info['matches'], str(match_path)
            )
            if fig5:
                plt.close(fig5)
                report_data['visualizations']['matching_analysis'] = str(match_path)
                
        # 保存报告数据
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
                
        return report_data
        
    def visualize_feature_quality(self, features, output_path=None):
        """
        可视化特征质量分析
        
        Args:
            features: 特征字典
            output_path: 输出路径
            
        Returns:
            可视化图像
        """
        descriptors = features['descriptors']
        scores = features['scores']
        
        if len(descriptors) == 0:
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=self.config['figsize'])
        
        # 描述符正交性
        desc_norm = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8)
        similarity_matrix = np.dot(desc_norm, desc_norm.T)
        
        # 只显示下三角部分（不包括对角线）
        mask = np.triu(np.ones_like(similarity_matrix), k=1)
        similarities = similarity_matrix[mask == 1]
        
        axes[0, 0].hist(similarities, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 0].set_title('Descriptor Orthogonality')
        axes[0, 0].set_xlabel('Cosine Similarity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 分数与描述符多样性
        desc_var = np.var(descriptors, axis=1)
        axes[0, 1].scatter(scores, desc_var, alpha=0.6, color='green')
        axes[0, 1].set_title('Score vs Descriptor Variance')
        axes[0, 1].set_xlabel('Keypoint Score')
        axes[0, 1].set_ylabel('Descriptor Variance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 描述符稀疏性
        sparsity = np.mean(np.abs(descriptors) < 0.1, axis=1)
        axes[1, 0].hist(sparsity, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Descriptor Sparsity')
        axes[1, 0].set_xlabel('Sparsity Ratio')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 特征质量综合评分
        quality_score = scores * desc_var * (1 - sparsity)
        axes[1, 1].hist(quality_score, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[1, 1].set_title('Feature Quality Score')
        axes[1, 1].set_xlabel('Quality Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            print(f"Feature quality visualization saved to: {output_path}")
            
        return fig


def create_test_image_with_semantics(output_path):
    """
    创建具有语义信息的测试图像
    
    Args:
        output_path: 输出路径
        
    Returns:
        图像数组
    """
    # 创建512x512的测试图像
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # 添加背景
    image[:] = (200, 200, 200)  # 灰色背景
    
    # 添加不同语义区域
    # 1. 圆形区域（类别0）
    cv2.circle(image, (150, 150), 80, (255, 0, 0), -1)  # 蓝色圆形
    
    # 2. 矩形区域（类别1）
    cv2.rectangle(image, (250, 100), (400, 200), (0, 255, 0), -1)  # 绿色矩形
    
    # 3. 椭圆区域（类别2）
    cv2.ellipse(image, (150, 350), (60, 40), 0, 0, 360, (0, 0, 255), -1)  # 红色椭圆
    
    # 4. 三角形区域（类别3）
    triangle_points = np.array([[350, 300], [450, 300], [400, 400]], np.int32)
    cv2.fillPoly(image, [triangle_points], (255, 255, 0))  # 黄色三角形
    
    # 5. 添加纹理
    for i in range(0, 512, 20):
        for j in range(0, 512, 20):
            if (i + j) % 40 == 0:
                cv2.circle(image, (i, j), 3, (100, 100, 100), -1)
                
    # 添加噪声
    noise = np.random.normal(0, 5, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 保存图像
    cv2.imwrite(output_path, image)
    
    return image


def main():
    """
    主函数
    """
    # 配置参数
    config = {
        'model_path': None,
        'image_path': './examples/test_image_semantic.jpg',
        'output_dir': './examples/output',
        'target_size': 256,
        'top_k': 1000,
        'threshold': 0.01,
        'create_second_image': True,
        'compute_matches': True
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print("语义引导的轻量级特征检测器 - 可视化示例")
    print("="*60)
    
    # 1. 创建模型
    print("1. 创建模型...")
    model = create_semantic_guided_xfeat()
    model.eval()
    print(f"   模型创建成功")
    
    # 2. 创建或加载测试图像
    print("\n2. 准备测试图像...")
    try:
        image1 = load_image(config['image_path'])
        print(f"   图像加载成功: {config['image_path']}")
    except Exception as e:
        print(f"   图像加载失败: {e}")
        print("   创建测试图像...")
        
        image1 = create_test_image_with_semantics(config['image_path'])
        print(f"   测试图像创建成功: {config['image_path']}")
        
    # 创建第二张图像（用于匹配可视化）
    image2 = None
    if config['create_second_image']:
        image2_path = config['image_path'].replace('.jpg', '_transformed.jpg')
        
        # 应用变换
        M = np.float32([[0.9, 0.1, 30], [-0.1, 0.9, 20]])
        image2 = cv2.warpAffine(image1, M, (512, 512))
        
        # 添加不同的噪声
        noise = np.random.normal(0, 8, image2.shape).astype(np.int16)
        image2 = np.clip(image2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        cv2.imwrite(image2_path, image2)
        print(f"   第二张图像创建成功: {image2_path}")
        
    # 3. 创建可视化器
    print("\n3. 创建可视化器...")
    visualizer = FeatureVisualizerAdvanced(model, config)
    print(f"   可视化器创建成功")
    
    # 4. 检测特征
    print("\n4. 检测特征...")
    
    # 第一张图像的特征
    image1_tensor, _ = preprocess_image(image1, config['target_size'])
    features1 = detect_features(
        model, 
        image1_tensor, 
        top_k=config['top_k'], 
        threshold=config['threshold']
    )
    print(f"   图像1检测到 {features1['num_keypoints']} 个关键点")
    
    # 第二张图像的特征
    features2 = None
    if image2 is not None:
        image2_tensor, _ = preprocess_image(image2, config['target_size'])
        features2 = detect_features(
            model, 
            image2_tensor, 
            top_k=config['top_k'], 
            threshold=config['threshold']
        )
        print(f"   图像2检测到 {features2['num_keypoints']} 个关键点")
        
    # 5. 计算匹配（可选）
    matches_info = None
    if config['compute_matches'] and features2 is not None:
        print("\n5. 计算特征匹配...")
        
        matches = match_features_semantic(
            features1['descriptors'], features2['descriptors'],
            features1['semantic_attention'], features2['semantic_attention']
        )
        
        print(f"   找到 {len(matches[0])} 个匹配")
        
        matches_info = {
            'features1': features1,
            'features2': features2,
            'matches': matches
        }
        
    # 6. 生成可视化报告
    print("\n6. 生成可视化报告...")
    
    report_path = os.path.join(config['output_dir'], 'visualization_report.json')
    report_data = visualizer.create_comprehensive_report(
        image1, features1, matches_info, report_path
    )
    
    print(f"   可视化报告生成完成")
    print(f"   报告数据保存到: {report_path}")
    
    # 7. 生成特征质量分析
    print("\n7. 生成特征质量分析...")
    
    quality_path = os.path.join(visualizer.viz_dir, 'feature_quality.png')
    quality_fig = visualizer.visualize_feature_quality(features1, str(quality_path))
    if quality_fig:
        plt.close(quality_fig)
        print(f"   特征质量分析保存到: {quality_path}")
        
    # 8. 显示摘要
    print("\n8. 可视化摘要:")
    print(f"   - 生成了 {len(report_data['visualizations'])} 种可视化")
    print(f"   - 检测到 {features1['num_keypoints']} 个关键点")
    print(f"   - 平均关键点分数: {np.mean(features1['scores']):.4f}")
    
    if matches_info:
        print(f"   - 匹配数量: {len(matches_info['matches'][0])}")
        print(f"   - 平均匹配分数: {np.mean(matches_info['matches'][2]):.4f}")
        
    # 列出生成的可视化文件
    print("\n   生成的可视化文件:")
    for viz_name, viz_path in report_data['visualizations'].items():
        print(f"     - {viz_name}: {viz_path}")
        
    print("\n" + "="*60)
    print("可视化示例完成！")
    print(f"结果保存在: {config['output_dir']}")
    print("="*60)
    
    return {
        'features1': features1,
        'features2': features2,
        'matches_info': matches_info,
        'report_data': report_data
    }


if __name__ == "__main__":
    # 运行主函数
    results = main()
    
    # 可选：返回结果供其他示例使用
    if results:
        print("\n可视化分析成功完成！")
    else:
        print("\n可视化分析失败！")