"""
特征匹配示例
演示如何使用语义引导的轻量级特征检测器进行图像特征匹配
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
from scipy.spatial.distance import cdist

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入项目模块
from semantic_guided_xfeat_implementation import create_semantic_guided_xfeat
from tools import FeatureVisualizer
from basic_feature_detection import load_image, preprocess_image, detect_features


def match_features(descriptors1, descriptors2, ratio_threshold=0.8, cross_check=True):
    """
    使用最近邻匹配和比率测试进行特征匹配
    """
    if len(descriptors1) == 0 or len(descriptors2) == 0:
        return [], [], [], []
        
    # 计算描述符距离矩阵
    distances = cdist(descriptors1, descriptors2, 'cosine')
    
    # 找到每个描述符的最佳和次佳匹配
    best_matches = np.argmin(distances, axis=1)
    best_distances = distances[np.arange(len(distances)), best_matches]
    
    # 找到次佳匹配
    distances_copy = distances.copy()
    distances_copy[np.arange(len(distances)), best_matches] = np.inf
    second_best_matches = np.argmin(distances_copy, axis=1)
    second_best_distances = distances_copy[np.arange(len(distances)), second_best_matches]
    
    # 比率测试
    ratios = best_distances / (second_best_distances + 1e-8)
    ratio_mask = ratios < ratio_threshold
    
    # 交叉验证
    if cross_check:
        best_matches2 = np.argmin(distances.T, axis=1)
        cross_mask = best_matches2[best_matches] == np.arange(len(descriptors1))
        ratio_mask = ratio_mask & cross_mask
    
    # 获取匹配结果
    valid_indices = np.where(ratio_mask)[0]
    matches1 = valid_indices
    matches2 = best_matches[valid_indices]
    match_scores = 1.0 - ratios[valid_indices]  # 转换为相似度分数
    
    return matches1, matches2, match_scores, ratios[valid_indices]


def match_features_semantic(
    descriptors1, descriptors2, 
    semantic_attention1, semantic_attention2,
    semantic_weight=0.3, ratio_threshold=0.8, cross_check=True
):
    """
    使用语义信息增强的特征匹配
    """
    if len(descriptors1) == 0 or len(descriptors2) == 0:
        return [], [], [], []
        
    # 计算视觉描述符距离
    visual_distances = cdist(descriptors1, descriptors2, 'cosine')
    
    # 计算语义相似度
    h, w = semantic_attention1.shape[1:]
    semantic_attention1_flat = semantic_attention1.reshape(semantic_attention1.shape[0], -1).T
    semantic_attention2_flat = semantic_attention2.reshape(semantic_attention2.shape[0], -1).T
    
    semantic_distances = cdist(semantic_attention1_flat, semantic_attention2_flat, 'cosine')
    
    # 融合视觉和语义距离
    combined_distances = (1 - semantic_weight) * visual_distances + semantic_weight * semantic_distances
    
    # 找到最佳和次佳匹配
    best_matches = np.argmin(combined_distances, axis=1)
    best_distances = combined_distances[np.arange(len(combined_distances)), best_matches]
    
    # 找到次佳匹配
    distances_copy = combined_distances.copy()
    distances_copy[np.arange(len(combined_distances)), best_matches] = np.inf
    second_best_matches = np.argmin(distances_copy, axis=1)
    second_best_distances = distances_copy[np.arange(len(combined_distances)), second_best_matches]
    
    # 比率测试
    ratios = best_distances / (second_best_distances + 1e-8)
    ratio_mask = ratios < ratio_threshold
    
    # 交叉验证
    if cross_check:
        best_matches2 = np.argmin(combined_distances.T, axis=1)
        cross_mask = best_matches2[best_matches] == np.arange(len(descriptors1))
        ratio_mask = ratio_mask & cross_mask
    
    # 获取匹配结果
    valid_indices = np.where(ratio_mask)[0]
    matches1 = valid_indices
    matches2 = best_matches[valid_indices]
    match_scores = 1.0 - ratios[valid_indices]
    
    return matches1, matches2, match_scores, ratios[valid_indices]


def filter_matches_by_homography(
    keypoints1, keypoints2, matches1, matches2,
    ransac_threshold=3.0, min_matches=4
):
    """
    使用单应性矩阵过滤匹配
    """
    if len(matches1) < min_matches:
        return [], [], []
        
    # 提取匹配的关键点
    src_points = keypoints1[matches1].reshape(-1, 1, 2)
    dst_points = keypoints2[matches2].reshape(-1, 1, 2)
    
    # 使用RANSAC计算单应性矩阵
    H, mask = cv2.findHomography(
        src_points, dst_points, 
        cv2.RANSAC, ransac_threshold
    )
    
    if H is None:
        return [], [], []
        
    # 获取内点
    inlier_mask = mask.ravel() == 1
    inlier_matches1 = matches1[inlier_mask]
    inlier_matches2 = matches2[inlier_mask]
    
    return inlier_matches1, inlier_matches2, H


def visualize_matches(
    image1, image2, keypoints1, keypoints2, 
    matches1, matches2, match_scores=None,
    output_path=None, max_matches=100, line_width=1
):
    """
    可视化特征匹配
    """
    # 创建拼接图像
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # 调整图像高度以匹配
    if h1 != h2:
        if h1 > h2:
            image2 = cv2.resize(image2, (w2, h1))
        else:
            image1 = cv2.resize(image1, (w1, h2))
            
    h = max(h1, h2)
    
    # 拼接图像
    composite_image = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    composite_image[:, :w1] = image1
    composite_image[:, w1:w1+w2] = image2
    
    # 限制匹配数量
    if len(matches1) > max_matches:
        if match_scores is not None:
            # 按分数排序
            indices = np.argsort(match_scores)[-max_matches:]
            matches1 = matches1[indices]
            matches2 = matches2[indices]
            match_scores = match_scores[indices]
        else:
            # 随机选择
            indices = np.random.choice(len(matches1), max_matches, replace=False)
            matches1 = matches1[indices]
            matches2 = matches2[indices]
            if match_scores is not None:
                match_scores = match_scores[indices]
                
    # 绘制匹配线
    for i, (idx1, idx2) in enumerate(zip(matches1, matches2)):
        kp1 = keypoints1[idx1]
        kp2 = keypoints2[idx2]
        
        # 计算颜色（基于匹配分数）
        if match_scores is not None:
            score = match_scores[i]
            color_intensity = int(score * 255)
            color = (0, color_intensity, 255 - color_intensity)
        else:
            color = (0, 255, 0)
            
        # 绘制匹配线
        pt1 = (int(kp1[0]), int(kp1[1]))
        pt2 = (int(kp2[0] + w1), int(kp2[1]))
        cv2.line(composite_image, pt1, pt2, color, line_width)
        
        # 绘制关键点
        cv2.circle(composite_image, pt1, 3, color, -1)
        cv2.circle(composite_image, pt2, 3, color, -1)
        
    # 添加文本信息
    cv2.putText(
        composite_image, 
        f"Matches: {len(matches1)}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (255, 255, 255), 
        2
    )
    
    if output_path:
        cv2.imwrite(output_path, composite_image)
        print(f"Matches visualization saved to: {output_path}")
        
    return composite_image


def visualize_matching_comparison(
    image1, image2, keypoints1, keypoints2, 
    matches_visual, matches_semantic, matches_filtered,
    output_path=None
):
    """
    可视化匹配结果对比
    """
    # 调整图像尺寸
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    h = max(h1, h2)
    
    if h1 != h:
        image1 = cv2.resize(image1, (w1, h))
    if h2 != h:
        image2 = cv2.resize(image2, (w2, h))
        
    # 创建大图
    composite_image = np.zeros((h * 2, w1 + w2, 3), dtype=np.uint8)
    
    # 原始图像
    composite_image[:h, :w1] = image1
    composite_image[:h, w1:w1+w2] = image2
    
    # 可视化匹配
    if matches_visual:
        vis_matches = visualize_matches(
            image1, image2, keypoints1, keypoints2,
            matches_visual[0], matches_visual[1],
            max_matches=50, line_width=1
        )
        composite_image[h:, :w1+w2] = vis_matches
        
        # 添加标签
        cv2.putText(
            composite_image, 
            "Visual Matches", 
            (10, h + 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2
        )
        
    if matches_semantic:
        # 创建语义匹配可视化
        semantic_vis = visualize_matches(
            image1, image2, keypoints1, keypoints2,
            matches_semantic[0], matches_semantic[1],
            max_matches=50, line_width=1
        )
        
        # 如果已有视觉匹配，替换为语义匹配
        if matches_visual:
            composite_image[h:, :w1+w2] = semantic_vis
            cv2.putText(
                composite_image, 
                "Semantic Matches", 
                (10, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2
            )
        
    if matches_filtered:
        # 创建过滤后匹配可视化
        filtered_vis = visualize_matches(
            image1, image2, keypoints1, keypoints2,
            matches_filtered[0], matches_filtered[1],
            max_matches=50, line_width=2
        )
        
        # 如果已有其他匹配，添加过滤后匹配
        if matches_visual or matches_semantic:
            # 创建更大的图像
            large_composite = np.zeros((h * 3, w1 + w2, 3), dtype=np.uint8)
            large_composite[:h, :w1+w2] = composite_image[:h, :w1+w2]
            large_composite[h:2*h, :w1+w2] = composite_image[h:, :w1+w2]
            large_composite[2*h:, :w1+w2] = filtered_vis
            
            # 添加标签
            cv2.putText(
                large_composite, 
                "Filtered Matches", 
                (10, 2*h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2
            )
            
            composite_image = large_composite
        else:
            composite_image[h:, :w1+w2] = filtered_vis
            cv2.putText(
                composite_image, 
                "Filtered Matches", 
                (10, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2
            )
            
    if output_path:
        cv2.imwrite(output_path, composite_image)
        print(f"Matching comparison saved to: {output_path}")
        
    return composite_image


def print_matching_statistics(
    features1, features2, 
    matches_visual, matches_semantic, matches_filtered
):
    """
    打印匹配统计信息
    """
    print("\n" + "="*60)
    print("特征匹配统计信息")
    print("="*60)
    
    print(f"图像1关键点数量: {features1['num_keypoints']}")
    print(f"图像2关键点数量: {features2['num_keypoints']}")
    
    if matches_visual:
        print(f"\n视觉匹配数量: {len(matches_visual[0])}")
        if len(matches_visual[2]) > 0:
            print(f"视觉匹配分数范围: {matches_visual[2].min():.4f} - {matches_visual[2].max():.4f}")
            print(f"视觉匹配平均分数: {matches_visual[2].mean():.4f}")
            
    if matches_semantic:
        print(f"\n语义匹配数量: {len(matches_semantic[0])}")
        if len(matches_semantic[2]) > 0:
            print(f"语义匹配分数范围: {matches_semantic[2].min():.4f} - {matches_semantic[2].max():.4f}")
            print(f"语义匹配平均分数: {matches_semantic[2].mean():.4f}")
            
    if matches_filtered:
        print(f"\n过滤后匹配数量: {len(matches_filtered[0])}")
        if matches_filtered[2] is not None:
            print(f"内点率: {len(matches_filtered[0]) / max(len(matches_visual[0]), 1) * 100:.2f}%")
            
    print("="*60 + "\n")


def create_test_images(output_dir):
    """
    创建测试图像对
    """
    # 创建第一张测试图像
    image1 = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # 添加一些几何形状
    cv2.circle(image1, (256, 256), 100, (255, 0, 0), -1)
    cv2.rectangle(image1, (100, 100), (200, 200), (0, 255, 0), -1)
    cv2.ellipse(image1, (400, 150), (50, 30), 0, 0, 360, (0, 0, 255), -1)
    
    # 添加噪声
    noise = np.random.normal(0, 10, image1.shape).astype(np.int16)
    image1 = np.clip(image1.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 创建第二张图像（变换后的版本）
    # 应用仿射变换
    M = np.float32([[0.8, 0.1, 50], [-0.1, 0.8, 30]])
    image2 = cv2.warpAffine(image1, M, (512, 512))
    
    # 添加不同的噪声
    noise2 = np.random.normal(0, 15, image2.shape).astype(np.int16)
    image2 = np.clip(image2.astype(np.int16) + noise2, 0, 255).astype(np.uint8)
    
    # 保存图像
    image1_path = os.path.join(output_dir, 'test_image1.jpg')
    image2_path = os.path.join(output_dir, 'test_image2.jpg')
    
    cv2.imwrite(image1_path, image1)
    cv2.imwrite(image2_path, image2)
    
    return image1_path, image2_path


def main():
    """
    主函数
    """
    # 配置参数
    config = {
        'model_path': None,
        'image1_path': './examples/test_image1.jpg',
        'image2_path': './examples/test_image2.jpg',
        'top_k': 2000,
        'threshold': 0.005,
        'ratio_threshold': 0.8,
        'semantic_weight': 0.3,
        'ransac_threshold': 3.0,
        'output_dir': './examples/output',
        'target_size': 256
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print("语义引导的轻量级特征检测器 - 特征匹配示例")
    print("="*60)
    
    # 1. 创建模型
    print("1. 创建模型...")
    model = create_semantic_guided_xfeat()
    model.eval()
    print(f"   模型创建成功")
    
    # 2. 加载或创建测试图像
    print("\n2. 加载测试图像...")
    try:
        image1 = load_image(config['image1_path'])
        image2 = load_image(config['image2_path'])
        print(f"   图像加载成功")
    except Exception as e:
        print(f"   图像加载失败: {e}")
        print("   创建测试图像...")
        
        config['image1_path'], config['image2_path'] = create_test_images(config['output_dir'])
        image1 = load_image(config['image1_path'])
        image2 = load_image(config['image2_path'])
        print(f"   测试图像创建成功")
        
    print(f"   图像1尺寸: {image1.shape}")
    print(f"   图像2尺寸: {image2.shape}")
    
    # 3. 预处理图像
    print("\n3. 预处理图像...")
    image1_tensor, image1_gray = preprocess_image(image1, config['target_size'])
    image2_tensor, image2_gray = preprocess_image(image2, config['target_size'])
    print(f"   预处理完成")
    
    # 4. 特征检测
    print("\n4. 检测特征...")
    
    # 检测图像1的特征
    features1 = detect_features(
        model, 
        image1_tensor, 
        top_k=config['top_k'], 
        threshold=config['threshold']
    )
    print(f"   图像1检测到 {features1['num_keypoints']} 个关键点")
    
    # 检测图像2的特征
    features2 = detect_features(
        model, 
        image2_tensor, 
        top_k=config['top_k'], 
        threshold=config['threshold']
    )
    print(f"   图像2检测到 {features2['num_keypoints']} 个关键点")
    
    # 5. 特征匹配
    print("\n5. 特征匹配...")
    
    # 视觉匹配
    matches_visual = match_features(
        features1['descriptors'], features2['descriptors'],
        ratio_threshold=config['ratio_threshold'], cross_check=True
    )
    print(f"   视觉匹配数量: {len(matches_visual[0])}")
    
    # 语义匹配
    matches_semantic = match_features_semantic(
        features1['descriptors'], features2['descriptors'],
        features1['semantic_attention'], features2['semantic_attention'],
        semantic_weight=config['semantic_weight'],
        ratio_threshold=config['ratio_threshold'], cross_check=True
    )
    print(f"   语义匹配数量: {len(matches_semantic[0])}")
    
    # 使用单应性矩阵过滤匹配
    matches_filtered = filter_matches_by_homography(
        features1['keypoints'], features2['keypoints'],
        matches_semantic[0], matches_semantic[1],
        ransac_threshold=config['ransac_threshold']
    )
    print(f"   过滤后匹配数量: {len(matches_filtered[0])}")
    
    # 6. 打印统计信息
    print("\n6. 匹配统计信息:")
    print_matching_statistics(
        features1, features2,
        matches_visual, matches_semantic, matches_filtered
    )
    
    # 7. 可视化结果
    print("\n7. 可视化结果...")
    
    # 可视化匹配对比
    comparison_output = os.path.join(config['output_dir'], 'matching_comparison.jpg')
    comparison_vis = visualize_matching_comparison(
        image1, image2,
        features1['keypoints'], features2['keypoints'],
        matches_visual, matches_semantic, matches_filtered,
        output_path=comparison_output
    )
    
    # 8. 保存匹配数据
    print("\n8. 保存匹配数据...")
    import json
    
    # 准备保存的数据
    save_data = {
        'image1_path': config['image1_path'],
        'image2_path': config['image2_path'],
        'image1_size': list(image1.shape),
        'image2_size': list(image2.shape),
        'features1': {
            'num_keypoints': features1['num_keypoints'],
            'keypoints': features1['keypoints'].tolist(),
            'scores': features1['scores'].tolist()
        },
        'features2': {
            'num_keypoints': features2['num_keypoints'],
            'keypoints': features2['keypoints'].tolist(),
            'scores': features2['scores'].tolist()
        },
        'matches': {
            'visual': {
                'count': len(matches_visual[0]),
                'matches1': matches_visual[0].tolist(),
                'matches2': matches_visual[1].tolist(),
                'scores': matches_visual[2].tolist()
            },
            'semantic': {
                'count': len(matches_semantic[0]),
                'matches1': matches_semantic[0].tolist(),
                'matches2': matches_semantic[1].tolist(),
                'scores': matches_semantic[2].tolist()
            },
            'filtered': {
                'count': len(matches_filtered[0]),
                'matches1': matches_filtered[0].tolist(),
                'matches2': matches_filtered[1].tolist()
            }
        },
        'config': config,
        'timestamp': str(torch.datetime.now())
    }
    
    # 保存为JSON文件
    matches_output = os.path.join(config['output_dir'], 'matching_results.json')
    with open(matches_output, 'w') as f:
        json.dump(save_data, f, indent=2)
        
    print(f"   匹配数据保存到: {matches_output}")
    
    # 9. 显示结果
    print("\n9. 显示结果...")
    
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(comparison_vis, cv2.COLOR_BGR2RGB))
    plt.title('Feature Matching Comparison')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'matching_summary.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("特征匹配示例完成！")
    print(f"结果保存在: {config['output_dir']}")
    print("="*60)
    
    return {
        'features1': features1,
        'features2': features2,
        'matches_visual': matches_visual,
        'matches_semantic': matches_semantic,
        'matches_filtered': matches_filtered
    }


if __name__ == "__main__":
    # 运行主函数
    results = main()
    
    # 可选：返回结果供其他示例使用
    if results:
        print("\n特征匹配成功完成！")
    else:
        print("\n特征匹配失败！")