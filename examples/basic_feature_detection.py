"""
基础特征检测示例
演示如何使用语义引导的轻量级特征检测器进行基础的特征检测
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入项目模块
from semantic_guided_xfeat_implementation import create_semantic_guided_xfeat
from tools import FeatureVisualizer


def load_image(image_path):
    """
    加载图像
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
        
    return image


def preprocess_image(image, target_size=256):
    """
    预处理图像
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
        
    # 调整大小
    image_resized = cv2.resize(image_gray, (target_size, target_size))
    
    # 归一化
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # 转换为tensor
    image_tensor = torch.from_numpy(image_normalized)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    return image_tensor, image_gray


def detect_features(model, image_tensor, top_k=1000, threshold=0.01):
    """
    检测特征
    """
    model.eval()
    
    with torch.no_grad():
        features = model.detectAndCompute(image_tensor, top_k=top_k)
        
    # 提取结果
    result = features[0]
    
    # 过滤低分关键点
    scores = result['scores'].cpu().numpy()
    keypoints = result['keypoints'].cpu().numpy()
    descriptors = result['descriptors'].cpu().numpy()
    semantic_attention = result['semantic_attention'].cpu().numpy()
    
    # 根据阈值过滤
    valid_indices = scores > threshold
    filtered_keypoints = keypoints[valid_indices]
    filtered_scores = scores[valid_indices]
    filtered_descriptors = descriptors[valid_indices]
    
    # 调整关键点坐标到原始图像尺寸
    original_h, original_w = image_tensor.shape[2], image_tensor.shape[3]
    scale_x = original_w / target_size
    scale_y = original_h / target_size
    
    filtered_keypoints[:, 0] *= scale_x
    filtered_keypoints[:, 1] *= scale_y
    
    return {
        'keypoints': filtered_keypoints,
        'scores': filtered_scores,
        'descriptors': filtered_descriptors,
        'semantic_attention': semantic_attention,
        'num_keypoints': len(filtered_keypoints)
    }


def visualize_keypoints(image, features, output_path=None, max_keypoints=100):
    """
    可视化关键点
    """
    # 创建图像副本
    vis_image = image.copy()
    
    # 获取关键点和分数
    keypoints = features['keypoints']
    scores = features['scores']
    
    if len(keypoints) == 0:
        print("No keypoints found")
        return vis_image
        
    # 按分数排序并选择top-k
    if len(keypoints) > max_keypoints:
        indices = np.argsort(scores)[-max_keypoints:]
        keypoints = keypoints[indices]
        scores = scores[indices]
        
    # 绘制关键点
    for i, (kp, score) in enumerate(zip(keypoints, scores)):
        x, y = int(kp[0]), int(kp[1])
        
        # 根据分数确定颜色和大小
        color_intensity = int(score * 255)
        color = (0, color_intensity, 255 - color_intensity)  # BGR格式
        radius = int(3 + score * 7)
        
        # 绘制圆圈
        cv2.circle(vis_image, (x, y), radius, color, 2)
        
        # 绘制中心点
        cv2.circle(vis_image, (x, y), 1, (255, 255, 255), -1)
        
    # 添加文本信息
    cv2.putText(
        vis_image, 
        f"Keypoints: {len(keypoints)}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (255, 255, 255), 
        2
    )
    
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"Keypoints visualization saved to: {output_path}")
        
    return vis_image


def visualize_semantic_attention(image, semantic_attention, output_path=None):
    """
    可视化语义注意力
    """
    # 获取语义类别
    semantic_classes = np.argmax(semantic_attention, axis=0)
    
    # 创建颜色映射
    h, w = semantic_classes.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 为每个语义类别分配颜色
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for i in range(20):
        mask = semantic_classes == i
        color_map[mask] = (colors[i][:3] * 255).astype(np.uint8)
        
    # 调整尺寸以匹配图像
    if color_map.shape[:2] != image.shape[:2]:
        color_map = cv2.resize(color_map, (image.shape[1], image.shape[0]))
        
    # 创建叠加图像
    alpha = 0.6
    overlay_image = cv2.addWeighted(image, 1 - alpha, color_map, alpha, 0)
    
    # 创建拼接图像
    h, w = image.shape[:2]
    composite_image = np.zeros((h, w * 2, 3), dtype=np.uint8)
    composite_image[:, :w] = image
    composite_image[:, w:] = overlay_image
    
    # 添加标签
    cv2.putText(
        composite_image, 
        "Original", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (255, 255, 255), 
        2
    )
    cv2.putText(
        composite_image, 
        "Semantic Attention", 
        (w + 10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (255, 255, 255), 
        2
    )
    
    if output_path:
        cv2.imwrite(output_path, composite_image)
        print(f"Semantic attention visualization saved to: {output_path}")
        
    return composite_image


def print_feature_statistics(features):
    """
    打印特征统计信息
    """
    print("\n" + "="*50)
    print("特征检测统计信息")
    print("="*50)
    print(f"检测到的关键点数量: {features['num_keypoints']}")
    
    if features['num_keypoints'] > 0:
        scores = features['scores']
        print(f"分数范围: {scores.min():.4f} - {scores.max():.4f}")
        print(f"平均分数: {scores.mean():.4f}")
        print(f"分数标准差: {scores.std():.4f}")
        
        # 描述符信息
        descriptors = features['descriptors']
        print(f"描述符维度: {descriptors.shape[1]}")
        print(f"描述符数值范围: {descriptors.min():.4f} - {descriptors.max():.4f}")
        
        # 语义注意力信息
        semantic_attention = features['semantic_attention']
        print(f"语义注意力形状: {semantic_attention.shape}")
        print(f"语义类别数量: {semantic_attention.shape[0]}")
        
    print("="*50 + "\n")


def main():
    """
    主函数
    """
    # 配置参数
    config = {
        'model_path': None,  # 使用默认模型
        'image_path': './examples/test_image.jpg',  # 测试图像路径
        'top_k': 1000,
        'threshold': 0.01,
        'output_dir': './examples/output',
        'target_size': 256
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print("语义引导的轻量级特征检测器 - 基础特征检测示例")
    print("="*60)
    
    # 1. 创建模型
    print("1. 创建模型...")
    model = create_semantic_guided_xfeat()
    model.eval()
    print(f"   模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. 加载图像
    print("\n2. 加载图像...")
    try:
        image = load_image(config['image_path'])
        print(f"   图像加载成功: {config['image_path']}")
        print(f"   图像尺寸: {image.shape}")
    except Exception as e:
        print(f"   图像加载失败: {e}")
        print("   创建测试图像...")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(config['image_path'], test_image)
        image = load_image(config['image_path'])
        print(f"   测试图像创建成功: {config['image_path']}")
        
    # 3. 预处理图像
    print("\n3. 预处理图像...")
    image_tensor, image_gray = preprocess_image(image, config['target_size'])
    print(f"   预处理完成，tensor形状: {image_tensor.shape}")
    
    # 4. 特征检测
    print("\n4. 检测特征...")
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    import time
    cpu_start_time = time.time()
    
    if start_time and end_time and torch.cuda.is_available():
        start_time.record()
        
    features = detect_features(
        model, 
        image_tensor, 
        top_k=config['top_k'], 
        threshold=config['threshold']
    )
    
    if start_time and end_time and torch.cuda.is_available():
        end_time.record()
        torch.cuda.synchronize()
        gpu_time = start_time.elapsed_time(end_time)
        print(f"   GPU推理时间: {gpu_time:.2f} ms")
        
    cpu_end_time = time.time()
    cpu_time = (cpu_end_time - cpu_start_time) * 1000
    print(f"   CPU推理时间: {cpu_time:.2f} ms")
    
    # 5. 打印统计信息
    print("\n5. 特征统计信息:")
    print_feature_statistics(features)
    
    # 6. 可视化结果
    print("\n6. 可视化结果...")
    
    # 可视化关键点
    keypoints_output = os.path.join(config['output_dir'], 'keypoints_visualization.jpg')
    keypoints_vis = visualize_keypoints(
        image, 
        features, 
        output_path=keypoints_output,
        max_keypoints=200
    )
    
    # 可视化语义注意力
    semantic_output = os.path.join(config['output_dir'], 'semantic_attention_visualization.jpg')
    semantic_vis = visualize_semantic_attention(
        image_gray,
        features['semantic_attention'],
        output_path=semantic_output
    )
    
    # 7. 保存特征数据
    print("\n7. 保存特征数据...")
    import json
    
    # 准备保存的数据
    save_data = {
        'image_path': config['image_path'],
        'image_size': list(image.shape),
        'num_keypoints': features['num_keypoints'],
        'keypoints': features['keypoints'].tolist(),
        'scores': features['scores'].tolist(),
        'descriptors': features['descriptors'].tolist(),
        'semantic_attention_shape': list(features['semantic_attention'].shape),
        'config': config,
        'timestamp': str(torch.datetime.now())
    }
    
    # 保存为JSON文件
    features_output = os.path.join(config['output_dir'], 'detected_features.json')
    with open(features_output, 'w') as f:
        json.dump(save_data, f, indent=2)
        
    print(f"   特征数据保存到: {features_output}")
    
    # 8. 显示结果
    print("\n8. 显示结果...")
    
    # 使用matplotlib显示结果
    plt.figure(figsize=(15, 5))
    
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # 关键点可视化
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(keypoints_vis, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected Keypoints ({features["num_keypoints"]})')
    plt.axis('off')
    
    # 语义注意力可视化
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(semantic_vis, cv2.COLOR_BGR2RGB))
    plt.title('Semantic Attention')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'feature_detection_summary.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("基础特征检测示例完成！")
    print(f"结果保存在: {config['output_dir']}")
    print("="*60)
    
    return features


if __name__ == "__main__":
    # 运行主函数
    features = main()
    
    # 可选：返回特征数据供其他示例使用
    if features:
        print("\n特征检测成功完成！")
    else:
        print("\n特征检测失败！")