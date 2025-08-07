"""
工具脚本集合
为语义引导的轻量级特征检测器提供各种实用工具
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import onnx
import onnxruntime as ort
from datetime import datetime

# 导入项目模块
from semantic_guided_xfeat_implementation import SemanticGuidedXFeat
from dataset_processing import ImageTransforms


class ModelConverter:
    """
    模型转换工具
    支持PyTorch到ONNX和TensorRT的转换
    """
    def __init__(self, model_path, output_dir="./converted_models"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        
    def _load_model(self):
        """
        加载PyTorch模型
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # 创建模型实例
        model = SemanticGuidedXFeat()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        
        return model
    
    def convert_to_onnx(self, input_size=(1, 1, 256, 256), opset_version=11):
        """
        转换为ONNX格式
        """
        print("Converting model to ONNX format...")
        
        # 创建测试输入
        dummy_input = torch.randn(input_size, device=self.device)
        
        # 定义输出名称
        output_names = [
            'features',
            'keypoint_logits',
            'heatmap',
            'descriptors',
            'semantic_attention'
        ]
        
        # 导出ONNX模型
        onnx_path = self.output_dir / "semantic_guided_xfeat.onnx"
        
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=output_names,
            dynamic_axes={
                'input': {0: 'batch_size'},
                'features': {0: 'batch_size'},
                'keypoint_logits': {0: 'batch_size'},
                'heatmap': {0: 'batch_size'},
                'descriptors': {0: 'batch_size'},
                'semantic_attention': {0: 'batch_size'}
            }
        )
        
        print(f"ONNX model saved to: {onnx_path}")
        
        # 验证ONNX模型
        self._verify_onnx_model(onnx_path, dummy_input)
        
        return onnx_path
    
    def _verify_onnx_model(self, onnx_path, dummy_input):
        """
        验证ONNX模型
        """
        print("Verifying ONNX model...")
        
        # 加载ONNX模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # 创建ONNX Runtime会话
        ort_session = ort.InferenceSession(str(onnx_path))
        
        # 获取输入输出名称
        input_name = ort_session.get_inputs()[0].name
        output_names = [output.name for output in ort_session.get_outputs()]
        
        # 运行推理
        ort_inputs = {input_name: dummy_input.cpu().numpy()}
        ort_outputs = ort_session.run(output_names, ort_inputs)
        
        # PyTorch推理
        with torch.no_grad():
            torch_outputs = self.model(dummy_input)
            
        # 比较结果
        print("Comparing PyTorch and ONNX outputs...")
        for i, (torch_out, ort_out) in enumerate(zip(torch_outputs, ort_outputs)):
            diff = np.abs(torch_out.cpu().numpy() - ort_out).max()
            print(f"  Output {i}: Max difference = {diff}")
            
            if diff > 1e-3:
                print(f"  Warning: Large difference detected in output {i}")
            else:
                print(f"  Output {i}: OK")
                
    def convert_to_tensorrt(self, onnx_path=None, fp16=False):
        """
        转换为TensorRT格式（需要TensorRT环境）
        """
        try:
            import tensorrt as trt
        except ImportError:
            print("TensorRT not installed. Skipping TensorRT conversion.")
            return None
            
        if onnx_path is None:
            onnx_path = self.output_dir / "semantic_guided_xfeat.onnx"
            
        if not onnx_path.exists():
            print("ONNX model not found. Please convert to ONNX first.")
            return None
            
        print("Converting ONNX model to TensorRT format...")
        
        # 创建TensorRT logger
        logger = trt.Logger(trt.Logger.WARNING)
        
        # 创建构建器
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # 解析ONNX模型
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print("ERROR: Failed to parse ONNX model.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
                
        # 创建配置
        config = builder.create_builder_config()
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            
        # 构建TensorRT引擎
        print("Building TensorRT engine...")
        engine_path = self.output_dir / "semantic_guided_xfeat.trt"
        
        with open(engine_path, 'wb') as f:
            f.write(builder.build_serialized_network(network, config))
            
        print(f"TensorRT engine saved to: {engine_path}")
        return engine_path


class BatchInference:
    """
    批量推理工具
    支持对图像目录进行批量特征提取和匹配
    """
    def __init__(self, model_path, device='cuda'):
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        
        # 加载模型
        self.model = self._load_model()
        
        # 图像变换
        self.transform = ImageTransforms(image_size=256, is_training=False)
        
    def _load_model(self):
        """
        加载模型
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        model = SemanticGuidedXFeat()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        
        return model
    
    def extract_features_batch(self, image_dir, output_dir, top_k=4096):
        """
        批量提取特征
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 获取图像文件列表
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(image_dir.glob(ext))
            
        image_files = sorted(image_files)
        print(f"Found {len(image_files)} images")
        
        # 批量处理
        all_features = []
        
        for image_file in tqdm(image_files, desc="Extracting features"):
            try:
                # 加载图像
                image = self.transform(str(image_file))
                image = image.unsqueeze(0).to(self.device)
                
                # 提取特征
                with torch.no_grad():
                    features = self.model.detectAndCompute(image, top_k=top_k)
                    
                # 保存特征
                feature_data = {
                    'image_path': str(image_file),
                    'keypoints': features[0]['keypoints'].cpu().numpy(),
                    'scores': features[0]['scores'].cpu().numpy(),
                    'descriptors': features[0]['descriptors'].cpu().numpy(),
                    'semantic_attention': features[0]['semantic_attention'].cpu().numpy()
                }
                
                all_features.append(feature_data)
                
                # 保存单个图像的特征
                feature_path = output_dir / f"{image_file.stem}_features.json"
                with open(feature_path, 'w') as f:
                    json.dump(feature_data, f, indent=2)
                    
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                
        # 保存所有特征
        all_features_path = output_dir / "all_features.json"
        with open(all_features_path, 'w') as f:
            json.dump(all_features, f, indent=2)
            
        print(f"Features saved to: {output_dir}")
        return all_features
    
    def match_image_pairs(self, image_pairs_file, output_dir, threshold=0.8):
        """
        匹配图像对
        """
        image_pairs_file = Path(image_pairs_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 加载图像对列表
        with open(image_pairs_file, 'r') as f:
            image_pairs = json.load(f)
            
        print(f"Loaded {len(image_pairs)} image pairs")
        
        # 批量匹配
        all_matches = []
        
        for pair in tqdm(image_pairs, desc="Matching images"):
            try:
                img1_path = pair['image1']
                img2_path = pair['image2']
                
                # 加载图像
                img1 = self.transform(img1_path).unsqueeze(0).to(self.device)
                img2 = self.transform(img2_path).unsqueeze(0).to(self.device)
                
                # 提取特征
                with torch.no_grad():
                    features1 = self.model.detectAndCompute(img1)
                    features2 = self.model.detectAndCompute(img2)
                    
                # 匹配特征
                matches = self._match_features(features1[0], features2[0], threshold)
                
                # 保存匹配结果
                match_data = {
                    'image1': img1_path,
                    'image2': img2_path,
                    'matches': matches,
                    'num_matches': len(matches)
                }
                
                all_matches.append(match_data)
                
                # 保存单个匹配结果
                pair_name = f"{Path(img1_path).stem}_{Path(img2_path).stem}"
                match_path = output_dir / f"{pair_name}_matches.json"
                with open(match_path, 'w') as f:
                    json.dump(match_data, f, indent=2)
                    
            except Exception as e:
                print(f"Error matching {img1_path} and {img2_path}: {e}")
                
        # 保存所有匹配结果
        all_matches_path = output_dir / "all_matches.json"
        with open(all_matches_path, 'w') as f:
            json.dump(all_matches, f, indent=2)
            
        print(f"Matches saved to: {output_dir}")
        return all_matches
    
    def _match_features(self, features1, features2, threshold):
        """
        匹配两组特征
        """
        desc1 = features1['descriptors']
        desc2 = features2['descriptors']
        
        if len(desc1) == 0 or len(desc2) == 0:
            return []
            
        # 计算距离矩阵
        dist_matrix = torch.cdist(desc1, desc2)
        
        # 最近邻匹配
        matches = []
        for i in range(len(desc1)):
            distances = dist_matrix[i]
            min_dist, min_idx = torch.min(distances, 0)
            
            # 比率测试
            second_min_dist = torch.kthvalue(distances, 1)[0]
            ratio = min_dist / (second_min_dist + 1e-8)
            
            if ratio < threshold:
                matches.append([i, min_idx.item()])
                
        return matches


class FeatureVisualizer:
    """
    特征可视化工具
    提供特征点、匹配和语义注意力的可视化功能
    """
    def __init__(self, save_dir="./visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 颜色映射
        self.colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
    def visualize_keypoints(self, image_path, features, save_name=None, max_keypoints=100):
        """
        可视化关键点
        """
        # 加载图像
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取关键点
        keypoints = features['keypoints']
        scores = features['scores']
        
        if len(keypoints) == 0:
            print("No keypoints found")
            return None
            
        # 按分数排序并选择top-k
        if len(keypoints) > max_keypoints:
            indices = np.argsort(scores)[-max_keypoints:]
            keypoints = keypoints[indices]
            scores = scores[indices]
            
        # 创建图像副本
        vis_image = image.copy()
        
        # 绘制关键点
        for i, (kp, score) in enumerate(zip(keypoints, scores)):
            x, y = int(kp[0]), int(kp[1])
            
            # 根据分数确定颜色和大小
            color = plt.cm.hot(score / max(scores))
            radius = int(3 + score * 5)
            
            cv2.circle(vis_image, (x, y), radius, color[:3] * 255, 2)
            
        # 保存结果
        if save_name is None:
            save_name = Path(image_path).stem + "_keypoints"
            
        save_path = self.save_dir / f"{save_name}.jpg"
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image)
        plt.title(f"Keypoints: {len(keypoints)}")
        plt.axis('off')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Keypoints visualization saved to: {save_path}")
        return save_path
    
    def visualize_matches(self, image1_path, image2_path, matches, features1, features2, save_name=None):
        """
        可视化匹配结果
        """
        # 加载图像
        img1 = cv2.imread(str(image1_path))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(str(image2_path))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # 获取关键点
        kpts1 = features1['keypoints']
        kpts2 = features2['keypoints']
        
        if len(kpts1) == 0 or len(kpts2) == 0 or len(matches) == 0:
            print("No valid matches found")
            return None
            
        # 创建拼接图像
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h = max(h1, h2)
        w = w1 + w2
        
        vis_image = np.zeros((h, w, 3), dtype=np.uint8)
        vis_image[:h1, :w1] = img1
        vis_image[:h2, w1:w1+w2] = img2
        
        # 绘制匹配
        for match in matches:
            idx1, idx2 = match
            
            if idx1 < len(kpts1) and idx2 < len(kpts2):
                kp1 = kpts1[idx1]
                kp2 = kpts2[idx2]
                
                x1, y1 = int(kp1[0]), int(kp1[1])
                x2, y2 = int(kp2[0] + w1), int(kp2[1])
                
                # 随机颜色
                color = np.random.rand(3) * 255
                
                cv2.line(vis_image, (x1, y1), (x2, y2), color, 1)
                cv2.circle(vis_image, (x1, y1), 3, color, -1)
                cv2.circle(vis_image, (x2, y2), 3, color, -1)
                
        # 保存结果
        if save_name is None:
            save_name = f"{Path(image1_path).stem}_{Path(image2_path).stem}_matches"
            
        save_path = self.save_dir / f"{save_name}.jpg"
        plt.figure(figsize=(16, 8))
        plt.imshow(vis_image)
        plt.title(f"Matches: {len(matches)}")
        plt.axis('off')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Matches visualization saved to: {save_path}")
        return save_path
    
    def visualize_semantic_attention(self, image_path, semantic_attention, save_name=None):
        """
        可视化语义注意力
        """
        # 加载图像
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取语义注意力
        if isinstance(semantic_attention, torch.Tensor):
            semantic_attention = semantic_attention.cpu().numpy()
            
        # 获取语义类别
        semantic_classes = np.argmax(semantic_attention, axis=0)
        
        # 创建颜色映射
        h, w = semantic_classes.shape
        color_map = np.zeros((h, w, 3))
        
        for i in range(20):  # 20个语义类别
            mask = semantic_classes == i
            color_map[mask] = self.colors[i][:3]
            
        # 调整尺寸以匹配图像
        if color_map.shape[:2] != image.shape[:2]:
            color_map = cv2.resize(color_map, (image.shape[1], image.shape[0]))
            
        # 创建叠加图像
        alpha = 0.6
        overlay_image = image * (1 - alpha) + color_map * 255 * alpha
        overlay_image = overlay_image.astype(np.uint8)
        
        # 保存结果
        if save_name is None:
            save_name = Path(image_path).stem + "_semantic_attention"
            
        save_path = self.save_dir / f"{save_name}.jpg"
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(color_map)
        axes[1].set_title("Semantic Attention")
        axes[1].axis('off')
        
        axes[2].imshow(overlay_image)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Semantic attention visualization saved to: {save_path}")
        return save_path
    
    def create_feature_report(self, results, output_name="feature_report"):
        """
        创建特征分析报告
        """
        report_path = self.save_dir / f"{output_name}.html"
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Feature Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin-bottom: 30px; }
                .metric { background: #f0f0f0; padding: 10px; margin: 5px 0; }
                .plot { text-align: center; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Feature Analysis Report</h1>
            <p>Generated on: {}</p>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # 添加统计信息
        if 'model_size' in results:
            html_content += """
            <div class="section">
                <h2>Model Information</h2>
                <div class="metric">Total Parameters: {:,}</div>
                <div class="metric">Model Size: {:.2f} MB</div>
            </div>
            """.format(results['model_size']['total_params'], results['model_size']['model_size_mb'])
            
        if 'runtime' in results:
            html_content += """
            <div class="section">
                <h2>Runtime Performance</h2>
                <div class="metric">Average Time: {:.2f} ms</div>
                <div class="metric">FPS: {:.1f}</div>
            </div>
            """.format(results['runtime']['avg_time_ms'], results['runtime']['fps'])
            
        if 'matching' in results:
            html_content += """
            <div class="section">
                <h2>Matching Performance</h2>
                <div class="metric">Inlier Rate: {:.4f}</div>
                <div class="metric">Precision: {:.4f}</div>
                <div class="metric">Recall: {:.4f}</div>
                <div class="metric">F1 Score: {:.4f}</div>
            </div>
            """.format(
                results['matching']['inlier_rate'],
                results['matching']['precision'],
                results['matching']['recall'],
                results['matching']['f1_score']
            )
            
        html_content += """
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"Feature report saved to: {report_path}")
        return report_path


def main():
    """
    主函数 - 命令行接口
    """
    parser = argparse.ArgumentParser(description="Semantic-Guided XFeat Tools")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 模型转换命令
    convert_parser = subparsers.add_parser('convert', help='Convert model to different formats')
    convert_parser.add_argument('--model', required=True, help='Path to PyTorch model')
    convert_parser.add_argument('--output-dir', default='./converted_models', help='Output directory')
    convert_parser.add_argument('--format', choices=['onnx', 'tensorrt', 'both'], default='both', help='Target format')
    convert_parser.add_argument('--fp16', action='store_true', help='Use FP16 precision for TensorRT')
    
    # 批量推理命令
    inference_parser = subparsers.add_parser('inference', help='Batch inference on images')
    inference_parser.add_argument('--model', required=True, help='Path to PyTorch model')
    inference_parser.add_argument('--mode', choices=['extract', 'match'], required=True, help='Inference mode')
    inference_parser.add_argument('--input', required=True, help='Input directory or file')
    inference_parser.add_argument('--output', required=True, help='Output directory')
    inference_parser.add_argument('--top-k', type=int, default=4096, help='Number of keypoints to extract')
    inference_parser.add_argument('--threshold', type=float, default=0.8, help='Matching threshold')
    
    # 可视化命令
    viz_parser = subparsers.add_parser('visualize', help='Visualize features and matches')
    viz_parser.add_argument('--mode', choices=['keypoints', 'matches', 'semantic', 'report'], required=True, help='Visualization mode')
    viz_parser.add_argument('--input', required=True, help='Input file or directory')
    viz_parser.add_argument('--output-dir', default='./visualizations', help='Output directory')
    viz_parser.add_argument('--features', help='Path to features file (for keypoints and matches)')
    viz_parser.add_argument('--image2', help='Path to second image (for matches)')
    viz_parser.add_argument('--features2', help='Path to second features file (for matches)')
    viz_parser.add_argument('--matches', help='Path to matches file (for matches)')
    
    args = parser.parse_args()
    
    if args.command == 'convert':
        converter = ModelConverter(args.model, args.output_dir)
        
        if args.format in ['onnx', 'both']:
            onnx_path = converter.convert_to_onnx()
            
        if args.format in ['tensorrt', 'both']:
            converter.convert_to_tensorrt(onnx_path if 'onnx_path' in locals() else None, args.fp16)
            
    elif args.command == 'inference':
        inference = BatchInference(args.model)
        
        if args.mode == 'extract':
            inference.extract_features_batch(args.input, args.output, args.top_k)
        elif args.mode == 'match':
            inference.match_image_pairs(args.input, args.output, args.threshold)
            
    elif args.command == 'visualize':
        visualizer = FeatureVisualizer(args.output_dir)
        
        if args.mode == 'keypoints':
            with open(args.features, 'r') as f:
                features = json.load(f)
            visualizer.visualize_keypoints(args.input, features)
            
        elif args.mode == 'matches':
            with open(args.features, 'r') as f:
                features1 = json.load(f)
            with open(args.features2, 'r') as f:
                features2 = json.load(f)
            with open(args.matches, 'r') as f:
                matches = json.load(f)['matches']
            visualizer.visualize_matches(args.input, args.image2, matches, features1, features2)
            
        elif args.mode == 'semantic':
            with open(args.features, 'r') as f:
                features = json.load(f)
            semantic_attention = np.array(features['semantic_attention'])
            visualizer.visualize_semantic_attention(args.input, semantic_attention)
            
        elif args.mode == 'report':
            with open(args.input, 'r') as f:
                results = json.load(f)
            visualizer.create_feature_report(results)
            
    else:
        parser.print_help()


if __name__ == "__main__":
    main()