# 语义引导的轻量级特征检测器使用示例

本目录包含项目的使用示例和演示代码，帮助用户快速上手使用语义引导的轻量级特征检测器。

## 示例列表

### 1. 基础特征检测示例

文件：`basic_feature_detection.py`

```python
import torch
import cv2
import numpy as np
from semantic_guided_xfeat_implementation import create_semantic_guided_xfeat

# 创建模型
model = create_semantic_guided_xfeat()
model.eval()

# 加载图像
image = cv2.imread('example_image.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 预处理
image_tensor = torch.from_numpy(image_gray).float() / 255.0
image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

# 特征检测
with torch.no_grad():
    features = model.detectAndCompute(image_tensor, top_k=1000)

# 提取结果
keypoints = features[0]['keypoints'].numpy()
scores = features[0]['scores'].numpy()
descriptors = features[0]['descriptors'].numpy()

print(f"检测到 {len(keypoints)} 个关键点")
```

### 2. 特征匹配示例

文件：`feature_matching_demo.py`

```python
import torch
import cv2
import numpy as np
from semantic_guided_xfeat_implementation import create_semantic_guided_xfeat

def match_features(image1_path, image2_path):
    # 创建模型
    model = create_semantic_guided_xfeat()
    model.eval()
    
    # 加载图像
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    # 预处理
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    image1_tensor = torch.from_numpy(image1_gray).float() / 255.0
    image1_tensor = image1_tensor.unsqueeze(0).unsqueeze(0)
    
    image2_tensor = torch.from_numpy(image2_gray).float() / 255.0
    image2_tensor = image2_tensor.unsqueeze(0).unsqueeze(0)
    
    # 特征检测
    with torch.no_grad():
        features1 = model.detectAndCompute(image1_tensor, top_k=2000)
        features2 = model.detectAndCompute(image2_tensor, top_k=2000)
    
    # 提取描述符
    desc1 = features1[0]['descriptors']
    desc2 = features2[0]['descriptors']
    
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
        
        if ratio < 0.8:
            matches.append([i, min_idx.item()])
    
    return matches, features1[0], features2[0]

# 使用示例
matches, features1, features2 = match_features('image1.jpg', 'image2.jpg')
print(f"找到 {len(matches)} 个匹配点")
```

### 3. 批量处理示例

文件：`batch_processing_demo.py`

```python
import os
import glob
from tools import BatchInference

def batch_feature_extraction(image_dir, output_dir, model_path):
    # 创建批量推理器
    inference = BatchInference(model_path)
    
    # 批量提取特征
    features = inference.extract_features_batch(image_dir, output_dir, top_k=1024)
    
    print(f"处理完成，共提取 {len(features)} 张图像的特征")
    return features

# 使用示例
image_dir = './test_images'
output_dir = './extracted_features'
model_path = './models/semantic_guided_xfeat.pth'

features = batch_feature_extraction(image_dir, output_dir, model_path)
```

### 4. 可视化示例

文件：`visualization_demo.py`

```python
import cv2
import json
from tools import FeatureVisualizer

def visualize_features(image_path, features_path, output_dir):
    # 创建可视化器
    visualizer = FeatureVisualizer(output_dir)
    
    # 加载特征数据
    with open(features_path, 'r') as f:
        features = json.load(f)
    
    # 可视化关键点
    keypoints_path = visualizer.visualize_keypoints(image_path, features)
    print(f"关键点可视化保存到: {keypoints_path}")
    
    # 可视化语义注意力
    semantic_attention = np.array(features['semantic_attention'])
    semantic_path = visualizer.visualize_semantic_attention(image_path, semantic_attention)
    print(f"语义注意力可视化保存到: {semantic_path}")

# 使用示例
image_path = './test_images/example.jpg'
features_path = './extracted_features/example_features.json'
output_dir = './visualizations'

visualize_features(image_path, features_path, output_dir)
```

### 5. API使用示例

文件：`api_client_demo.py`

```python
import requests
import base64
import json
from PIL import Image
import io

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def detect_features_via_api(image_path, api_url="http://localhost:8000"):
    # 转换图像为base64
    image_base64 = image_to_base64(image_path)
    
    # 发送请求
    response = requests.post(
        f"{api_url}/detect_features",
        json={
            "image_base64": image_base64,
            "top_k": 1000,
            "return_visualization": True
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        return result['data']
    else:
        print(f"API调用失败: {response.status_code}")
        return None

# 使用示例
result = detect_features_via_api('./test_images/example.jpg')
if result:
    print(f"检测到 {result['num_keypoints']} 个关键点")
    
    # 保存可视化结果
    if 'keypoints_visualization' in result:
        with open('./keypoints_visualization.jpg', 'wb') as f:
            f.write(base64.b64decode(result['keypoints_visualization']))
```

### 6. 训练示例

文件：`training_demo.py`

```python
import torch
from train_semantic_xfeat import SemanticXFeatTrainer
from dataset_processing import DataModule

def train_model(dataset_file, config_file, output_dir):
    # 创建数据模块
    data_module = DataModule(
        dataset_file=dataset_file,
        batch_size=16,
        num_workers=4
    )
    
    # 创建训练器
    trainer = SemanticXFeatTrainer(
        config_file=config_file,
        output_dir=output_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 开始训练
    trainer.train()
    
    print("训练完成！")

# 使用示例
dataset_file = './data/dataset.json'
config_file = './config_semantic_xfeat.json'
output_dir = './training_output'

train_model(dataset_file, config_file, output_dir)
```

### 7. 评估示例

文件：`evaluation_demo.py`

```python
from evaluate_semantic_xfeat import BenchmarkSuite

def evaluate_model(model_path, test_data_dir, output_dir):
    # 创建基准测试套件
    benchmark = BenchmarkSuite(
        model_path=model_path,
        test_data_dir=test_data_dir,
        output_dir=output_dir
    )
    
    # 运行完整评估
    results = benchmark.run_full_benchmark()
    
    # 生成报告
    benchmark.generate_report(results)
    
    print("评估完成！")
    return results

# 使用示例
model_path = './models/semantic_guided_xfeat.pth'
test_data_dir = './test_data'
output_dir = './evaluation_results'

results = evaluate_model(model_path, test_data_dir, output_dir)
print(f"模型大小: {results['model_size']['model_size_mb']:.2f} MB")
print(f"推理时间: {results['runtime']['avg_time_ms']:.2f} ms")
print(f"匹配准确率: {results['matching']['precision']:.4f}")
```

### 8. 模型转换示例

文件：`model_conversion_demo.py`

```python
from tools import ModelConverter

def convert_model(model_path, output_dir, formats=['onnx', 'tensorrt']):
    # 创建模型转换器
    converter = ModelConverter(model_path, output_dir)
    
    # 转换为指定格式
    if 'onnx' in formats:
        onnx_path = converter.convert_to_onnx()
        print(f"ONNX模型保存到: {onnx_path}")
    
    if 'tensorrt' in formats:
        trt_path = converter.convert_to_tensorrt(fp16=True)
        if trt_path:
            print(f"TensorRT模型保存到: {trt_path}")

# 使用示例
model_path = './models/semantic_guided_xfeat.pth'
output_dir = './converted_models'

convert_model(model_path, output_dir, formats=['onnx', 'tensorrt'])
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载预训练模型

```bash
# 从模型仓库下载预训练权重
wget https://example.com/models/semantic_guided_xfeat.pth -O ./models/semantic_guided_xfeat.pth
```

### 3. 运行基础示例

```bash
# 基础特征检测
python examples/basic_feature_detection.py

# 特征匹配
python examples/feature_matching_demo.py

# 启动API服务
python app.py --model ./models/semantic_guided_xfeat.pth
```

### 4. 测试API

```bash
# 发送特征检测请求
curl -X POST "http://localhost:8000/detect_features" \
     -H "Content-Type: application/json" \
     -d '{
           "image_base64": "'$BASE64_ENCODED_IMAGE'",
           "top_k": 1000
         }'
```

## 预期输出

### 特征检测输出
```json
{
  "success": true,
  "message": "Feature detection completed",
  "data": {
    "num_keypoints": 1024,
    "keypoints": [[x1, y1], [x2, y2], ...],
    "scores": [score1, score2, ...],
    "descriptors": [[d1, d2, ...], [d1, d2, ...], ...],
    "semantic_attention": [[[a1, a2, ...], ...], ...]
  },
  "timestamp": "2023-12-01T10:00:00",
  "processing_time": 0.045
}
```

### 特征匹配输出
```json
{
  "success": true,
  "message": "Feature matching completed",
  "data": {
    "num_matches": 256,
    "matches": [[idx1, idx2], [idx3, idx4], ...],
    "image1_keypoints": [[x1, y1], [x2, y2], ...],
    "image2_keypoints": [[x1, y1], [x2, y2], ...],
    "image1_num_keypoints": 1024,
    "image2_num_keypoints": 1024
  },
  "timestamp": "2023-12-01T10:00:00",
  "processing_time": 0.089
}
```

## 常见问题

### Q: 如何处理大尺寸图像？
A: 模型会自动将图像调整为256x256进行特征提取，关键点坐标会根据原始图像尺寸进行缩放。

### Q: 如何提高特征检测速度？
A: 可以使用模型量化功能，或者减少top_k参数值。

### Q: 如何自定义语义类别？
A: 在模型配置中修改semantic_classes参数，并提供相应的语义标签数据。

### Q: 如何部署到生产环境？
A: 使用Docker容器化部署，参考Dockerfile和docker-compose.yml配置文件。

## 性能优化建议

1. **使用GPU加速**：确保安装CUDA版本的PyTorch并在GPU上运行
2. **模型量化**：使用int8量化可以显著提高推理速度
3. **批量处理**：对于大量图像，使用批量处理API
4. **缓存机制**：对于重复图像，可以实现特征缓存
5. **异步处理**：对于耗时操作，使用异步API调用

## 扩展功能

### 自定义数据集
```python
from dataset_processing import SemanticFeatureDataset

class CustomDataset(SemanticFeatureDataset):
    def __init__(self, custom_data_dir, **kwargs):
        # 自定义数据集初始化
        pass
    
    def __getitem__(self, idx):
        # 自定义数据加载逻辑
        pass
```

### 自定义损失函数
```python
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # 自定义损失计算逻辑
        pass
```

### 自定义后处理
```python
def custom_post_processing(features, threshold=0.5):
    # 自定义后处理逻辑
    filtered_keypoints = []
    filtered_scores = []
    
    for kp, score in zip(features['keypoints'], features['scores']):
        if score > threshold:
            filtered_keypoints.append(kp)
            filtered_scores.append(score)
    
    return {
        'keypoints': filtered_keypoints,
        'scores': filtered_scores,
        'descriptors': features['descriptors']
    }
```

## 贡献指南

1. Fork项目仓库
2. 创建功能分支
3. 提交代码更改
4. 创建Pull Request
5. 等待代码审查

## 许可证

本项目采用MIT许可证，详见LICENSE文件。