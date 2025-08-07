"""
语义引导的轻量级特征检测器 - RESTful API
提供特征检测、匹配和模型管理的HTTP接口
"""

import os
import sys
import json
import time
import base64
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from PIL import Image
import cv2
from datetime import datetime

# 导入项目模块
from semantic_guided_xfeat_implementation import SemanticGuidedXFeat, create_semantic_guided_xfeat
from tools import BatchInference, FeatureVisualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="Semantic-Guided XFeat API",
    description="RESTful API for semantic-guided lightweight feature detection and matching",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
model = None
model_config = None
batch_inference = None
visualizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据模型
class FeatureDetectionRequest(BaseModel):
    """特征检测请求"""
    image_base64: str
    top_k: int = 4096
    threshold: float = 0.01
    return_visualization: bool = False

class FeatureMatchingRequest(BaseModel):
    """特征匹配请求"""
    image1_base64: str
    image2_base64: str
    top_k: int = 4096
    matching_threshold: float = 0.8
    return_visualization: bool = False

class BatchDetectionRequest(BaseModel):
    """批量特征检测请求"""
    images_base64: List[str]
    top_k: int = 4096
    threshold: float = 0.01

class ModelConfig(BaseModel):
    """模型配置"""
    input_channels: int = 1
    feature_channels: int = 64
    semantic_classes: int = 20
    use_attention: bool = True
    dropout_rate: float = 0.1

class TrainingConfig(BaseModel):
    """训练配置"""
    learning_rate: float = 0.001
    batch_size: int = 16
    num_epochs: int = 100
    save_interval: int = 10

class APIResponse(BaseModel):
    """API响应"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str
    processing_time: float

# 辅助函数
def base64_to_image(base64_string: str) -> np.ndarray:
    """将base64字符串转换为图像数组"""
    try:
        # 解码base64
        image_data = base64.b64decode(base64_string)
        
        # 转换为PIL图像
        image = Image.open(io.BytesIO(image_data))
        
        # 转换为numpy数组
        image_array = np.array(image)
        
        # 转换为BGR格式（OpenCV格式）
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
        return image_array
    except Exception as e:
        logger.error(f"Error converting base64 to image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")

def image_to_base64(image: np.ndarray) -> str:
    """将图像数组转换为base64字符串"""
    try:
        # 转换为RGB格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # 转换为PIL图像
        pil_image = Image.fromarray(image)
        
        # 保存到字节流
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        
        # 编码为base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return image_base64
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        raise HTTPException(status_code=500, detail="Image encoding failed")

def preprocess_image(image: np.ndarray, target_size: int = 256) -> torch.Tensor:
    """预处理图像"""
    try:
        # 转换为灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # 调整大小
        image = cv2.resize(image, (target_size, target_size))
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        # 转换为tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        
        return image_tensor.to(device)
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=500, detail="Image preprocessing failed")

def create_response(success: bool, message: str, data: Optional[Dict] = None) -> APIResponse:
    """创建API响应"""
    return APIResponse(
        success=success,
        message=message,
        data=data,
        timestamp=datetime.now().isoformat(),
        processing_time=0.0
    )

# 初始化函数
def initialize_model(model_path: str = None, config: ModelConfig = None):
    """初始化模型"""
    global model, model_config, batch_inference, visualizer
    
    try:
        if model_path and os.path.exists(model_path):
            # 加载预训练模型
            logger.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            
            # 获取模型配置
            if 'config' in checkpoint:
                model_config = checkpoint['config']
            else:
                model_config = config.dict() if config else ModelConfig().dict()
                
            # 创建模型
            model = create_semantic_guided_xfeat(**model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else:
            # 创建新模型
            logger.info("Creating new model")
            model_config = config.dict() if config else ModelConfig().dict()
            model = create_semantic_guided_xfeat(**model_config)
            
        # 设置模型为评估模式
        model.eval()
        model.to(device)
        
        # 创建推理和可视化工具
        batch_inference = BatchInference(model_path if model_path else "temp_model.pth", device=str(device))
        visualizer = FeatureVisualizer("./api_visualizations")
        
        logger.info("Model initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return False

# API端点
@app.get("/")
async def root():
    """根端点"""
    return create_response(
        success=True,
        message="Semantic-Guided XFeat API is running",
        data={
            "version": "1.0.0",
            "device": str(device),
            "model_loaded": model is not None
        }
    )

@app.post("/initialize")
async def initialize_endpoint(config: ModelConfig):
    """初始化模型端点"""
    start_time = time.time()
    
    success = initialize_model(config=config)
    
    processing_time = time.time() - start_time
    
    if success:
        return create_response(
            success=True,
            message="Model initialized successfully",
            data={"config": model_config}
        )
    else:
        raise HTTPException(status_code=500, detail="Model initialization failed")

@app.post("/load_model")
async def load_model_endpoint(model_path: str):
    """加载模型端点"""
    start_time = time.time()
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
        
    success = initialize_model(model_path=model_path)
    
    processing_time = time.time() - start_time
    
    if success:
        return create_response(
            success=True,
            message="Model loaded successfully",
            data={"model_path": model_path, "config": model_config}
        )
    else:
        raise HTTPException(status_code=500, detail="Model loading failed")

@app.post("/detect_features")
async def detect_features(request: FeatureDetectionRequest):
    """特征检测端点"""
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
        
    try:
        # 转换图像
        image = base64_to_image(request.image_base64)
        
        # 预处理
        image_tensor = preprocess_image(image)
        
        # 特征检测
        with torch.no_grad():
            features = model.detectAndCompute(image_tensor, top_k=request.top_k)
            
        # 提取结果
        result = features[0]
        keypoints = result['keypoints'].cpu().numpy()
        scores = result['scores'].cpu().numpy()
        descriptors = result['descriptors'].cpu().numpy()
        semantic_attention = result['semantic_attention'].cpu().numpy()
        
        # 准备响应数据
        response_data = {
            "num_keypoints": len(keypoints),
            "keypoints": keypoints.tolist(),
            "scores": scores.tolist(),
            "descriptors": descriptors.tolist(),
            "semantic_attention": semantic_attention.tolist()
        }
        
        # 创建可视化
        if request.return_visualization:
            # 保存临时图像文件
            temp_image_path = "temp_image.jpg"
            cv2.imwrite(temp_image_path, image)
            
            # 创建可视化
            vis_features = {
                'keypoints': keypoints,
                'scores': scores,
                'descriptors': descriptors,
                'semantic_attention': semantic_attention
            }
            
            keypoints_vis_path = visualizer.visualize_keypoints(
                temp_image_path, vis_features, save_name="api_keypoints"
            )
            semantic_vis_path = visualizer.visualize_semantic_attention(
                temp_image_path, semantic_attention, save_name="api_semantic"
            )
            
            # 读取可视化图像并转换为base64
            keypoints_image = cv2.imread(keypoints_vis_path)
            semantic_image = cv2.imread(semantic_vis_path)
            
            response_data["keypoints_visualization"] = image_to_base64(keypoints_image)
            response_data["semantic_visualization"] = image_to_base64(semantic_image)
            
            # 清理临时文件
            os.remove(temp_image_path)
            
        processing_time = time.time() - start_time
        
        return create_response(
            success=True,
            message="Feature detection completed",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"Error in feature detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/match_features")
async def match_features(request: FeatureMatchingRequest):
    """特征匹配端点"""
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
        
    try:
        # 转换图像
        image1 = base64_to_image(request.image1_base64)
        image2 = base64_to_image(request.image2_base64)
        
        # 预处理
        image1_tensor = preprocess_image(image1)
        image2_tensor = preprocess_image(image2)
        
        # 特征检测
        with torch.no_grad():
            features1 = model.detectAndCompute(image1_tensor, top_k=request.top_k)
            features2 = model.detectAndCompute(image2_tensor, top_k=request.top_k)
            
        # 提取结果
        result1 = features1[0]
        result2 = features2[0]
        
        keypoints1 = result1['keypoints'].cpu().numpy()
        descriptors1 = result1['descriptors'].cpu().numpy()
        keypoints2 = result2['keypoints'].cpu().numpy()
        descriptors2 = result2['descriptors'].cpu().numpy()
        
        # 特征匹配
        desc1 = torch.from_numpy(descriptors1)
        desc2 = torch.from_numpy(descriptors2)
        
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
            
            if ratio < request.matching_threshold:
                matches.append([i, min_idx.item()])
                
        # 准备响应数据
        response_data = {
            "num_matches": len(matches),
            "matches": matches,
            "image1_keypoints": keypoints1.tolist(),
            "image2_keypoints": keypoints2.tolist(),
            "image1_num_keypoints": len(keypoints1),
            "image2_num_keypoints": len(keypoints2)
        }
        
        # 创建可视化
        if request.return_visualization:
            # 保存临时图像文件
            temp_image1_path = "temp_image1.jpg"
            temp_image2_path = "temp_image2.jpg"
            cv2.imwrite(temp_image1_path, image1)
            cv2.imwrite(temp_image2_path, image2)
            
            # 创建特征数据
            features1_data = {
                'keypoints': keypoints1,
                'scores': result1['scores'].cpu().numpy(),
                'descriptors': descriptors1,
                'semantic_attention': result1['semantic_attention'].cpu().numpy()
            }
            features2_data = {
                'keypoints': keypoints2,
                'scores': result2['scores'].cpu().numpy(),
                'descriptors': descriptors2,
                'semantic_attention': result2['semantic_attention'].cpu().numpy()
            }
            
            # 创建匹配可视化
            matches_vis_path = visualizer.visualize_matches(
                temp_image1_path, temp_image2_path, matches,
                features1_data, features2_data, save_name="api_matches"
            )
            
            # 读取可视化图像并转换为base64
            matches_image = cv2.imread(matches_vis_path)
            response_data["matches_visualization"] = image_to_base64(matches_image)
            
            # 清理临时文件
            os.remove(temp_image1_path)
            os.remove(temp_image2_path)
            
        processing_time = time.time() - start_time
        
        return create_response(
            success=True,
            message="Feature matching completed",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"Error in feature matching: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_detect")
async def batch_detect(request: BatchDetectionRequest):
    """批量特征检测端点"""
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
        
    try:
        results = []
        
        for i, image_base64 in enumerate(request.images_base64):
            # 转换图像
            image = base64_to_image(image_base64)
            
            # 预处理
            image_tensor = preprocess_image(image)
            
            # 特征检测
            with torch.no_grad():
                features = model.detectAndCompute(image_tensor, top_k=request.top_k)
                
            # 提取结果
            result = features[0]
            keypoints = result['keypoints'].cpu().numpy()
            scores = result['scores'].cpu().numpy()
            descriptors = result['descriptors'].cpu().numpy()
            semantic_attention = result['semantic_attention'].cpu().numpy()
            
            # 添加到结果列表
            results.append({
                "image_index": i,
                "num_keypoints": len(keypoints),
                "keypoints": keypoints.tolist(),
                "scores": scores.tolist(),
                "descriptors": descriptors.tolist(),
                "semantic_attention": semantic_attention.tolist()
            })
            
        processing_time = time.time() - start_time
        
        return create_response(
            success=True,
            message="Batch feature detection completed",
            data={
                "num_images": len(results),
                "results": results
            }
        )
        
    except Exception as e:
        logger.error(f"Error in batch detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info")
async def model_info():
    """获取模型信息"""
    if model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
        
    try:
        # 计算模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 估算模型大小
        model_size_mb = total_params * 4 / 1024 / 1024  # 假设float32
        
        response_data = {
            "model_config": model_config,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
            "device": str(device)
        }
        
        return create_response(
            success=True,
            message="Model information retrieved",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return create_response(
        success=True,
        message="API is healthy",
        data={
            "model_loaded": model is not None,
            "device": str(device),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.post("/save_model")
async def save_model(save_path: str):
    """保存模型端点"""
    if model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
        
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': model_config,
            'timestamp': datetime.now().isoformat()
        }, save_path)
        
        return create_response(
            success=True,
            message="Model saved successfully",
            data={"save_path": save_path}
        )
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 启动函数
def start_api(host: str = "0.0.0.0", port: int = 8000, model_path: str = None):
    """启动API服务"""
    logger.info(f"Starting Semantic-Guided XFeat API on {host}:{port}")
    
    # 初始化模型
    if model_path:
        initialize_model(model_path=model_path)
    
    # 启动服务器
    uvicorn.run(app, host=host, port=port)

# 主函数
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Semantic-Guided XFeat API")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--model", help="Path to pre-trained model")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    start_api(host=args.host, port=args.port, model_path=args.model)