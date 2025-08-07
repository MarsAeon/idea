"""
API使用示例
演示如何使用语义引导的轻量级特征检测器的RESTful API
"""

import requests
import json
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import threading
import subprocess
import sys
import os

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入项目模块
from basic_feature_detection import load_image


class APIClient:
    """
    API客户端类
    """
    
    def __init__(self, base_url="http://localhost:8000"):
        ""
        初始化API客户端
        
        Args:
            base_url: API服务器基础URL
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def health_check(self):
        ""
        健康检查
        
        Returns:
            健康状态信息
        ""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return None
            
    def get_model_info(self):
        ""
        获取模型信息
        
        Returns:
            模型信息字典
        ""
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to get model info: {e}")
            return None
            
    def detect_features(self, image_path, params=None):
        ""
        检测特征
        
        Args:
            image_path: 图像文件路径
            params: 检测参数
            
        Returns:
            特征检测结果
        ""
        try:
            # 读取并编码图像
            with open(image_path, 'rb') as f:
                image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                
            # 准备请求数据
            request_data = {
                "image": image_base64,
                "params": params or {}
            }
            
            # 发送请求
            response = self.session.post(
                f"{self.base_url}/detect",
                json=request_data,
                timeout=30
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Feature detection failed: {e}")
            return None
            
    def match_features(self, image1_path, image2_path, params=None):
        ""
        匹配特征
        
        Args:
            image1_path: 第一张图像路径
            image2_path: 第二张图像路径
            params: 匹配参数
            
        Returns:
            特征匹配结果
        ""
        try:
            # 读取并编码图像
            with open(image1_path, 'rb') as f:
                image1_data = f.read()
                image1_base64 = base64.b64encode(image1_data).decode('utf-8')
                
            with open(image2_path, 'rb') as f:
                image2_data = f.read()
                image2_base64 = base64.b64encode(image2_data).decode('utf-8')
                
            # 准备请求数据
            request_data = {
                "image1": image1_base64,
                "image2": image2_base64,
                "params": params or {}
            }
            
            # 发送请求
            response = self.session.post(
                f"{self.base_url}/match",
                json=request_data,
                timeout=60
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Feature matching failed: {e}")
            return None
            
    def batch_detect(self, image_paths, params=None):
        ""
        批量检测特征
        
        Args:
            image_paths: 图像路径列表
            params: 检测参数
            
        Returns:
            批量检测结果
        ""
        try:
            # 读取并编码所有图像
            images_base64 = []
            for image_path in image_paths:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    images_base64.append(image_base64)
                    
            # 准备请求数据
            request_data = {
                "images": images_base64,
                "params": params or {}
            }
            
            # 发送请求
            response = self.session.post(
                f"{self.base_url}/batch-detect",
                json=request_data,
                timeout=120
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Batch detection failed: {e}")
            return None
            
    def load_model(self, model_path):
        ""
        加载模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            加载结果
        ""
        try:
            # 读取并编码模型文件
            with open(model_path, 'rb') as f:
                model_data = f.read()
                model_base64 = base64.b64encode(model_data).decode('utf-8')
                
            # 准备请求数据
            request_data = {
                "model_data": model_base64,
                "model_name": Path(model_path).stem
            }
            
            # 发送请求
            response = self.session.post(
                f"{self.base_url}/model/load",
                json=request_data,
                timeout=60
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Model loading failed: {e}")
            return None
            
    def save_model(self, save_path):
        ""
        保存模型
        
        Args:
            save_path: 保存路径
            
        Returns:
            保存结果
        ""
        try:
            request_data = {
                "save_path": save_path
            }
            
            # 发送请求
            response = self.session.post(
                f"{self.base_url}/model/save",
                json=request_data,
                timeout=30
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Model saving failed: {e}")
            return None


def create_test_images(output_dir, num_images=3):
    """
    创建测试图像
    
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
        # 创建测试图像
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # 添加不同的几何形状
        if i == 0:
            # 圆形
            cv2.circle(image, (128, 128), 60, (255, 0, 0), -1)
        elif i == 1:
            # 矩形
            cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), -1)
        else:
            # 椭圆
            cv2.ellipse(image, (128, 128), (80, 50), 0, 0, 360, (0, 0, 255), -1)
            
        # 添加噪声
        noise = np.random.normal(0, 10, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 保存图像
        image_path = output_dir / f'test_image_{i}.jpg'
        cv2.imwrite(str(image_path), image)
        image_paths.append(str(image_path))
        
    return image_paths


def start_api_server():
    """
    启动API服务器
    
    Returns:
        服务器进程
    """
    try:
        # 启动API服务器
        cmd = [sys.executable, "app.py"]
        process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 等待服务器启动
        time.sleep(5)
        
        return process
        
    except Exception as e:
        print(f"Failed to start API server: {e}")
        return None


def stop_api_server(process):
    """
    停止API服务器
    
    Args:
        process: 服务器进程
    """
    if process:
        process.terminate()
        process.wait()


def test_api_functionality():
    """
    测试API功能
    
    Returns:
        测试结果字典
    """
    results = {}
    
    # 创建API客户端
    client = APIClient()
    
    # 1. 健康检查
    print("1. 健康检查...")
    health = client.health_check()
    if health:
        print(f"   状态: {health.get('status', 'unknown')}")
        print(f"   消息: {health.get('message', 'no message')}")
        results['health_check'] = True
    else:
        print("   健康检查失败")
        results['health_check'] = False
        
    # 2. 获取模型信息
    print("\n2. 获取模型信息...")
    model_info = client.get_model_info()
    if model_info:
        print(f"   模型名称: {model_info.get('model_name', 'unknown')}")
        print(f"   参数数量: {model_info.get('param_count', 'unknown')}")
        print(f"   输入尺寸: {model_info.get('input_size', 'unknown')}")
        results['model_info'] = True
    else:
        print("   获取模型信息失败")
        results['model_info'] = False
        
    # 3. 创建测试图像
    print("\n3. 创建测试图像...")
    test_dir = Path("./examples/test_images_api")
    image_paths = create_test_images(test_dir, 3)
    print(f"   创建了 {len(image_paths)} 张测试图像")
    
    if len(image_paths) < 2:
        print("   测试图像创建失败")
        return results
        
    # 4. 特征检测
    print("\n4. 测试特征检测...")
    detect_params = {
        "top_k": 500,
        "threshold": 0.01
    }
    
    start_time = time.time()
    detect_result = client.detect_features(image_paths[0], detect_params)
    detect_time = time.time() - start_time
    
    if detect_result:
        num_keypoints = detect_result.get('num_keypoints', 0)
        print(f"   检测到 {num_keypoints} 个关键点")
        print(f"   检测时间: {detect_time:.3f} 秒")
        results['feature_detection'] = True
        results['detection_time'] = detect_time
    else:
        print("   特征检测失败")
        results['feature_detection'] = False
        
    # 5. 特征匹配
    print("\n5. 测试特征匹配...")
    match_params = {
        "ratio_threshold": 0.8,
        "semantic_weight": 0.3
    }
    
    start_time = time.time()
    match_result = client.match_features(image_paths[0], image_paths[1], match_params)
    match_time = time.time() - start_time
    
    if match_result:
        num_matches = match_result.get('num_matches', 0)
        print(f"   找到 {num_matches} 个匹配")
        print(f"   匹配时间: {match_time:.3f} 秒")
        results['feature_matching'] = True
        results['matching_time'] = match_time
    else:
        print("   特征匹配失败")
        results['feature_matching'] = False
        
    # 6. 批量检测
    print("\n6. 测试批量检测...")
    start_time = time.time()
    batch_result = client.batch_detect(image_paths, detect_params)
    batch_time = time.time() - start_time
    
    if batch_result:
        batch_results = batch_result.get('results', [])
        print(f"   批量检测了 {len(batch_results)} 张图像")
        print(f"   批量检测时间: {batch_time:.3f} 秒")
        
        # 计算平均每张图像的处理时间
        if len(batch_results) > 0:
            avg_time_per_image = batch_time / len(batch_results)
            print(f"   平均每张图像: {avg_time_per_image:.3f} 秒")
            
        results['batch_detection'] = True
        results['batch_time'] = batch_time
    else:
        print("   批量检测失败")
        results['batch_detection'] = False
        
    # 7. 模型保存
    print("\n7. 测试模型保存...")
    save_path = "./examples/output/api_test_model.pth"
    save_result = client.save_model(save_path)
    
    if save_result and save_result.get('success'):
        print(f"   模型保存成功: {save_path}")
        results['model_save'] = True
    else:
        print("   模型保存失败")
        results['model_save'] = False
        
    return results


def create_performance_report(results, output_path):
    """
    创建性能报告
    
    Args:
        results: 测试结果字典
        output_path: 输出路径
    """
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_tests": results,
        "performance_summary": {}
    }
    
    # 计算性能摘要
    if results.get('feature_detection'):
        report['performance_summary']['detection_time'] = results.get('detection_time', 0)
        
    if results.get('feature_matching'):
        report['performance_summary']['matching_time'] = results.get('matching_time', 0)
        
    if results.get('batch_detection'):
        report['performance_summary']['batch_time'] = results.get('batch_time', 0)
        
    # 计算成功率
    total_tests = len([k for k in results.keys() if k.endswith('_check') or k.endswith('_detection') or k.endswith('_matching') or k.endswith('_save')])
    passed_tests = len([k for k in results.keys() if results.get(k, False)])
    
    report['performance_summary']['success_rate'] = passed_tests / total_tests if total_tests > 0 else 0
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    return report


def visualize_api_results(results, output_dir):
    """
    可视化API测试结果
    
    Args:
        results: 测试结果字典
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备数据
    test_names = []
    test_results = []
    test_times = []
    
    for key, value in results.items():
        if key.endswith('_check') or key.endswith('_detection') or key.endswith('_matching') or key.endswith('_save'):
            test_names.append(key.replace('_', ' ').title())
            test_results.append(1 if value else 0)
            
        if key.endswith('_time'):
            test_names.append(key.replace('_', ' ').title())
            test_times.append(value)
            
    # 创建可视化图表
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 测试结果饼图
    if test_results:
        passed = sum(test_results)
        failed = len(test_results) - passed
        
        axes[0].pie([passed, failed], labels=['Passed', 'Failed'], autopct='%1.1f%%', colors=['green', 'red'])
        axes[0].set_title('API Test Results')
        
    # 性能时间条形图
    if test_times:
        time_labels = [name.replace(' Time', '') for name in test_names[-len(test_times):]]
        axes[1].bar(time_labels, test_times, color=['blue', 'orange', 'green'])
        axes[1].set_title('API Performance (seconds)')
        axes[1].set_ylabel('Time (s)')
        axes[1].tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.savefig(output_dir / 'api_test_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"API测试结果可视化保存到: {output_dir / 'api_test_results.png'}")


def main():
    """
    主函数
    """
    print("语义引导的轻量级特征检测器 - API使用示例")
    print("="*60)
    
    # 配置参数
    config = {
        'output_dir': './examples/output',
        'auto_start_server': True,
        'server_timeout': 30,
        'create_visualization': True
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 启动API服务器
    server_process = None
    if config['auto_start_server']:
        print("\n启动API服务器...")
        server_process = start_api_server()
        
        if server_process:
            print("API服务器启动成功")
        else:
            print("API服务器启动失败，请手动启动")
            print("运行命令: python app.py")
            return
            
    try:
        # 等待服务器完全启动
        print("\n等待服务器启动...")
        time.sleep(3)
        
        # 测试API功能
        print("\n开始API功能测试...")
        results = test_api_functionality()
        
        # 创建性能报告
        print("\n创建性能报告...")
        report_path = os.path.join(config['output_dir'], 'api_test_report.json')
        report = create_performance_report(results, report_path)
        
        print(f"性能报告保存到: {report_path}")
        
        # 显示测试摘要
        print("\n测试摘要:")
        print(f"- 健康检查: {'通过' if results.get('health_check') else '失败'}")
        print(f"- 模型信息: {'通过' if results.get('model_info') else '失败'}")
        print(f"- 特征检测: {'通过' if results.get('feature_detection') else '失败'}")
        print(f"- 特征匹配: {'通过' if results.get('feature_matching') else '失败'}")
        print(f"- 批量检测: {'通过' if results.get('batch_detection') else '失败'}")
        print(f"- 模型保存: {'通过' if results.get('model_save') else '失败'}")
        
        # 显示性能数据
        if results.get('detection_time'):
            print(f"- 检测时间: {results['detection_time']:.3f} 秒")
        if results.get('matching_time'):
            print(f"- 匹配时间: {results['matching_time']:.3f} 秒")
        if results.get('batch_time'):
            print(f"- 批量检测时间: {results['batch_time']:.3f} 秒")
            
        # 计算成功率
        total_tests = len([k for k in results.keys() if k.endswith('_check') or k.endswith('_detection') or k.endswith('_matching') or k.endswith('_save')])
        passed_tests = len([k for k in results.keys() if results.get(k, False)])
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        print(f"- 总体成功率: {success_rate * 100:.1f}% ({passed_tests}/{total_tests})")
        
        # 创建可视化
        if config['create_visualization']:
            print("\n创建可视化结果...")
            viz_dir = Path(config['output_dir']) / 'api_visualizations'
            visualize_api_results(results, viz_dir)
            
        print("\n" + "="*60)
        print("API使用示例完成！")
        print(f"结果保存在: {config['output_dir']}")
        print("="*60)
        
        return results
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
    finally:
        # 停止API服务器
        if server_process:
            print("\n停止API服务器...")
            stop_api_server(server_process)
            print("API服务器已停止")


if __name__ == "__main__":\    # 运行主函数
    results = main()
    
    # 可选：返回结果供其他示例使用
    if results:
        print("\nAPI测试成功完成！")
    else:
        print("\nAPI测试失败！")