"""
部署示例
演示如何使用Docker容器来部署语义引导的轻量级特征检测器
"""

import os
import sys
import json
import time
import requests
import subprocess
from pathlib import Path
import docker
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入项目模块
from semantic_guided_xfeat_implementation import SemanticGuidedXFeat
from tools import ModelConverter


class DeploymentExample:
    """
    部署示例类
    """
    
    def __init__(self, config_path=None):
        """
        初始化部署示例
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        
        # 加载配置
        self.config = self._load_config()
        
        # 初始化Docker客户端
        try:
            self.docker_client = docker.from_env()
            print("Docker客户端初始化成功")
        except Exception as e:
            print(f"Docker客户端初始化失败: {e}")
            self.docker_client = None
            
        # 设置输出目录
        self.output_dir = Path("./examples/output/deployment")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self):
        """
        加载配置
        
        Returns:
            配置字典
        """
        if self.config_path is None:
            # 使用默认配置
            config = {
                "docker": {
                    "image_name": "semantic-xfeat",
                    "image_tag": "latest",
                    "container_name": "semantic-xfeat-container",
                    "host_port": 8000,
                    "container_port": 8000,
                    "gpu_enabled": True,
                    "memory_limit": "4g",
                    "cpu_limit": 2.0
                },
                "api": {
                    "base_url": "http://localhost:8000",
                    "endpoints": {
                        "health": "/health",
                        "detect": "/detect",
                        "match": "/match",
                        "batch_detect": "/batch-detect",
                        "model_info": "/model/info"
                    },
                    "timeout": 30
                },
                "deployment": {
                    "build_image": True,
                    "run_container": True,
                    "test_api": True,
                    "performance_test": True,
                    "cleanup": False
                },
                "performance": {
                    "num_requests": 100,
                    "concurrent_requests": 10,
                    "test_image_size": [256, 256]
                }
            }
        else:
            # 从文件加载配置
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
        return config
        
    def build_docker_image(self):
        """
        构建Docker镜像
        
        Returns:
            构建结果
        """
        print("\n构建Docker镜像...")
        
        if self.docker_client is None:
            return {
                'success': False,
                'error': 'Docker客户端未初始化'
            }
            
        docker_config = self.config['docker']
        image_name = docker_config['image_name']
        image_tag = docker_config['image_tag']
        full_image_name = f"{image_name}:{image_tag}"
        
        try:
            # 获取Dockerfile路径
            dockerfile_path = project_root / "Dockerfile"
            
            if not dockerfile_path.exists():
                return {
                    'success': False,
                    'error': f'Dockerfile不存在: {dockerfile_path}'
                }
                
            # 构建镜像
            print(f"开始构建镜像: {full_image_name}")
            start_time = time.time()
            
            # 构建参数
            build_args = {
                'PYTHON_VERSION': '3.8',
                'CUDA_VERSION': '11.3' if docker_config['gpu_enabled'] else 'cpu'
            }
            
            # 构建镜像
            image, build_logs = self.docker_client.images.build(
                path=str(project_root),
                dockerfile=str(dockerfile_path.relative_to(project_root)),
                tag=full_image_name,
                buildargs=build_args,
                rm=True
            )
            
            build_time = time.time() - start_time
            
            # 输出构建日志
            print("构建日志:")
            for log in build_logs:
                if 'stream' in log:
                    print(log['stream'].strip())
                    
            print(f"\n镜像构建成功！")
            print(f"镜像ID: {image.id}")
            print(f"镜像名称: {full_image_name}")
            print(f"构建时间: {build_time:.2f} 秒")
            print(f"镜像大小: {image.attrs['Size'] / (1024*1024):.2f} MB")
            
            return {
                'success': True,
                'image_id': image.id,
                'image_name': full_image_name,
                'build_time': build_time,
                'image_size_mb': image.attrs['Size'] / (1024*1024)
            }
            
        except Exception as e:
            print(f"镜像构建失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def run_docker_container(self, image_name=None):
        """
        运行Docker容器
        
        Args:
            image_name: 镜像名称
            
        Returns:
            容器运行结果
        """
        print("\n运行Docker容器...")
        
        if self.docker_client is None:
            return {
                'success': False,
                'error': 'Docker客户端未初始化'
            }
            
        docker_config = self.config['docker']
        
        if image_name is None:
            image_name = f"{docker_config['image_name']}:{docker_config['image_tag']}"
            
        container_name = docker_config['container_name']
        host_port = docker_config['host_port']
        container_port = docker_config['container_port']
        
        try:
            # 检查容器是否已存在
            existing_container = None
            try:
                existing_container = self.docker_client.containers.get(container_name)
                if existing_container.status == 'running':
                    print(f"容器 {container_name} 已在运行")
                    return {
                        'success': True,
                        'container_id': existing_container.id,
                        'container_name': container_name,
                        'status': 'already_running'
                    }
                else:
                    # 删除已停止的容器
                    existing_container.remove()
                    print(f"删除已停止的容器: {container_name}")
            except docker.errors.NotFound:
                pass
                
            # 运行容器
            print(f"启动容器: {container_name}")
            start_time = time.time()
            
            # 容器运行参数
            run_params = {
                'image': image_name,
                'name': container_name,
                'ports': {f'{container_port}/tcp': host_port},
                'detach': True,
                'restart_policy': {'Name': 'unless-stopped'}
            }
            
            # GPU支持
            if docker_config['gpu_enabled']:
                run_params['device_requests'] = [
                    docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
                ]
                
            # 资源限制
            if docker_config.get('memory_limit'):
                run_params['mem_limit'] = docker_config['memory_limit']
                
            if docker_config.get('cpu_limit'):
                run_params['nano_cpus'] = int(docker_config['cpu_limit'] * 1e9)
                
            # 运行容器
            container = self.docker_client.containers.run(**run_params)
            
            # 等待容器启动
            time.sleep(5)
            
            # 检查容器状态
            container.reload()
            if container.status != 'running':
                logs = container.logs().decode('utf-8')
                return {
                    'success': False,
                    'error': f'容器启动失败，状态: {container.status}\n日志: {logs}'
                }
                
            startup_time = time.time() - start_time
            
            print(f"容器启动成功！")
            print(f"容器ID: {container.id}")
            print(f"容器名称: {container_name}")
            print(f"启动时间: {startup_time:.2f} 秒")
            print(f"端口映射: {host_port} -> {container_port}")
            
            return {
                'success': True,
                'container_id': container.id,
                'container_name': container_name,
                'startup_time': startup_time,
                'status': 'running'
            }
            
        except Exception as e:
            print(f"容器运行失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def test_api_endpoints(self):
        """
        测试API端点
        
        Returns:
            测试结果
        """
        print("\n测试API端点...")
        
        api_config = self.config['api']
        base_url = api_config['base_url']
        timeout = api_config['timeout']
        
        test_results = {}
        
        # 创建测试图像
        test_image = self._create_test_image()
        
        # 测试健康检查端点
        try:
            print("测试健康检查端点...")
            response = requests.get(f"{base_url}{api_config['endpoints']['health']}", timeout=timeout)
            
            if response.status_code == 200:
                health_result = response.json()
                test_results['health'] = {
                    'success': True,
                    'status_code': response.status_code,
                    'response': health_result
                }
                print(f"健康检查成功: {health_result}")
            else:
                test_results['health'] = {
                    'success': False,
                    'status_code': response.status_code,
                    'error': f'HTTP {response.status_code}'
                }
                print(f"健康检查失败: HTTP {response.status_code}")
                
        except Exception as e:
            test_results['health'] = {
                'success': False,
                'error': str(e)
            }
            print(f"健康检查异常: {e}")
            
        # 测试模型信息端点
        try:
            print("测试模型信息端点...")
            response = requests.get(f"{base_url}{api_config['endpoints']['model_info']}", timeout=timeout)
            
            if response.status_code == 200:
                model_info = response.json()
                test_results['model_info'] = {
                    'success': True,
                    'status_code': response.status_code,
                    'response': model_info
                }
                print(f"模型信息获取成功")
            else:
                test_results['model_info'] = {
                    'success': False,
                    'status_code': response.status_code,
                    'error': f'HTTP {response.status_code}'
                }
                print(f"模型信息获取失败: HTTP {response.status_code}")
                
        except Exception as e:
            test_results['model_info'] = {
                'success': False,
                'error': str(e)
            }
            print(f"模型信息获取异常: {e}")
            
        # 测试特征检测端点
        try:
            print("测试特征检测端点...")
            
            # 准备请求数据
            _, buffer = cv2.imencode('.jpg', test_image)
            image_base64 = buffer.tobytes().decode('latin-1')
            
            data = {
                'image': image_base64,
                'num_keypoints': 500,
                'detection_threshold': 0.01
            }
            
            response = requests.post(
                f"{base_url}{api_config['endpoints']['detect']}",
                json=data,
                timeout=timeout
            )
            
            if response.status_code == 200:
                detection_result = response.json()
                test_results['detect'] = {
                    'success': True,
                    'status_code': response.status_code,
                    'response': detection_result
                }
                keypoints_count = len(detection_result.get('keypoints', []))
                print(f"特征检测成功，检测到 {keypoints_count} 个关键点")
            else:
                test_results['detect'] = {
                    'success': False,
                    'status_code': response.status_code,
                    'error': f'HTTP {response.status_code}'
                }
                print(f"特征检测失败: HTTP {response.status_code}")
                
        except Exception as e:
            test_results['detect'] = {
                'success': False,
                'error': str(e)
            }
            print(f"特征检测异常: {e}")
            
        # 测试特征匹配端点
        try:
            print("测试特征匹配端点...")
            
            # 准备两幅测试图像
            test_image1 = self._create_test_image()
            test_image2 = self._create_test_image()
            
            # 编码图像
            _, buffer1 = cv2.imencode('.jpg', test_image1)
            _, buffer2 = cv2.imencode('.jpg', test_image2)
            
            image1_base64 = buffer1.tobytes().decode('latin-1')
            image2_base64 = buffer2.tobytes().decode('latin-1')
            
            data = {
                'image1': image1_base64,
                'image2': image2_base64,
                'num_keypoints': 500,
                'matching_threshold': 0.8
            }
            
            response = requests.post(
                f"{base_url}{api_config['endpoints']['match']}",
                json=data,
                timeout=timeout
            )
            
            if response.status_code == 200:
                matching_result = response.json()
                test_results['match'] = {
                    'success': True,
                    'status_code': response.status_code,
                    'response': matching_result
                }
                matches_count = len(matching_result.get('matches', []))
                print(f"特征匹配成功，匹配到 {matches_count} 对特征点")
            else:
                test_results['match'] = {
                    'success': False,
                    'status_code': response.status_code,
                    'error': f'HTTP {response.status_code}'
                }
                print(f"特征匹配失败: HTTP {response.status_code}")
                
        except Exception as e:
            test_results['match'] = {
                'success': False,
                'error': str(e)
            }
            print(f"特征匹配异常: {e}")
            
        return test_results
        
    def _create_test_image(self, size=None):
        """
        创建测试图像
        
        Args:
            size: 图像大小
            
        Returns:
            测试图像
        """
        if size is None:
            size = self.config['performance']['test_image_size']
            
        # 创建带有简单几何形状的测试图像
        image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        
        # 添加一些几何形状
        cv2.circle(image, (size[1]//4, size[0]//4), 30, (255, 255, 255), -1)
        cv2.rectangle(image, (size[1]//2, size[0]//2), (3*size[1]//4, 3*size[0]//4), (255, 255, 255), -1)
        cv2.line(image, (0, size[0]//2), (size[1], size[0]//2), (255, 255, 255), 2)
        
        # 添加一些噪声
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
        
        return image
        
    def performance_test(self):
        """
        性能测试
        
        Returns:
            性能测试结果
        """
        print("\n进行性能测试...")
        
        api_config = self.config['api']
        performance_config = self.config['performance']
        base_url = api_config['base_url']
        timeout = api_config['timeout']
        
        num_requests = performance_config['num_requests']
        concurrent_requests = performance_config['concurrent_requests']
        
        # 创建测试图像
        test_image = self._create_test_image()
        _, buffer = cv2.imencode('.jpg', test_image)
        image_base64 = buffer.tobytes().decode('latin-1')
        
        # 准备请求数据
        data = {
            'image': image_base64,
            'num_keypoints': 500,
            'detection_threshold': 0.01
        }
        
        import concurrent.futures
        import threading
        
        # 性能测试结果
        results = {
            'total_requests': num_requests,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'errors': []
        }
        
        # 线程锁
        lock = threading.Lock()
        
        def make_request(request_id):
            """
            发送单个请求
            """
            try:
                start_time = time.time()
                response = requests.post(
                    f"{base_url}{api_config['endpoints']['detect']}",
                    json=data,
                    timeout=timeout
                )
                end_time = time.time()
                
                response_time = end_time - start_time
                
                with lock:
                    results['response_times'].append(response_time)
                    
                    if response.status_code == 200:
                        results['successful_requests'] += 1
                    else:
                        results['failed_requests'] += 1
                        results['errors'].append(f"Request {request_id}: HTTP {response.status_code}")
                        
            except Exception as e:
                with lock:
                    results['failed_requests'] += 1
                    results['errors'].append(f"Request {request_id}: {str(e)}")
                    
        # 并发发送请求
        print(f"发送 {num_requests} 个请求，并发数: {concurrent_requests}")
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            concurrent.futures.wait(futures)
            
        total_time = time.time() - start_time
        
        # 计算性能指标
        if results['successful_requests'] > 0:
            avg_response_time = np.mean(results['response_times'])
            std_response_time = np.std(results['response_times'])
            min_response_time = np.min(results['response_times'])
            max_response_time = np.max(results['response_times'])
            p95_response_time = np.percentile(results['response_times'], 95)
            p99_response_time = np.percentile(results['response_times'], 99)
            
            throughput = results['successful_requests'] / total_time
            
            results.update({
                'total_time': total_time,
                'avg_response_time': avg_response_time,
                'std_response_time': std_response_time,
                'min_response_time': min_response_time,
                'max_response_time': max_response_time,
                'p95_response_time': p95_response_time,
                'p99_response_time': p99_response_time,
                'throughput': throughput,
                'success_rate': results['successful_requests'] / num_requests
            })
            
            print(f"\n性能测试结果:")
            print(f"总请求数: {num_requests}")
            print(f"成功请求数: {results['successful_requests']}")
            print(f"失败请求数: {results['failed_requests']}")
            print(f"成功率: {results['success_rate']:.2%}")
            print(f"总耗时: {total_time:.2f} 秒")
            print(f"吞吐量: {throughput:.2f} 请求/秒")
            print(f"平均响应时间: {avg_response_time:.3f} 秒")
            print(f"响应时间标准差: {std_response_time:.3f} 秒")
            print(f"最小响应时间: {min_response_time:.3f} 秒")
            print(f"最大响应时间: {max_response_time:.3f} 秒")
            print(f"95%响应时间: {p95_response_time:.3f} 秒")
            print(f"99%响应时间: {p99_response_time:.3f} 秒")
            
        else:
            print("\n性能测试失败：没有成功的请求")
            results['error'] = 'No successful requests'
            
        return results
        
    def cleanup(self):
        """
        清理资源
        
        Returns:
            清理结果
        """
        print("\n清理资源...")
        
        if self.docker_client is None:
            return {
                'success': False,
                'error': 'Docker客户端未初始化'
            }
            
        docker_config = self.config['docker']
        container_name = docker_config['container_name']
        image_name = f"{docker_config['image_name']}:{docker_config['image_tag']}"
        
        cleanup_results = {}
        
        # 停止并删除容器
        try:
            print(f"停止并删除容器: {container_name}")
            container = self.docker_client.containers.get(container_name)
            container.stop()
            container.remove()
            cleanup_results['container'] = {
                'success': True,
                'action': 'stopped_and_removed'
            }
            print(f"容器 {container_name} 已停止并删除")
        except docker.errors.NotFound:
            cleanup_results['container'] = {
                'success': True,
                'action': 'not_found'
            }
            print(f"容器 {container_name} 不存在")
        except Exception as e:
            cleanup_results['container'] = {
                'success': False,
                'error': str(e)
            }
            print(f"容器清理失败: {e}")
            
        # 删除镜像
        try:
            print(f"删除镜像: {image_name}")
            image = self.docker_client.images.get(image_name)
            self.docker_client.images.remove(image.id)
            cleanup_results['image'] = {
                'success': True,
                'action': 'removed'
            }
            print(f"镜像 {image_name} 已删除")
        except docker.errors.ImageNotFound:
            cleanup_results['image'] = {
                'success': True,
                'action': 'not_found'
            }
            print(f"镜像 {image_name} 不存在")
        except Exception as e:
            cleanup_results['image'] = {
                'success': False,
                'error': str(e)
            }
            print(f"镜像删除失败: {e}")
            
        return cleanup_results
        
    def create_deployment_report(self, build_result, run_result, test_results, performance_result, cleanup_result=None):
        """
        创建部署报告
        
        Args:
            build_result: 构建结果
            run_result: 运行结果
            test_results: 测试结果
            performance_result: 性能测试结果
            cleanup_result: 清理结果
            
        Returns:
            部署报告
        """
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.config,
            "build_result": build_result,
            "run_result": run_result,
            "test_results": test_results,
            "performance_result": performance_result,
            "cleanup_result": cleanup_result,
            "summary": self._create_deployment_summary(build_result, run_result, test_results, performance_result)
        }
        
        # 保存报告
        report_path = self.output_dir / 'deployment_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"\n部署报告保存到: {report_path}")
        
        return report
        
    def _create_deployment_summary(self, build_result, run_result, test_results, performance_result):
        """
        创建部署摘要
        
        Args:
            build_result: 构建结果
            run_result: 运行结果
            test_results: 测试结果
            performance_result: 性能测试结果
            
        Returns:
            摘要字典
        """
        summary = {
            "build_success": build_result.get('success', False),
            "run_success": run_result.get('success', False),
            "api_tests_passed": 0,
            "api_tests_total": len(test_results),
            "performance_success": performance_result.get('success_rate', 0) > 0.9,
            "deployment_ready": False,
            "recommendations": []
        }
        
        # 计算API测试通过率
        for test_name, test_result in test_results.items():
            if test_result.get('success', False):
                summary['api_tests_passed'] += 1
                
        # 检查部署是否就绪
        summary['deployment_ready'] = (
            summary['build_success'] and
            summary['run_success'] and
            summary['api_tests_passed'] == summary['api_tests_total'] and
            summary['performance_success']
        )
        
        # 生成建议
        if not summary['build_success']:
            summary['recommendations'].append("镜像构建失败，请检查Dockerfile和构建环境")
            
        if not summary['run_success']:
            summary['recommendations'].append("容器运行失败，请检查端口映射和资源限制")
            
        if summary['api_tests_passed'] < summary['api_tests_total']:
            failed_tests = [name for name, result in test_results.items() if not result.get('success', False)]
            summary['recommendations'].append(f"API测试失败: {', '.join(failed_tests)}")
            
        if not summary['performance_success']:
            summary['recommendations'].append("性能测试未通过，请优化模型或增加资源")
            
        if summary['deployment_ready']:
            summary['recommendations'].append("部署就绪，可以投入生产使用")
            
        return summary
        
    def quick_deployment_demo(self):
        """
        快速部署演示
        
        Returns:
            部署结果
        """
        print("\n开始快速部署演示...")
        
        deployment_config = self.config['deployment']
        
        build_result = None
        run_result = None
        test_results = None
        performance_result = None
        cleanup_result = None
        
        # 构建镜像
        if deployment_config['build_image']:
            build_result = self.build_docker_image()
            if not build_result['success']:
                print("镜像构建失败，停止部署")
                return {
                    'build_result': build_result,
                    'error': 'Image build failed'
                }
                
        # 运行容器
        if deployment_config['run_container']:
            run_result = self.run_docker_container()
            if not run_result['success']:
                print("容器运行失败，停止部署")
                return {
                    'build_result': build_result,
                    'run_result': run_result,
                    'error': 'Container run failed'
                }
                
            # 等待服务启动
            print("等待服务启动...")
            time.sleep(10)
            
        # 测试API
        if deployment_config['test_api']:
            test_results = self.test_api_endpoints()
            
        # 性能测试
        if deployment_config['performance_test']:
            performance_result = self.performance_test()
            
        # 清理
        if deployment_config['cleanup']:
            cleanup_result = self.cleanup()
            
        # 创建部署报告
        report = self.create_deployment_report(
            build_result, run_result, test_results, performance_result, cleanup_result
        )
        
        print("\n快速部署演示完成！")
        
        return {
            'build_result': build_result,
            'run_result': run_result,
            'test_results': test_results,
            'performance_result': performance_result,
            'cleanup_result': cleanup_result,
            'report': report
        }


def main():
    """
    主函数
    """
    print("语义引导的轻量级特征检测器 - 部署示例")
    print("="*60)
    
    # 配置参数
    config = {
        'config_path': None,  # 配置文件路径
        'quick_demo': True,  # 使用快速演示模式
        'build_image': True,  # 构建镜像
        'run_container': True,  # 运行容器
        'test_api': True,  # 测试API
        'performance_test': True,  # 性能测试
        'cleanup': False  # 清理资源
    }
    
    # 创建部署示例
    deployment_example = DeploymentExample(config_path=config['config_path'])
    
    try:
        if config['quick_demo']:
            # 快速部署演示
            print("\n使用快速部署演示模式...")
            deployment_config = deployment_example.config['deployment']
            deployment_config.update({
                'build_image': config['build_image'],
                'run_container': config['run_container'],
                'test_api': config['test_api'],
                'performance_test': config['performance_test'],
                'cleanup': config['cleanup']
            })
            
            results = deployment_example.quick_deployment_demo()
        else:
            # 完整部署流程
            print("\n使用完整部署流程...")
            
            # 构建镜像
            if config['build_image']:
                build_result = deployment_example.build_docker_image()
                if not build_result['success']:
                    print("镜像构建失败，停止部署")
                    return {
                        'build_result': build_result,
                        'error': 'Image build failed'
                    }
            else:
                build_result = None
                
            # 运行容器
            if config['run_container']:
                run_result = deployment_example.run_docker_container()
                if not run_result['success']:
                    print("容器运行失败，停止部署")
                    return {
                        'build_result': build_result,
                        'run_result': run_result,
                        'error': 'Container run failed'
                    }
                    
                # 等待服务启动
                print("等待服务启动...")
                time.sleep(10)
            else:
                run_result = None
                
            # 测试API
            if config['test_api']:
                test_results = deployment_example.test_api_endpoints()
            else:
                test_results = None
                
            # 性能测试
            if config['performance_test']:
                performance_result = deployment_example.performance_test()
            else:
                performance_result = None
                
            # 清理
            if config['cleanup']:
                cleanup_result = deployment_example.cleanup()
            else:
                cleanup_result = None
                
            # 创建部署报告
            report = deployment_example.create_deployment_report(
                build_result, run_result, test_results, performance_result, cleanup_result
            )
            
            results = {
                'build_result': build_result,
                'run_result': run_result,
                'test_results': test_results,
                'performance_result': performance_result,
                'cleanup_result': cleanup_result,
                'report': report
            }
            
        print("\n" + "="*60)
        print("部署示例完成！")
        print(f"结果保存在: {deployment_example.output_dir}")
        print("="*60)
        
        return results
        
    except KeyboardInterrupt:
        print("\n部署被用户中断")
        # 尝试清理资源
        if deployment_example.docker_client:
            print("尝试清理资源...")
            deployment_example.cleanup()
    except Exception as e:
        print(f"\n部署过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试清理资源
        if deployment_example.docker_client:
            print("尝试清理资源...")
            deployment_example.cleanup()


if __name__ == "__main__":
    # 检查Docker是否可用
    try:
        client = docker.from_env()
        print("Docker环境检查通过")
    except Exception as e:
        print(f"Docker环境检查失败: {e}")
        print("请确保Docker已安装并运行")
        sys.exit(1)
        
    # 运行主函数
    results = main()
    
    # 可选：返回结果供其他示例使用
    if results:
        print("\n部署成功完成！")
    else:
        print("\n部署失败！")