"""
模型转换示例
演示如何使用工具脚本来转换语义引导的轻量级特征检测器模型格式
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import time
import os
import sys
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入项目模块
from semantic_guided_xfeat_implementation import SemanticGuidedXFeat
from tools import ModelConverter


class ModelConversionExample:
    """
    模型转换示例类
    """
    
    def __init__(self, model_path=None, config_path=None):
        """
        初始化模型转换示例
        
        Args:
            model_path: 模型文件路径
            config_path: 配置文件路径
        """
        self.model_path = model_path
        self.config_path = config_path
        
        # 加载配置
        self.config = self._load_config()
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化模型转换器
        self.converter = ModelConverter()
        
    def _load_config(self):
        """
        加载配置
        
        Returns:
            配置字典
        """
        if self.config_path is None:
            # 使用默认配置
            config = {
                "model": {
                    "input_channels": 1,
                    "feature_dim": 128,
                    "hidden_dim": 256,
                    "num_keypoints": 500,
                    "backbone": "resnet18",
                    "use_semantic": True,
                    "semantic_channels": 64
                },
                "conversion": {
                    "target_formats": ["onnx", "tensorrt"],
                    "onnx_params": {
                        "opset_version": 11,
                        "dynamic_axes": {
                            "input": {0: "batch_size", 2: "height", 3: "width"},
                            "output": {0: "batch_size"}
                        }
                    },
                    "tensorrt_params": {
                        "precision": "fp16",
                        "max_batch_size": 8,
                        "max_workspace_size": 1073741824  # 1GB
                    }
                },
                "validation": {
                    "test_input_shape": [1, 1, 256, 256],
                    "tolerance": 1e-3,
                    "num_test_runs": 10
                }
            }
        else:
            # 从文件加载配置
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
        return config
        
    def create_dummy_model(self, save_path=None):
        """
        创建虚拟模型（用于演示）
        
        Args:
            save_path: 保存路径
            
        Returns:
            模型文件路径
        """
        print("\n创建虚拟模型...")
        
        if save_path is None:
            save_dir = Path("./examples/output")
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "dummy_model.pth"
            
        # 创建模型
        model_config = self.config['model']
        model = SemanticGuidedXFeat(
            input_channels=model_config['input_channels'],
            feature_dim=model_config['feature_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_keypoints=model_config['num_keypoints'],
            backbone_type=model_config['backbone'],
            use_semantic=model_config['use_semantic'],
            semantic_channels=model_config['semantic_channels']
        )
        
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config
        }, save_path)
        
        print(f"虚拟模型已保存到: {save_path}")
        return str(save_path)
        
    def convert_to_onnx(self, model_path, output_path=None):
        """
        转换模型到ONNX格式
        
        Args:
            model_path: 输入模型路径
            output_path: 输出路径
            
        Returns:
            转换结果
        """
        print("\n转换模型到ONNX格式...")
        
        if output_path is None:
            output_dir = Path("./examples/output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "model.onnx"
            
        # 转换参数
        onnx_params = self.config['conversion']['onnx_params']
        
        # 执行转换
        start_time = time.time()
        result = self.converter.convert_to_onnx(
            model_path=model_path,
            output_path=str(output_path),
            opset_version=onnx_params['opset_version'],
            dynamic_axes=onnx_params['dynamic_axes']
        )
        conversion_time = time.time() - start_time
        
        if result['success']:
            print(f"ONNX转换成功！")
            print(f"输出文件: {result['output_path']}")
            print(f"转换时间: {conversion_time:.2f} 秒")
            print(f"模型大小: {result['model_size_mb']:.2f} MB")
        else:
            print(f"ONNX转换失败: {result['error']}")
            
        return result
        
    def convert_to_tensorrt(self, model_path, output_path=None):
        """
        转换模型到TensorRT格式
        
        Args:
            model_path: 输入模型路径
            output_path: 输出路径
            
        Returns:
            转换结果
        """
        print("\n转换模型到TensorRT格式...")
        
        if output_path is None:
            output_dir = Path("./examples/output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "model.trt"
            
        # 转换参数
        trt_params = self.config['conversion']['tensorrt_params']
        
        # 执行转换
        start_time = time.time()
        result = self.converter.convert_to_tensorrt(
            model_path=model_path,
            output_path=str(output_path),
            precision=trt_params['precision'],
            max_batch_size=trt_params['max_batch_size'],
            max_workspace_size=trt_params['max_workspace_size']
        )
        conversion_time = time.time() - start_time
        
        if result['success']:
            print(f"TensorRT转换成功！")
            print(f"输出文件: {result['output_path']}")
            print(f"转换时间: {conversion_time:.2f} 秒")
            print(f"模型大小: {result['model_size_mb']:.2f} MB")
        else:
            print(f"TensorRT转换失败: {result['error']}")
            
        return result
        
    def validate_conversion(self, original_model_path, converted_model_path, format_type):
        """
        验证转换结果
        
        Args:
            original_model_path: 原始模型路径
            converted_model_path: 转换后模型路径
            format_type: 转换格式类型
            
        Returns:
            验证结果
        """
        print(f"\n验证{format_type.upper()}转换结果...")
        
        # 验证参数
        validation_config = self.config['validation']
        
        # 执行验证
        start_time = time.time()
        result = self.converter.validate_conversion(
            original_model_path=original_model_path,
            converted_model_path=converted_model_path,
            format_type=format_type,
            input_shape=validation_config['test_input_shape'],
            tolerance=validation_config['tolerance'],
            num_runs=validation_config['num_test_runs']
        )
        validation_time = time.time() - start_time
        
        if result['success']:
            print(f"{format_type.upper()}验证通过！")
            print(f"最大差异: {result['max_difference']:.6f}")
            print(f"平均差异: {result['mean_difference']:.6f}")
            print(f"平均推理时间 (原始): {result['original_inference_time']:.4f} 秒")
            print(f"平均推理时间 ({format_type.upper()}): {result['converted_inference_time']:.4f} 秒")
            print(f"加速比: {result['speedup']:.2f}x")
        else:
            print(f"{format_type.upper()}验证失败: {result['error']}")
            
        result['validation_time'] = validation_time
        return result
        
    def benchmark_models(self, model_paths, format_types):
        """
        基准测试不同格式的模型
        
        Args:
            model_paths: 模型路径列表
            format_types: 格式类型列表
            
        Returns:
            基准测试结果
        """
        print("\n基准测试不同格式的模型...")
        
        validation_config = self.config['validation']
        input_shape = validation_config['test_input_shape']
        num_runs = validation_config['num_test_runs']
        
        results = {}
        
        for model_path, format_type in zip(model_paths, format_types):
            print(f"\n测试 {format_type.upper()} 模型...")
            
            result = self.converter.benchmark_model(
                model_path=model_path,
                format_type=format_type,
                input_shape=input_shape,
                num_runs=num_runs
            )
            
            if result['success']:
                print(f"{format_type.upper()} 基准测试完成")
                print(f"平均推理时间: {result['mean_inference_time']:.4f} 秒")
                print(f"FPS: {result['fps']:.2f}")
                print(f"内存使用: {result['memory_usage_mb']:.2f} MB")
                
                results[format_type] = result
            else:
                print(f"{format_type.upper()} 基准测试失败: {result['error']}")
                
        return results
        
    def create_conversion_report(self, conversion_results, validation_results, benchmark_results, output_dir):
        """
        创建转换报告
        
        Args:
            conversion_results: 转换结果字典
            validation_results: 验证结果字典
            benchmark_results: 基准测试结果字典
            output_dir: 输出目录
            
        Returns:
            转换报告字典
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建报告
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "conversion_config": self.config['conversion'],
            "conversion_results": conversion_results,
            "validation_results": validation_results,
            "benchmark_results": benchmark_results,
            "summary": self._create_summary(conversion_results, validation_results, benchmark_results)
        }
        
        # 保存报告
        report_path = output_dir / 'conversion_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"转换报告保存到: {report_path}")
        
        return report
        
    def _create_summary(self, conversion_results, validation_results, benchmark_results):
        """
        创建转换摘要
        
        Args:
            conversion_results: 转换结果字典
            validation_results: 验证结果字典
            benchmark_results: 基准测试结果字典
            
        Returns:
            摘要字典
        """
        summary = {
            "total_conversions": len(conversion_results),
            "successful_conversions": len([r for r in conversion_results.values() if r.get('success', False)]),
            "successful_validations": len([r for r in validation_results.values() if r.get('success', False)]),
            "best_format": None,
            "best_speedup": 0.0,
            "recommendations": []
        }
        
        # 找出最佳格式
        if benchmark_results:
            best_fps = 0
            best_format = None
            
            for format_type, result in benchmark_results.items():
                if result.get('success', False):
                    fps = result.get('fps', 0)
                    if fps > best_fps:
                        best_fps = fps
                        best_format = format_type
                        
            summary['best_format'] = best_format
            summary['best_fps'] = best_fps
            
            # 计算最佳加速比
            if 'pytorch' in benchmark_results and best_format in benchmark_results:
                pytorch_fps = benchmark_results['pytorch'].get('fps', 1)
                best_fps = benchmark_results[best_format].get('fps', 1)
                summary['best_speedup'] = best_fps / pytorch_fps
                
        # 生成建议
        if summary['successful_conversions'] < summary['total_conversions']:
            summary['recommendations'].append("部分格式转换失败，建议检查模型结构和依赖")
            
        if summary['successful_validations'] < summary['successful_conversions']:
            summary['recommendations'].append("部分转换验证失败，建议调整转换参数")
            
        if summary['best_format'] == 'tensorrt':
            summary['recommendations'].append("TensorRT格式性能最佳，推荐用于生产环境")
        elif summary['best_format'] == 'onnx':
            summary['recommendations'].append("ONNX格式性能良好，推荐用于跨平台部署")
            
        if summary['best_speedup'] > 2.0:
            summary['recommendations'].append(f"模型转换带来显著性能提升（{summary['best_speedup']:.1f}x加速比）")
            
        return summary
        
    def visualize_conversion_results(self, conversion_results, validation_results, benchmark_results, output_dir):
        """
        可视化转换结果
        
        Args:
            conversion_results: 转换结果字典
            validation_results: 验证结果字典
            benchmark_results: 基准测试结果字典
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建可视化图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 转换成功率
        if conversion_results:
            formats = list(conversion_results.keys())
            success_rates = [1 if conversion_results[f].get('success', False) else 0 for f in formats]
            
            colors = ['green' if rate == 1 else 'red' for rate in success_rates]
            bars = axes[0, 0].bar(formats, success_rates, color=colors, alpha=0.7)
            axes[0, 0].set_title('模型转换成功率')
            axes[0, 0].set_ylabel('成功率')
            axes[0, 0].set_ylim(0, 1.2)
            
            # 添加数值标签
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{rate:.0%}', ha='center', va='bottom')
                               
        # 2. 验证准确率
        if validation_results:
            formats = list(validation_results.keys())
            accuracies = []
            
            for format_type in formats:
                if validation_results[format_type].get('success', False):
                    # 计算准确率（1 - 平均差异）
                    mean_diff = validation_results[format_type].get('mean_difference', 1.0)
                    accuracy = max(0, 1 - mean_diff)
                else:
                    accuracy = 0
                accuracies.append(accuracy)
                
            bars = axes[0, 1].bar(formats, accuracies, color=['blue', 'orange'], alpha=0.7)
            axes[0, 1].set_title('转换验证准确率')
            axes[0, 1].set_ylabel('准确率')
            axes[0, 1].set_ylim(0, 1.2)
            
            # 添加数值标签
            for bar, accuracy in zip(bars, accuracies):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{accuracy:.2%}', ha='center', va='bottom')
                               
        # 3. 性能对比
        if benchmark_results:
            formats = list(benchmark_results.keys())
            fps_values = []
            memory_values = []
            
            for format_type in formats:
                if benchmark_results[format_type].get('success', False):
                    fps_values.append(benchmark_results[format_type].get('fps', 0))
                    memory_values.append(benchmark_results[format_type].get('memory_usage_mb', 0))
                else:
                    fps_values.append(0)
                    memory_values.append(0)
                    
            # FPS对比
            x = np.arange(len(formats))
            width = 0.35
            
            bars1 = axes[1, 0].bar(x - width/2, fps_values, width, label='FPS', color='green', alpha=0.7)
            axes[1, 0].set_title('推理性能对比 (FPS)')
            axes[1, 0].set_ylabel('FPS')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(formats)
            axes[1, 0].legend()
            
            # 添加数值标签
            for bar, fps in zip(bars1, fps_values):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{fps:.1f}', ha='center', va='bottom')
                               
            # 内存使用对比
            bars2 = axes[1, 1].bar(x + width/2, memory_values, width, label='内存 (MB)', color='red', alpha=0.7)
            axes[1, 1].set_title('内存使用对比')
            axes[1, 1].set_ylabel('内存使用 (MB)')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(formats)
            axes[1, 1].legend()
            
            # 添加数值标签
            for bar, memory in zip(bars2, memory_values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{memory:.1f}', ha='center', va='bottom')
                               
        plt.tight_layout()
        plt.savefig(output_dir / 'conversion_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"转换结果可视化保存到: {output_dir / 'conversion_results.png'}")
        
    def quick_conversion_demo(self):
        """
        快速转换演示
        
        Returns:
            转换结果
        """
        print("\n开始快速转换演示...")
        
        # 创建虚拟模型
        dummy_model_path = self.create_dummy_model()
        
        # 转换结果
        conversion_results = {}
        validation_results = {}
        benchmark_results = {}
        
        # 转换到ONNX
        try:\            conversion_results['onnx'] = self.convert_to_onnx(dummy_model_path)
        except Exception as e:
            print(f"ONNX转换失败: {e}")
            conversion_results['onnx'] = {'success': False, 'error': str(e)}
            
        # 转换到TensorRT
        try:
            conversion_results['tensorrt'] = self.convert_to_tensorrt(dummy_model_path)
        except Exception as e:
            print(f"TensorRT转换失败: {e}")
            conversion_results['tensorrt'] = {'success': False, 'error': str(e)}
            
        # 验证转换结果
        for format_type, result in conversion_results.items():
            if result.get('success', False):
                try:
                    validation_results[format_type] = self.validate_conversion(
                        dummy_model_path, result['output_path'], format_type
                    )
                except Exception as e:
                    print(f"{format_type}验证失败: {e}")
                    validation_results[format_type] = {'success': False, 'error': str(e)}
                    
        # 基准测试
        model_paths = [dummy_model_path]
        format_types = ['pytorch']
        
        for format_type, result in conversion_results.items():
            if result.get('success', False):
                model_paths.append(result['output_path'])
                format_types.append(format_type)
                
        try:
            benchmark_results = self.benchmark_models(model_paths, format_types)
        except Exception as e:
            print(f"基准测试失败: {e}")
            
        # 创建输出目录
        output_dir = Path("./examples/output/conversion_demo")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 可视化结果
        self.visualize_conversion_results(conversion_results, validation_results, benchmark_results, output_dir)
        
        # 创建转换报告
        self.create_conversion_report(conversion_results, validation_results, benchmark_results, output_dir)
        
        print("\n快速转换演示完成！")
        
        return {
            'conversion_results': conversion_results,
            'validation_results': validation_results,
            'benchmark_results': benchmark_results
        }


def main():
    """
    主函数
    """
    print("语义引导的轻量级特征检测器 - 模型转换示例")
    print("="*60)
    
    # 配置参数
    config = {
        'model_path': None,  # 模型路径（None表示创建虚拟模型）
        'config_path': None,  # 配置文件路径
        'quick_demo': True,  # 使用快速演示模式
        'target_formats': ['onnx', 'tensorrt'],  # 目标格式
        'create_visualization': True,  # 创建可视化
        'output_dir': './examples/output'
    }
    
    # 创建模型转换示例
    conversion_example = ModelConversionExample(
        model_path=config['model_path'],
        config_path=config['config_path']
    )
    
    try:
        if config['quick_demo']:
            # 快速转换演示
            print("\n使用快速演示模式...")
            results = conversion_example.quick_conversion_demo()
        else:
            # 完整转换流程
            print("\n使用完整转换流程...")
            
            # 创建或加载模型
            if config['model_path'] is None:
                dummy_model_path = conversion_example.create_dummy_model()
            else:
                dummy_model_path = config['model_path']
                
            # 转换结果
            conversion_results = {}
            validation_results = {}
            benchmark_results = {}
            
            # 转换到各种格式
            for format_type in config['target_formats']:
                if format_type == 'onnx':
                    conversion_results['onnx'] = conversion_example.convert_to_onnx(dummy_model_path)
                elif format_type == 'tensorrt':
                    conversion_results['tensorrt'] = conversion_example.convert_to_tensorrt(dummy_model_path)
                    
            # 验证转换结果
            for format_type, result in conversion_results.items():
                if result.get('success', False):
                    validation_results[format_type] = conversion_example.validate_conversion(
                        dummy_model_path, result['output_path'], format_type
                    )
                    
            # 基准测试
            model_paths = [dummy_model_path]
            format_types = ['pytorch']
            
            for format_type, result in conversion_results.items():
                if result.get('success', False):
                    model_paths.append(result['output_path'])
                    format_types.append(format_type)
                    
            benchmark_results = conversion_example.benchmark_models(model_paths, format_types)
            
            # 创建输出目录
            output_dir = Path(config['output_dir']) / 'conversion_results'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 可视化结果
            if config['create_visualization']:
                conversion_example.visualize_conversion_results(
                    conversion_results, validation_results, benchmark_results, output_dir
                )
                
            # 创建转换报告
            conversion_example.create_conversion_report(
                conversion_results, validation_results, benchmark_results, output_dir
            )
            
        print("\n" + "="*60)
        print("模型转换示例完成！")
        print(f"结果保存在: {config['output_dir']}")
        print("="*60)
        
        return results
        
    except KeyboardInterrupt:
        print("\n转换被用户中断")
    except Exception as e:
        print(f"\n转换过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行主函数
    results = main()
    
    # 可选：返回结果供其他示例使用
    if results:
        print("\n转换成功完成！")
    else:
        print("\n转换失败！")