"""
贴片天线设计系统 - 模型使用模块
Patch Antenna Design System - Model Usage Module
"""

import sys
import os
import time
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
try:
    from patch_antenna_design import PatchAntennaDesignSystem
    from python_hfss import calculate_from_hfss as calculate_from_hfss_py
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)

def get_device():
    """自动检测可用设备（优先GPU，没有则用CPU）"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"ℹ️  未检测到GPU，使用CPU推理")
    return device

def use_trained_gan_model(model_info_path='models/trained_gan_model_info.npy',
                         target_performances=None,
                         gan_generator_path='models/gan_generator.pth',
                         forward_gan_generator_path='models/forward_gan_generator.pth'):
    """
    使用已训练的GAN模型生成天线设计

    Args:
        model_info_path: 模型信息文件路径
        target_performances: 目标性能参数列表
    """
    print("\n" + "=" * 70)
    print("使用已训练的GAN模型")
    print("=" * 70)

    device = get_device()
    system = PatchAntennaDesignSystem()

    # 1. 加载训练信息和模型状态
    print(f"\n1. 加载训练信息从 {model_info_path}...")
    if os.path.exists(model_info_path):
        training_info = np.load(model_info_path, allow_pickle=True).item()
        print(f"✅ 训练信息加载完成！")
        print(f"   训练时间: {training_info.get('timestamp', '未知')}")
        print(f"   训练样本数: {training_info.get('data_samples', '未知')}")
        print(f"   训练设备: {training_info.get('device', '未知')}")

        # 加载预处理器状态（如果存在）
        if 'scalers' in training_info:
            system.input_scaler = training_info['scalers']['input_scaler']
            system.target_scaler = training_info['scalers']['target_scaler']
            print("✅ 数据预处理器加载完成！")
    else:
        print(f"⚠️  未找到训练信息文件，使用默认配置")
        training_info = {}

    # 2. 定义设计目标
    if target_performances is None:
        target_performances = [
            [-35.0, 2.45, 7.0],   # WiFi 2.45GHz 高性能设计
            [-30.0, 2.4, 6.5],    # WiFi 2.4GHz 标准设计
            [-25.0, 2.5, 6.0],    # 低成本设计
            [-40.0, 2.42, 7.5]    # 超高性能设计
        ]
    # 加载模型
    # 确保模型已加载
    if system.generator is None:
        try:
            system.create_gan_models()
            state_dict = torch.load(gan_generator_path, map_location=system.device)
            system.generator.load_state_dict(state_dict)
            print("成功加载预训练的反向GAN生成器")
        except Exception as e:
            print(f"加载反向GAN模型失败: {e}")

    if system.forward_generator is None:
        try:
            system.create_forward_gan_models()
            state_dict = torch.load(forward_gan_generator_path, map_location=system.device)
            system.forward_generator.load_state_dict(state_dict)
            print("成功加载预训练的正向GAN生成器")
        except Exception as e:
            print(f"加载正向GAN模型失败: {e}")

    if system.performance_predictor is None:
        try:
            system.performance_predictor = system.create_performance_predictor()
            state_dict = torch.load('best_performance_predictor.pth', map_location=system.device)
            system.performance_predictor.load_state_dict(state_dict)
            print("成功加载预训练的性能预测器")
        except Exception as e:
            print(f"加载性能预测器失败: {e}")

    # 3. 使用GAN生成天线设计
    print(f"\n2. 使用GAN生成天线设计...")
    try:
        generated_designs, generated_performances = system.generate_antenna_designs(
            target_performances, num_samples=20
        )
    except Exception as e:
        print(f"❌ GAN生成过程出错: {e}")
        print("💡 请确认模型文件和预处理器状态是否完整保存")
        return None

    # 4. 保存生成的设计
    design_df = pd.DataFrame({
        'patch_length': generated_designs[:, 0],
        'patch_width': generated_designs[:, 1],
        's11_min': generated_performances[:, 0],
        'freq_at_s11_min': generated_performances[:, 1],
        'far_field_gain': generated_performances[:, 2]
    })

    # 创建结果目录
    if not os.path.exists('results'):
        os.makedirs('results')

    design_csv_path = 'results/gan_generated_designs.csv'
    design_df.to_csv(design_csv_path, index=False)
    print(f"生成的天线设计已保存到 {design_csv_path}")

    # 后续代码保持不变...


if __name__ == "__main__":
    print("贴片天线GAN模型使用系统")
    print("=" * 70)

    # 使用已训练模型
    model_info_path = 'models/trained_gan_model_info.npy'

    # 可以自定义目标性能
    target_specs = [
        [-35.0, 2.45, 7.0],  # WiFi 2.45GHz 高性能设计
    ]

    result = use_trained_gan_model(model_info_path, target_specs)

    print("\n" + "=" * 70)
    print("模型使用完成！")
    print("=" * 70)
    print("\n您可以在 results 目录中查看生成的设计结果。")
