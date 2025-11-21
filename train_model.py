"""
贴片天线设计系统 - 模型训练模块
Patch Antenna Design System - Model Training Module
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
    import calculate_by_hfss
    from patch_antenna_design import PatchAntennaDesignSystem
    from merge_csv_files import merge_single_line_csv_files
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
        print(f"ℹ️  未检测到GPU，使用CPU训练（速度可能较慢）")
    return device

def to_tensor_and_device(data, device):
    """将数据转换为tensor并移到指定设备"""
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    return data.to(device)

def train_gan_model(create_antenna_data=0, model_save_path='trained_gan_model.pth',
                    training_type = 's_params'):
    """
    训练GAN模型并保存

    Args:
        create_antenna_data: 需要生成的天线数据数量
        model_save_path: 模型保存路径
        training_type: 训练类型 ('s_params' 或 'far_field')
    """
    print("\n" + "=" * 70)
    print("GAN 模型训练")
    print("=" * 70)

    device = get_device()
    system = PatchAntennaDesignSystem()

    # 1. 数据准备阶段
    print("\n1. 准备天线数据...")

    # 生成天线数据（如果需要）
    if create_antenna_data != 0:
        print(f"\n 生成{create_antenna_data}个天线数据...")
        calculate_by_hfss.Generate_test_data(create_antenna_data)

    if training_type == 's_params':
        # 合并数据文件
        print("=============================合并所有数据=============================")
        input_pattern = "./Train_data/data_dict_pandas_*.csv"
        output_file = "merged_detailed_antenna_data.csv"
        header_check_count = 40
        merge_single_line_csv_files(input_pattern, output_file, header_check_count)
        print(f"\n=============================合并完成！=============================")

    elif training_type == 'far_field':
        # 合并数据文件
        print("=============================合并所有数据=============================")
        input_pattern = "./Train_data/data_dict_pandas_*.csv"
        output_file = "merged_detailed_antenna_data_far_field.csv"
        header_check_count = 40
        merge_single_line_csv_files(input_pattern, output_file, header_check_count)
        print(f"\n=============================合并完成！=============================")

    # 加载数据
    print("=============================加载数据=============================")
    try:
        X_scaled, y, X_original, y_original = system.load_csv_data(
            csv_file='./merged_detailed_antenna_data_far_field.csv',
            param_cols=['patch_length', 'patch_width'],
            perf_cols=None  # 让函数自动检测列名
        )
        print(
            f"=============================数据加载完成: {X_original.shape[0]}个样本=============================")
    except Exception as e:
        print(f"=============================❌ 数据加载失败，使用合成数据: {e}=============================")
        X_scaled, y, X_original, y_original = system.generate_synthetic_data(num_samples=create_antenna_data)


    # 不使用数据增强，直接对原始数据进行归一化
    X_scaled_original = system.scaler.fit_transform(X_original)
    y_scaled_original = system.target_scaler.fit_transform(y_original)

    # 划分数据集并移到设备
    print("=============================划分数据集并移到设备=============================")
    X_train, X_val, y_train, y_val = train_test_split(X_scaled_original, y_scaled_original, test_size=0.2, random_state=42)


    def to_tensor_and_device(data, device):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        return data.to(device)

    X_train = to_tensor_and_device(X_train, device)
    y_train = to_tensor_and_device(y_train, device)
    X_val = to_tensor_and_device(X_val, device)
    y_val = to_tensor_and_device(y_val, device)

    # 2. 模型训练阶段
    print("=============================训练模型=============================")
    print(f"\n2. GAN模型训练...")
    # 同时训练正向和反向GAN
    history = system.train_gan(X_train, y_train, epochs=3000, batch_size=128, train_both=True)
   # # 或者保持原有功能，只训练正向GAN
   #  history = system.train_gan(X_train, y_train, epochs=3000, batch_size=128, forward_gan=True)
   #
   # # 或者保持原有功能，只训练反向GAN
   #  history = system.train_gan(X_train, y_train, epochs=3000, batch_size=128, forward_gan=False)

    system.visualize_gan_results(history)

    # 3. 保存训练好的模型和相关信息
    print(f"\n3. 保存训练模型到 {model_save_path}...")

    # 创建保存目录
    save_dir = os.path.dirname(model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存训练信息
    training_info = {
        'gan_history_forward': history['forward'] if 'forward' in history else None,
        'gan_history_reverse': history['reverse'] if 'reverse' in history else None,
        'scalers': {
            'input_scaler': {
                'scale_': system.scaler.scale_,
                'mean_': system.scaler.mean_,
                'var_': system.scaler.var_,
                'n_features_in_': getattr(system.scaler, 'n_features_in_',
                                          system.scaler.scale_.shape[0] if hasattr(system.scaler, 'scale_') else 0),
                'n_samples_seen_': getattr(system.scaler, 'n_samples_seen_', 1)
            },
            'target_scaler': {
                'scale_': system.target_scaler.scale_,
                'min_': system.target_scaler.min_,
                'data_min_': system.target_scaler.data_min_,
                'data_max_': system.target_scaler.data_max_,
                'data_range_': system.target_scaler.data_range_,
                'n_features_in_': getattr(system.target_scaler, 'n_features_in_',
                                          system.target_scaler.scale_.shape[0] if hasattr(system.target_scaler,
                                                                                          'scale_') else 0),
                'n_samples_seen_': getattr(system.target_scaler, 'n_samples_seen_', 1)
            },
        },
        'X_train_shape': X_train.shape,
        'y_train_shape': y_train.shape,
        'X_val_shape': X_val.shape,
        'y_val_shape': y_val.shape,
        'device': str(device),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'data_samples': X_original.shape[0]
    }

    # 保存到文件
    np.save(model_save_path.replace('.pth', '_info.npy'), training_info)
    print("✅ 模型训练和保存完成！")

    return training_info

if __name__ == "__main__":
    print("贴片天线GAN模型训练系统")
    print("=" * 70)

    # 训练模型
    create_antenna_data = 0  # 根据需要调整数据量
    model_save_path = 'models/trained_gan_model.pth'

    train_gan_model(create_antenna_data, model_save_path)
    train_gan_model(create_antenna_data, model_save_path, 'far_field')

    print("\n" + "=" * 70)
    print("模型训练完成！")
    print("=" * 70)
