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
        output_file = "merged_detailed_antenna_data.csv"
    elif training_type == 'far_field':
        output_file = "merged_detailed_antenna_data_far_field.csv"
    else:
        print("无效的训练类型")
        return  -1
    # 合并数据文件
    print("=============================合并所有数据=============================")
    input_pattern = "./Train_data/data_dict_pandas_*.csv"
    header_check_count = 40
    merge_single_line_csv_files(input_pattern, output_file, header_check_count)
    print(f"\n=============================合并完成！=============================")

    # 加载数据
    print("=============================加载数据=============================")
    try:
        X_scaled, y, X_original, y_original = system.load_csv_data(
            # csv_file='./merged_detailed_antenna_data_far_field.csv',
            csv_file = output_file,
            param_cols=['patch_length', 'patch_width'],
            perf_cols=None  # 让函数自动检测列名
        )
        print(f"=============================数据加载完成: {X_original.shape[0]}个样本=============================")
        print(f"从{output_file}中加载数据成功！")
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
def train_inverse_model(create_antenna_data=0, model_save_path='inverse_model.pth'):
    """
    训练逆向模型，根据S参数曲线和远区场方向图预测天线结构尺寸

    Args:
        create_antenna_data: 需要生成的天线数据数量
        model_save_path: 模型保存路径
    """
    print("\n" + "=" * 70)
    print("逆向模型训练")
    print("=" * 70)

    device = get_device()
    system = PatchAntennaDesignSystem()

    # 1. 数据准备阶段
    print("\n1. 准备天线数据...")

    # 生成天线数据（如果需要）
    if create_antenna_data > 0:
        print(f"\n 生成{create_antenna_data}个天线数据...")
        calculate_by_hfss.Generate_test_data(create_antenna_data)

    # 合并包含S参数和远区场的数据
    print("=============================合并所有数据=============================")
    # input_pattern = "./RESULT/data_dict_pandas_*.csv"
    output_file = "merged_multi_output_data.csv"
    # header_check_count = 40
    # merge_single_line_csv_files(input_pattern, output_file, header_check_count)
    print("=============================合并完成！=============================")

    # 加载数据并反转输入输出关系
    print("=============================加载数据=============================")
    try:
        # 加载完整数据
        df = pd.read_csv(output_file)

        # 原始输出作为新输入：S参数和远区场
        # 加载S参数数据（自动检测S参数列）
        s_param_columns = [col for col in df.columns if col.startswith(('S11_', 's11_'))]
        if not s_param_columns:
            # 如果没有找到S参数列，尝试使用load_csv_data方法获取
            _, _, _, y_s_original = system.load_csv_data(
                csv_file=output_file,
                param_cols=['patch_length', 'patch_width'],
                perf_cols=None  # 让函数自动检测列名
            )
        else:
            y_s_original = df[s_param_columns].values

        # 加载远区场输出数据（从Gain_dB_matrix列）
        y_far_field_list = []
        actual_far_field_dim = None

        for idx, row in df.iterrows():
            try:
                # 从 Gain_dB_matrix 列提取二维矩阵数据
                matrix_str = row['Gain_dB_matrix']
                # 安全地解析字符串形式的矩阵
                matrix_data = np.array(eval(matrix_str))

                # 记录实际的远区场维度
                if actual_far_field_dim is None:
                    actual_far_field_dim = matrix_data.size
                    print(f"远区场矩阵实际维度: {matrix_data.shape}, 展平后: {actual_far_field_dim}")

                y_far_field_list.append(matrix_data.flatten())  # 展平为一维数组以便处理
            except Exception as parse_error:
                print(f"解析第{idx}行远区场数据时出错: {parse_error}")
                # 使用默认大小填充
                if actual_far_field_dim:
                    y_far_field_list.append(np.zeros(actual_far_field_dim))
                else:
                    # 假设标准尺寸 181x361 (theta: 0-180度, phi: 0-360度)
                    y_far_field_list.append(np.zeros(181 * 361))

        y_f_original = np.array(y_far_field_list)

        # 合并S参数和远区场数据作为新的输入
        X_inverse_original = np.concatenate([y_s_original, y_f_original], axis=1)

        # 原始输入作为新输出：天线尺寸
        y_inverse_original = df[['patch_length', 'patch_width']].values

        print(f"输入数据加载完成: {X_inverse_original.shape[0]}个样本")
        print(f"输入特征维度(S参数+远区场): {X_inverse_original.shape[1]}")
        print(f"输出目标维度(尺寸): {y_inverse_original.shape[1]}")
        print("数据预处理完成！")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 数据标准化和划分
    print("=============================数据预处理=============================")

    # 为输入特征创建标准化器（S参数+远区场）
    from sklearn.preprocessing import MinMaxScaler
    input_scaler = MinMaxScaler()
    X_inverse_scaled = input_scaler.fit_transform(X_inverse_original)

    # 为输出目标创建标准化器（尺寸）
    output_scaler = MinMaxScaler()
    y_inverse_scaled = output_scaler.fit_transform(y_inverse_original)

    # 划分数据集并移到设备
    print("=============================划分数据集并移到设备=============================")
    X_train, X_val, y_train, y_val = train_test_split(
        X_inverse_scaled, y_inverse_scaled, test_size=0.2, random_state=42)

    # 转换为张量并移至设备
    X_train = to_tensor_and_device(X_train, device)
    y_train = to_tensor_and_device(y_train, device)
    X_val = to_tensor_and_device(X_val, device)
    y_val = to_tensor_and_device(y_val, device)

    # 2. 模型训练阶段
    print("=============================训练逆向模型=============================")
    print(f"\n2. 逆向模型训练...")

    # 训练逆向模型
    history = system.train_inverse_model(
        X_train, y_train, X_val, y_val,
        epochs=2400, batch_size=128
    )

    # 3. 保存训练好的模型和相关信息
    print(f"\n3. 保存训练模型到 {model_save_path}...")

    # 创建保存目录
    save_dir = os.path.dirname(model_save_path)
    if save_dir and not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
            print(f"📁 创建保存目录: {save_dir}")
        except Exception as e:
            print(f"❌ 创建目录失败: {e}")
            # 如果创建目录失败，尝试使用当前目录
            model_save_path = os.path.basename(model_save_path)
            print(f"⚠️  使用当前目录保存模型: {model_save_path}")

    # 保存训练信息
    training_info = {
        'inverse_model_history': history,
        'scalers': {
            'input_scaler': {
                'scale_': input_scaler.scale_,
                'min_': input_scaler.min_,
                'data_min_': input_scaler.data_min_,
                'data_max_': input_scaler.data_max_,
                'data_range_': input_scaler.data_range_,
                'n_features_in_': getattr(input_scaler, 'n_features_in_',
                                         input_scaler.scale_.shape[0] if hasattr(input_scaler,
                                                                                'scale_') else 0),
                'n_samples_seen_': getattr(input_scaler, 'n_samples_seen_', 1)
            },
            'output_scaler': {
                'scale_': output_scaler.scale_,
                'min_': output_scaler.min_,
                'data_min_': output_scaler.data_min_,
                'data_max_': output_scaler.data_max_,
                'data_range_': output_scaler.data_range_,
                'n_features_in_': getattr(output_scaler, 'n_features_in_',
                                         output_scaler.scale_.shape[0] if hasattr(output_scaler,
                                                                                 'scale_') else 0),
                'n_samples_seen_': getattr(output_scaler, 'n_samples_seen_', 1)
            },
        },
        'X_train_shape': X_train.shape,
        'y_train_shape': y_train.shape,
        'X_val_shape': X_val.shape,
        'y_val_shape': y_val.shape,
        'device': str(device),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'data_samples': X_inverse_original.shape[0]
    }

    # 保存训练信息
    info_file_path = model_save_path.replace('.pth', '_info.npy')
    try:
        np.save(info_file_path, training_info)
        print(f"✅ 训练信息保存成功: {info_file_path}")
    except Exception as e:
        print(f"❌ 训练信息保存失败: {e}")
        return None

    # 保存模型权重
    try:
        if hasattr(system, 'inverse_model') and system.inverse_model is not None:
            torch.save(system.inverse_model.state_dict(), model_save_path)
            print(f"✅ 模型权重保存成功: {model_save_path}")
        else:
            print("❌ 模型未正确初始化，无法保存")
            return None
    except Exception as e:
        print(f"❌ 模型权重保存失败: {e}")
        return None

    print("✅ 逆向模型训练和保存完成！")
    return training_info

def train_multi_output_model(create_antenna_data=0, model_save_path='multi_output_model.pth'):
    """
    训练多输出模型，同时预测S参数曲线和远区场方向图

    Args:
        create_antenna_data: 需要生成的天线数据数量
        model_save_path: 模型保存路径
    """
    print("\n" + "=" * 70)
    print("多输出GAN模型训练")
    print("=" * 70)

    device = get_device()
    system = PatchAntennaDesignSystem()

    # 1. 数据准备阶段
    print("\n1. 准备天线数据...")

    # 生成天线数据（如果需要）
    if create_antenna_data > 0:
        print(f"\n 生成{create_antenna_data}个天线数据...")
        calculate_by_hfss.Generate_test_data(create_antenna_data)

    # 合并包含S参数和远区场的数据
    print("=============================合并所有数据=============================")
    input_pattern = "../RESULT/data_dict_pandas_*.csv"
    output_file = "merged_multi_output_data.csv"
    header_check_count = 40
    merge_single_line_csv_files(input_pattern, output_file, header_check_count)
    print("=============================合并完成！=============================")

    # 加载输入数据（贴片尺寸）
    print("=============================加载数据=============================")
    try:
        # 加载输入参数
        df = pd.read_csv(output_file)
        X_original = df[['patch_length', 'patch_width']].values

        # 加载S参数输出数据（自动检测S参数列）
        s_param_columns = [col for col in df.columns if col.startswith(('S11_', 's11_'))]
        if not s_param_columns:
            # 如果没有找到S参数列，尝试使用load_csv_data方法
            X_s, y_s, X_s_original, y_s_original = system.load_csv_data(
                csv_file=output_file,
                param_cols=['patch_length', 'patch_width'],
                perf_cols=None  # 让函数自动检测列名
            )
            y_s_original = y_s_original
        else:
            y_s_original = df[s_param_columns].values

        # 加载远区场输出数据（从Gain_dB_matrix列）
        y_far_field_list = []
        actual_far_field_dim = None

        for idx, row in df.iterrows():
            try:
                # 从 Gain_dB_matrix 列提取二维矩阵数据
                matrix_str = row['Gain_dB_matrix']
                # 安全地解析字符串形式的矩阵
                matrix_data = np.array(eval(matrix_str))

                # 记录实际的远区场维度
                if actual_far_field_dim is None:
                    actual_far_field_dim = matrix_data.size
                    print(f"远区场矩阵实际维度: {matrix_data.shape}, 展平后: {actual_far_field_dim}")

                y_far_field_list.append(matrix_data.flatten())  # 展平为一维数组以便处理
            except Exception as parse_error:
                print(f"解析第{idx}行远区场数据时出错: {parse_error}")
                # 使用默认大小填充
                if actual_far_field_dim:
                    y_far_field_list.append(np.zeros(actual_far_field_dim))
                else:
                    # 假设标准尺寸 181x361 (theta: 0-180度, phi: 0-360度)
                    y_far_field_list.append(np.zeros(181 * 361))

        y_f_original = np.array(y_far_field_list)

        print(f"输入数据加载完成: {X_original.shape[0]}个样本")
        print(f"S参数数据维度: {y_s_original.shape}")
        print(f"远区场数据维度: {y_f_original.shape}")
        # print(f"数据为：{y_f_original}")
        print("数据预处理完成！")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 数据标准化和划分
    print("=============================数据预处理=============================")

    # 统一输入特征范围
    X_scaled_original = system.scaler.fit_transform(X_original)

    # 分别标准化输出
    y_s_scaled = system.target_scaler.fit_transform(y_s_original)

    # 为远区场创建独立的标准化器
    from sklearn.preprocessing import MinMaxScaler
    far_field_scaler = MinMaxScaler()
    y_f_scaled = far_field_scaler.fit_transform(y_f_original)

    # 划分数据集并移到设备
    print("=============================划分数据集并移到设备=============================")
    # S参数数据划分
    X_s_train, X_s_val, y_s_train, y_s_val = train_test_split(
        X_scaled_original, y_s_scaled, test_size=0.2, random_state=42)

    # 远区场数据划分
    X_f_train, X_f_val, y_f_train, y_f_val = train_test_split(
        X_scaled_original, y_f_scaled, test_size=0.2, random_state=42)

    # 转换为张量并移至设备
    X_s_train = to_tensor_and_device(X_s_train, device)
    y_s_train = to_tensor_and_device(y_s_train, device)
    X_s_val = to_tensor_and_device(X_s_val, device)
    y_s_val = to_tensor_and_device(y_s_val, device)

    X_f_train = to_tensor_and_device(X_f_train, device)
    y_f_train = to_tensor_and_device(y_f_train, device)
    X_f_val = to_tensor_and_device(X_f_val, device)
    y_f_val = to_tensor_and_device(y_f_val, device)

    # 2. 模型训练阶段
    print("=============================训练多输出模型=============================")
    print(f"\n2. 多输出GAN模型训练...")

    # 获取实际的远区场维度
    actual_far_field_dim = y_f_original.shape[1] if len(y_f_original.shape) > 1 else y_f_original.shape[0]
    print(f"实际使用的远区场输出维度: {actual_far_field_dim}")

    # 训练多输出GAN模型
    history = system.train_multi_output_gan(
        X_s_train, y_s_train, X_f_train, y_f_train,
        epochs=2500, batch_size=128,
        far_field_dim=actual_far_field_dim
    )

    # 3. 保存训练好的模型和相关信息
    print(f"\n3. 保存训练模型到 {model_save_path}...")

    # 创建保存目录
    save_dir = os.path.dirname(model_save_path)
    if save_dir and not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
            print(f"📁 创建保存目录: {save_dir}")
        except Exception as e:
            print(f"❌ 创建目录失败: {e}")
            # 如果创建目录失败，尝试使用当前目录
            model_save_path = os.path.basename(model_save_path)
            print(f"⚠️  使用当前目录保存模型: {model_save_path}")

    # 保存训练信息
    training_info = {
        'multi_output_history': history,
        'actual_far_field_dim': actual_far_field_dim,
        'scalers': {
            'input_scaler': {
                'scale_': system.scaler.scale_,
                'mean_': system.scaler.mean_,
                'var_': system.scaler.var_,
                'n_features_in_': getattr(system.scaler, 'n_features_in_',
                                          system.scaler.scale_.shape[0] if hasattr(system.scaler, 'scale_') else 0),
                'n_samples_seen_': getattr(system.scaler, 'n_samples_seen_', 1)
            },
            's_params_scaler': {
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
            'far_field_scaler': {
                'scale_': far_field_scaler.scale_,
                'min_': far_field_scaler.min_,
                'data_min_': far_field_scaler.data_min_,
                'data_max_': far_field_scaler.data_max_,
                'data_range_': far_field_scaler.data_range_,
                'n_features_in_': getattr(far_field_scaler, 'n_features_in_',
                                         far_field_scaler.scale_.shape[0] if hasattr(far_field_scaler,
                                                                                     'scale_') else 0),
                'n_samples_seen_': getattr(far_field_scaler, 'n_samples_seen_', 1)
            },
        },
        'X_s_train_shape': X_s_train.shape,
        'y_s_train_shape': y_s_train.shape,
        'X_f_train_shape': X_f_train.shape,
        'y_f_train_shape': y_f_train.shape,
        'device': str(device),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'data_samples': X_original.shape[0]
    }

    # 保存训练信息
    info_file_path = model_save_path.replace('.pth', '_info.npy')
    try:
        np.save(info_file_path, training_info)
        print(f"✅ 训练信息保存成功: {info_file_path}")
    except Exception as e:
        print(f"❌ 训练信息保存失败: {e}")
        return None

    # 保存到文件
    np.save(model_save_path.replace('.pth', '_info.npy'), training_info)
    print("✅ 多输出模型训练和保存完成！")
    # 保存模型权重
    try:
        if hasattr(system, 'multi_generator') and system.multi_generator is not None:
            torch.save(system.multi_generator.state_dict(), model_save_path)
            print(f"✅ 模型权重保存成功: {model_save_path}")
        else:
            print("❌ 模型未正确初始化，无法保存")
            return None
    except Exception as e:
        print(f"❌ 模型权重保存失败: {e}")
        return None

    print("✅ 多输出模型训练和保存完成！")
    return training_info



if __name__ == "__main__":
    print("贴片天线GAN模型训练系统")
    print("=" * 70)

    # 训练模型
    create_antenna_data = 0  # 根据需要调整数据量

    # model_save_path = 'models/trained_gan_model.pth'
    # train_gan_model(create_antenna_data, model_save_path)
    # train_gan_model(create_antenna_data, model_save_path, 'far_field')

    # 训练多输出模型
    # multi_model_save_path = 'models/multi_output_trained_model.pth'
    # train_multi_output_model(create_antenna_data, multi_model_save_path)

    try:
        # 训练多输出模型
        multi_model_save_path = 'models/multi_output_trained_model.pth'
        result = train_multi_output_model(create_antenna_data, multi_model_save_path)

        if result is not None:
            print("\n" + "=" * 70)
            print("✅ 模型训练和保存成功！")

            # 验证模型文件是否存在
            if os.path.exists(multi_model_save_path):
                model_size = os.path.getsize(multi_model_save_path)
                print(f"💾 模型文件大小: {model_size / (1024 * 1024):.2f} MB")
            else:
                print("❌ 模型文件未找到")

            info_file = multi_model_save_path.replace('.pth', '_info.npy')
            if os.path.exists(info_file):
                info_size = os.path.getsize(info_file)
                print(f"📋 训练信息文件大小: {info_size / 1024:.2f} KB")
            else:
                print("❌ 训练信息文件未找到")
        else:
            print("\n" + "=" * 70)
            print("❌ 模型训练或保存失败！")

    except Exception as e:
        print(f"\n💥 训练过程中出现异常: {e}")
        import traceback

        traceback.print_exc()
    print("\n" + "=" * 70)
    print("模型训练完成！")
    print("=" * 70)

    # 训练逆向模型
    try:
        inverse_model_save_path = 'models/inverse_trained_model.pth'
        result = train_inverse_model(create_antenna_data, model_save_path=inverse_model_save_path)

        if result is not None:
            print("\n" + "=" * 70)
            print("✅ 逆向模型训练和保存成功！")

            # 验证模型文件是否存在
            if os.path.exists(inverse_model_save_path):
                model_size = os.path.getsize(inverse_model_save_path)
                print(f"💾 模型文件大小: {model_size / (1024 * 1024):.2f} MB")
            else:
                print("❌ 模型文件未找到")

            info_file = inverse_model_save_path.replace('.pth', '_info.npy')
            if os.path.exists(info_file):
                info_size = os.path.getsize(info_file)
                print(f"📋 训练信息文件大小: {info_size / 1024:.2f} KB")
            else:
                print("❌ 训练信息文件未找到")
        else:
            print("\n" + "=" * 70)
            print("❌ 逆向模型训练或保存失败！")

    except Exception as e:
        print(f"\n💥 训练过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
    print("\n" + "=" * 70)
    print("反向模型训练完成！")
    print("=" * 70)

