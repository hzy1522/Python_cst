"""
贴片天线设计系统 - 模型使用模块
Patch Antenna Design System - Model Usage Module
"""

import sys
import os
import numpy as np
import torch
import pandas as pd
import time
from matplotlib import pyplot as plt

import plotly.graph_objects as go
import plotly.offline as pyo
from scipy.interpolate import griddata
from performance_error_evalution import *

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
        # 加载预处理器状态
        print(f"✅ 训练信息加载完成！")
        print(f"   训练时间: {training_info.get('timestamp', '未知')}")
        print(f"   训练样本数: {training_info.get('data_samples', '未知')}")
        print(f"   训练设备: {training_info.get('device', '未知')}")

        if 'scalers' in training_info:
            # 重建 input_scaler (system.scaler) - StandardScaler
            from sklearn.preprocessing import StandardScaler, MinMaxScaler
            system.scaler = StandardScaler()
            input_scaler_data = training_info['scalers']['input_scaler']
            system.scaler.scale_ = input_scaler_data['scale_']
            system.scaler.mean_ = input_scaler_data['mean_']
            system.scaler.var_ = input_scaler_data['var_']
            if 'n_features_in_' in input_scaler_data and input_scaler_data['n_features_in_'] is not None:
                system.scaler.n_features_in_ = input_scaler_data['n_features_in_']
            else:
                system.scaler.n_features_in_ = len(input_scaler_data['scale_']) if 'scale_' in input_scaler_data else 0
            if 'n_samples_seen_' in input_scaler_data and input_scaler_data['n_samples_seen_'] is not None:
                system.scaler.n_samples_seen_ = input_scaler_data['n_samples_seen_']
            else:
                system.scaler.n_samples_seen_ = 1

            # 重建 target_scaler - MinMaxScaler
            system.target_scaler = MinMaxScaler()
            target_scaler_data = training_info['scalers']['target_scaler']
            system.target_scaler.scale_ = target_scaler_data['scale_']
            system.target_scaler.min_ = target_scaler_data['min_']
            system.target_scaler.data_min_ = target_scaler_data['data_min_']
            system.target_scaler.data_max_ = target_scaler_data['data_max_']
            system.target_scaler.data_range_ = target_scaler_data['data_range_']
            if 'n_features_in_' in target_scaler_data and target_scaler_data['n_features_in_'] is not None:
                system.target_scaler.n_features_in_ = target_scaler_data['n_features_in_']
            else:
                system.target_scaler.n_features_in_ = len(target_scaler_data['scale_']) if 'scale_' in target_scaler_data else 0
            if 'n_samples_seen_' in target_scaler_data and target_scaler_data['n_samples_seen_'] is not None:
                system.target_scaler.n_samples_seen_ = target_scaler_data['n_samples_seen_']
            else:
                system.target_scaler.n_samples_seen_ = 1

            # 更新检查函数以适配不同类型的缩放器
            def _check_scalers_ready(system):
                """检查预处理器是否已就绪"""
                try:
                    # 检查 scaler (StandardScaler) 是否已拟合
                    _ = system.scaler.scale_
                    _ = system.scaler.mean_

                    # 检查 target_scaler (MinMaxScaler) 是否已拟合
                    _ = system.target_scaler.scale_
                    _ = system.target_scaler.min_

                    return True
                except AttributeError:
                    return False

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
    generation_start_time = time.time()  # 记录生成开始时间
    try:
        generated_designs, generated_performances = system.generate_antenna_designs(
            target_performances, num_samples=20
        )
        generation_time = time.time() - generation_start_time  # 计算生成耗时
        print(f"✅ GAN生成完成，耗时: {generation_time:.2f}秒")
        # 添加空的history字典用于可视化
        history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'adversarial_loss': [],
            'performance_loss': []
        }
        # 可视化生成结果
        system.visualize_gan_results(history, generated_designs, generated_performances)

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

def use_trained_gan_model_prediction_results(model_info_path='models/trained_gan_model_info.npy',
                         patch_lengths=None,
                         patch_widths=None,
                         gan_generator_path='models/gan_generator.pth',
                         forward_gan_generator_path='models/forward_gan_generator.pth'):
    """
    使用已训练的GAN模型生成天线设计

    Args:
        model_info_path: 模型信息文件路径
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
        # 加载预处理器状态
        print(f"✅ 训练信息加载完成！")
        print(f"   训练时间: {training_info.get('timestamp', '未知')}")
        print(f"   训练样本数: {training_info.get('data_samples', '未知')}")
        print(f"   训练设备: {training_info.get('device', '未知')}")

        if 'scalers' in training_info:
            # 重建 input_scaler (system.scaler) - StandardScaler
            from sklearn.preprocessing import StandardScaler, MinMaxScaler
            system.scaler = StandardScaler()
            input_scaler_data = training_info['scalers']['input_scaler']
            system.scaler.scale_ = input_scaler_data['scale_']
            system.scaler.mean_ = input_scaler_data['mean_']
            system.scaler.var_ = input_scaler_data['var_']
            if 'n_features_in_' in input_scaler_data and input_scaler_data['n_features_in_'] is not None:
                system.scaler.n_features_in_ = input_scaler_data['n_features_in_']
            else:
                system.scaler.n_features_in_ = len(input_scaler_data['scale_']) if 'scale_' in input_scaler_data else 0
            if 'n_samples_seen_' in input_scaler_data and input_scaler_data['n_samples_seen_'] is not None:
                system.scaler.n_samples_seen_ = input_scaler_data['n_samples_seen_']
            else:
                system.scaler.n_samples_seen_ = 1

            # 重建 target_scaler - MinMaxScaler
            system.target_scaler = MinMaxScaler()
            target_scaler_data = training_info['scalers']['target_scaler']
            system.target_scaler.scale_ = target_scaler_data['scale_']
            system.target_scaler.min_ = target_scaler_data['min_']
            system.target_scaler.data_min_ = target_scaler_data['data_min_']
            system.target_scaler.data_max_ = target_scaler_data['data_max_']
            system.target_scaler.data_range_ = target_scaler_data['data_range_']
            if 'n_features_in_' in target_scaler_data and target_scaler_data['n_features_in_'] is not None:
                system.target_scaler.n_features_in_ = target_scaler_data['n_features_in_']
            else:
                system.target_scaler.n_features_in_ = len(target_scaler_data['scale_']) if 'scale_' in target_scaler_data else 0
            if 'n_samples_seen_' in target_scaler_data and target_scaler_data['n_samples_seen_'] is not None:
                system.target_scaler.n_samples_seen_ = target_scaler_data['n_samples_seen_']
            else:
                system.target_scaler.n_samples_seen_ = 1

            # 更新检查函数以适配不同类型的缩放器
            def _check_scalers_ready(system):
                """检查预处理器是否已就绪"""
                try:
                    # 检查 scaler (StandardScaler) 是否已拟合
                    _ = system.scaler.scale_
                    _ = system.scaler.mean_

                    # 检查 target_scaler (MinMaxScaler) 是否已拟合
                    _ = system.target_scaler.scale_
                    _ = system.target_scaler.min_

                    return True
                except AttributeError:
                    return False

    else:
        print(f"⚠️  未找到训练信息文件，使用默认配置")
        training_info = {}

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

    hfss_results = []

    if patch_lengths is None or patch_widths is None:
        gan_data = pd.read_csv('results/gan_generated_designs.csv')
        patch_lengths = gan_data['patch_length'].values
        patch_widths = gan_data['patch_width'].values
        print(f"从GAN生成结果中读取了 {len(patch_lengths)} 行数据")
    else:
        # 确保单个值也被转换为数组形式
        if not isinstance(patch_lengths, (list, np.ndarray)):
            patch_lengths = [patch_lengths]
        if not isinstance(patch_widths, (list, np.ndarray)):
            patch_widths = [patch_widths]
        # 转换为numpy数组
        patch_lengths = np.array(patch_lengths)
        patch_widths = np.array(patch_widths)

    # 5. 使用HFSS计算所有生成天线的性能结果
    print(f"\n3. 使用HFSS验证所有生成的天线设计...")

    for i in range(len(patch_lengths)):
        design = np.zeros(2)  # 初始化design数组
        design[0] = patch_lengths[i]
        design[1] = patch_widths[i]
        print(f"\n验证设计 {i + 1}/{len(patch_lengths)}: 长度={design[0]:.2f}mm, 宽度={design[1]:.2f}mm")

        # HFSS仿真参数设置
        antenna_params = {
            "unit": "GHz",
            "patch_length": float(design[0]),
            "patch_width": float(design[1]),
            "patch_name": "Patch",
            "freq_step": "0.01GHz",
            "num_of_freq_points": 201,
            "start_frequency": 2,
            "stop_frequency": 3,
            "center_frequency": 2.5,
            "sweep_type": "Interpolating",
            "sub_length": 50,
            "sub_width": 60,
            "sub_high": 1.575,
            "feed_r1": 0.5,
            "feed_h": 1.575,
            "feed_center": 6.3,
            "lumpedport_r": 1.5,
            "lumpedport_D": 2.3 / 2,
        }
        # 确保模型处于评估模式
        if system.forward_generator is not None:
            system.forward_generator.eval()
        if system.performance_predictor is not None:
            system.performance_predictor.eval()

        # s11_curve_predict, s11_min_predict, freq_at_s11_min_predict, far_field_gain_predict = system.predict_s11_from_dimensions(
        #     design[0], design[1])
        prediction_start_time = time.time()
        s11_curve_predict = system.predict_s11_from_dimensions(design[0], design[1])
        prediction_time = time.time() - prediction_start_time  # 计算预测耗时
        print(f"  预测耗时: {prediction_time:.2f}秒")

        # 查找S11最小值及其对应的频率点
        s11_min_predict = np.min(s11_curve_predict)
        min_index = np.argmin(s11_curve_predict)
        freq_points = np.linspace(2.0, 3.0, len(s11_curve_predict))  # 201个频率点从2.0GHz到3.0GHz
        freq_at_s11_min_predict = freq_points[min_index]

        # print(f"预测的S11最小值: {s11_min_predict:.2f}dB")
        # print(f"对应的频率: {freq_at_s11_min_predict:.2f}GHz")

        # 调用HFSS计算
        train_model = False
        try:
            success, freq_at_s11_min, far_field_gain, s11_min, output_file, output_file_farfield = calculate_from_hfss_py(
                antenna_params, train_model
            )

            if success and output_file:
                print(f"  HFSS计算成功!")
                print(f"  实际性能: S11={s11_min:.2f}dB, 频率={freq_at_s11_min:.2f}GHz, 增益={far_field_gain:.2f}dBi")
                print(f"  模型预测性能: S11={s11_min_predict:.2f}dB, "
                      f"频率={freq_at_s11_min_predict:.2f}GHz, "
                      # f"增益={far_field_gain_predict:.2f}dBi"
                      )

                # 保存结果
                hfss_results.append({
                    'design_index': i,
                    'patch_length': design[0],
                    'patch_width': design[1],
                    'predicted_s11': s11_min_predict,
                    'predicted_freq': freq_at_s11_min_predict,
                    # 'predicted_gain': far_field_gain_predict,
                    'actual_s11': s11_min,
                    'actual_freq': freq_at_s11_min,
                    'actual_gain': far_field_gain,
                    'output_file': output_file
                })

                # 绘制S11对比图
                system.plot_s11_comparison_advanced(
                    float(design[0]), float(design[1]),
                    output_file, frequency_column=0, s11_column=1,
                    predict_s11_curve=s11_curve_predict
                )

            else:
                print(f"  HFSS计算失败")
                hfss_results.append({
                    'design_index': i,
                    'patch_length': design[0],
                    'patch_width': design[1],
                    'predicted_s11': s11_min_predict,
                    'predicted_freq': freq_at_s11_min_predict,
                    # 'predicted_gain': far_field_gain_predict,
                    'actual_s11': None,
                    'actual_freq': None,
                    'actual_gain': None,
                    'output_file': None
                })
        except Exception as e:
            print(f"  HFSS计算出错: {e}")
            hfss_results.append({
                'design_index': i,
                'patch_length': design[0],
                'patch_width': design[1],
                'predicted_s11': s11_min_predict,
                'predicted_freq': freq_at_s11_min_predict,
                # 'predicted_gain': far_field_gain_predict,
                'actual_s11': None,
                'actual_freq': None,
                'actual_gain': None,
                'output_file': None
            })

    # 6. 保存HFSS验证结果
    if hfss_results:
        hfss_df = pd.DataFrame(hfss_results)
        hfss_csv_path = 'results/hfss_validation_results.csv'
        hfss_df.to_csv(hfss_csv_path, index=False)
        print(f"\nHFSS验证结果已保存到 {hfss_csv_path}")
        return (s11_min_predict,
                freq_at_s11_min_predict,
                s11_curve_predict,
                s11_min,
                freq_at_s11_min,
                far_field_gain)

def extract_gain_matrix_from_csv(csv_file_path, gain_column_name='Gain_dB_matrix'):
    """
    从CSV文件中提取增益矩阵数据并展平为一维数组
    适用于增益数据存储在单个单元格中的情况

    Args:
        csv_file_path: CSV文件路径
        gain_column_name: 增益数据列名，默认为'Gain_dB_matrix'

    Returns:
        flattened_gain_data: 展平后的增益数据一维数组
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)

        # 检查指定列是否存在
        if gain_column_name not in df.columns:
            raise ValueError(f"列 '{gain_column_name}' 不存在于CSV文件中")

        # 获取第一个单元格的数据（假设所有行的增益矩阵相同或只需要第一行）
        gain_data_str = df[gain_column_name].iloc[0]

        # 解析存储在单个单元格中的二维矩阵数据
        # 根据实际数据格式选择合适的解析方法

        # 方法1: 如果数据是Python列表格式的字符串
        if gain_data_str.startswith('[') and gain_data_str.endswith(']'):
            # 移除最外层括号
            inner_data = gain_data_str[1:-1]

            # 处理多维数组格式
            if '],' in inner_data:  # 二维数组
                # 分割行
                rows = inner_data.split('],')
                matrix_data = []
                for i, row in enumerate(rows):
                    # 处理最后一行可能缺少括号的情况
                    if i == len(rows) - 1 and not row.endswith(']'):
                        row += ']'
                    elif i < len(rows) - 1 and not row.endswith(']'):
                        row += ']'

                    # 提取数字
                    numbers_str = row.strip('[] ')
                    if numbers_str:
                        numbers = [float(x.strip()) for x in numbers_str.split(',') if x.strip()]
                        matrix_data.append(numbers)

                # 转换为numpy数组并展平
                gain_matrix = np.array(matrix_data)
                flattened_gain_data = gain_matrix.flatten()
            else:
                # 一维数组
                numbers = [float(x.strip()) for x in inner_data.split(',') if x.strip()]
                flattened_gain_data = np.array(numbers)

        # 方法2: 如果数据是其他格式（如JSON），可以添加相应的解析逻辑
        else:
            # 尝试直接转换为浮点数数组（适用于简单格式）
            flattened_gain_data = np.array([float(x) for x in gain_data_str.split(',') if x.strip()])

        print(f"✅ 成功提取增益矩阵数据，展平后维度: {flattened_gain_data.shape}")
        return flattened_gain_data

    except Exception as e:
        print(f"❌ 提取增益矩阵数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_gain_matrix_string(matrix_string):
    """
    解析增益矩阵字符串为numpy数组

    Args:
        matrix_string: 包含矩阵数据的字符串

    Returns:
        numpy数组形式的矩阵数据
    """
    try:
        # 清理字符串
        matrix_string = matrix_string.strip()

        # 如果是嵌套列表格式
        if matrix_string.startswith('[') and matrix_string.endswith(']'):
            # 尝试使用eval（注意：仅在可信数据上使用）
            # 或者使用更安全的解析方法
            matrix_data = eval(matrix_string)
            return np.array(matrix_data)
        else:
            # 尝试按逗号分割并转换为数组
            data_list = [float(x) for x in matrix_string.split(',') if x.strip()]
            return np.array(data_list)

    except Exception as e:
        print(f"解析矩阵字符串时出错: {e}")
        return None

def load_target_specs_from_csv(csv_file_path):
    """
    从CSV文件读取S参数最小值、增益、频率以及201个S参数点，保存成target_specs格式

    Args:
        csv_file_path: CSV文件路径

    Returns:
        target_specs: 204维的目标性能参数列表
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    target_specs = []

    # 遍历每一行数据
    for index, row in df.iterrows():
        # 提取主要性能指标
        s11_min = row['_最小值']  # S11最小值
        freq = row['Freq [GHz]']  # 对应频率
        gain = row['Gain_dB']   # 远区场增益
        patch_length = row['patch_length']
        patch_width = row['patch_width']

        # 提取201个S参数点
        # 方法1: 如果列名是频率值（如2.000, 2.010, ...）
        s_parameters = []
        freq_points = np.linspace(2.0, 3.0, 201)  # 生成201个频率点
        for freq_point in freq_points:
            col_name = f"{freq_point:.3f}"  # 根据实际列名格式调整
            if col_name in row:
                s_parameters.append(row[col_name])
            else:
                # 如果找不到对应列，可以使用默认值或插值
                s_parameters.append(0.0)  # 使用默认值

        # 方法2: 如果S参数是连续的列（如s11_1, s11_2, ..., s11_201）
        # s_parameters = [row[f's11_{i}'] for i in range(1, 202)]

        # 构造204维向量：[S11最小值, 对应频率, 远区场增益, 201个S参数点]
        # target_spec = [s11_min, freq, gain] + s_parameters
        target_spec = s_parameters
        target_specs.append(target_spec)

    return target_specs, patch_length, patch_width

def use_multi_output_model(model_path, patch_length, patch_width):
    """
    使用训练好的多输出模型进行预测

    Args:
        model_path: 模型保存路径
        patch_length: 贴片长度(mm)
        patch_width: 贴片宽度(mm)

    Returns:
        dict: 包含预测结果的字典
    """
    # 加载模型和相关信息
    # 在 use_multi_output_model 函数中添加文件检查
    model_path = 'models/multi_output_trained_model.pth'
    info_path = model_path.replace('.pth', '_info.npy')

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"⚠️  模型文件不存在: {model_path}")
        return {
            'input_dimensions': {'length': patch_length, 'width': patch_width},
            'predicted_s_parameters': None,
            'predicted_far_field_pattern': None,
            'error': '模型文件不存在'
        }

    if not os.path.exists(info_path):
        raise FileNotFoundError(f"未找到模型信息文件: {info_path}")

    training_info = np.load(info_path, allow_pickle=True).item()

    # 初始化系统
    system = PatchAntennaDesignSystem()
    device = get_device()
    system.device = device

    # 恢复标准化器状态
    if 'scalers' in training_info:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        # 重建 input_scaler (StandardScaler)
        system.scaler = StandardScaler()
        input_scaler_data = training_info['scalers']['input_scaler']
        system.scaler.scale_ = input_scaler_data['scale_']
        system.scaler.mean_ = input_scaler_data['mean_']
        system.scaler.var_ = input_scaler_data['var_']
        system.scaler.n_features_in_ = input_scaler_data.get('n_features_in_',
                                   len(input_scaler_data['scale_']) if 'scale_' in input_scaler_data else 0)
        system.scaler.n_samples_seen_ = input_scaler_data.get('n_samples_seen_', 1)

        # 检查是否存在 s_params_scaler，如果不存在则使用默认的 target_scaler
        if 's_params_scaler' in training_info['scalers']:
            # 重建 s_params_scaler (MinMaxScaler)
            system.target_scaler = MinMaxScaler()
            s_params_scaler_data = training_info['scalers']['s_params_scaler']
            system.target_scaler.scale_ = s_params_scaler_data['scale_']
            system.target_scaler.min_ = s_params_scaler_data['min_']
            system.target_scaler.data_min_ = s_params_scaler_data['data_min_']
            system.target_scaler.data_max_ = s_params_scaler_data['data_max_']
            system.target_scaler.data_range_ = s_params_scaler_data['data_range_']
            system.target_scaler.n_features_in_ = s_params_scaler_data.get('n_features_in_',
                                            len(s_params_scaler_data['scale_']) if 'scale_' in s_params_scaler_data else 0)
            system.target_scaler.n_samples_seen_ = s_params_scaler_data.get('n_samples_seen_', 1)
        else:
            # 使用旧版本的 target_scaler
            system.target_scaler = MinMaxScaler()
            target_scaler_data = training_info['scalers']['target_scaler']
            system.target_scaler.scale_ = target_scaler_data['scale_']
            system.target_scaler.min_ = target_scaler_data['min_']
            system.target_scaler.data_min_ = target_scaler_data['data_min_']
            system.target_scaler.data_max_ = target_scaler_data['data_max_']
            system.target_scaler.data_range_ = target_scaler_data['data_range_']
            system.target_scaler.n_features_in_ = target_scaler_data.get('n_features_in_',
                                            len(target_scaler_data['scale_']) if 'scale_' in target_scaler_data else 0)
            system.target_scaler.n_samples_seen_ = target_scaler_data.get('n_samples_seen_', 1)

        # 检查是否存在 far_field_scaler
        if 'far_field_scaler' in training_info['scalers']:
            # 重建 far_field_scaler (MinMaxScaler)
            system.far_field_scaler = MinMaxScaler()
            far_field_scaler_data = training_info['scalers']['far_field_scaler']
            system.far_field_scaler.scale_ = far_field_scaler_data['scale_']
            system.far_field_scaler.min_ = far_field_scaler_data['min_']
            system.far_field_scaler.data_min_ = far_field_scaler_data['data_min_']
            system.far_field_scaler.data_max_ = far_field_scaler_data['data_max_']
            system.far_field_scaler.data_range_ = far_field_scaler_data['data_range_']
            system.far_field_scaler.n_features_in_ = far_field_scaler_data.get('n_features_in_',
                                                 len(far_field_scaler_data['scale_']) if 'scale_' in far_field_scaler_data else 0)
            system.far_field_scaler.n_samples_seen_ = far_field_scaler_data.get('n_samples_seen_', 1)
        else:
            # 如果不存在远区场标准化器，创建一个默认的
            from sklearn.preprocessing import MinMaxScaler
            system.far_field_scaler = MinMaxScaler()

    # 获取远区场维度信息
    actual_far_field_dim = training_info.get('actual_far_field_dim',
                                           training_info.get('y_f_train_shape', [0, 2701])[1])
    print(f"使用远区场输出维度: {actual_far_field_dim}")

    # 进行预测
    try:
        print(f"🔍 开始预测: 长度={patch_length}mm, 宽度={patch_width}mm")
        prediction_start_time = time.time()
        # 在调用预测方法前确保模型已正确初始化
        s_params_pred, far_field_pred = system.predict_s_params_and_far_field(
            patch_length, patch_width, far_field_dim=actual_far_field_dim
        )
        prediction_time = time.time() - prediction_start_time  # 计算预测耗时
        print(f"S参数&远区场预测耗时: {prediction_time:.2f}秒")
        print(f"📊 预测结果:")
        print(f"   S参数预测: {'成功' if s_params_pred is not None else '失败'}")
        print(f"   远区场预测: {'成功' if far_field_pred is not None else '失败'}")

        if far_field_pred is not None:
            print(f"   远区场数据维度: {far_field_pred.shape}")
            print(f"   远区场数据范围: [{np.min(far_field_pred):.2f}, {np.max(far_field_pred):.2f}]")

        # 如果需要，可以将远区场数据重塑为二维矩阵
        # 假设标准的 theta(181) x phi(361) 网格
        far_field_matrix = None
        if far_field_pred is not None:
            # print(f"len(far_field_pred) == {len(far_field_pred) } ")
            if len(far_field_pred) == 37 * 73:
                far_field_matrix = far_field_pred.reshape(37, 73)
                print(f"   远区场矩阵维度: {far_field_matrix.shape}")
            else:
                far_field_matrix = far_field_pred
                print(f"   远区场数据未 reshape，保持原维度: {far_field_pred.shape}")

        # 保存预测的远区场数据
        if far_field_matrix is not None:
            # 创建theta和phi的坐标数组
            theta_values = np.arange(0, 181, 5)  # 0到180度，每5度一个间隔
            phi_values = np.arange(-180, 181, 5)  # -180到180度，每5度一个间隔

            # 确保数组长度与矩阵维度匹配
            if far_field_matrix.shape[0] == len(theta_values) and far_field_matrix.shape[1] == len(phi_values):
                # 创建DataFrame
                data_list = []
                for i, theta in enumerate(theta_values):
                    for j, phi in enumerate(phi_values):
                        data_list.append({
                            'Theta(deg)': theta,
                            'Phi(deg)': phi,
                            'Gain_dB': far_field_matrix[i, j]
                        })

                far_field_df = pd.DataFrame(data_list)
                far_field_csv_path = f'results/predicted_far_field_{patch_length}x{patch_width}.csv'
                far_field_df.to_csv(far_field_csv_path, index=False)
                print(f"📊 预测远区场数据已保存到: {far_field_csv_path}")
                print("📊 预测远区场数据已绘制3D图: ./results/far_field_3d_predicted.html")
                plot_3d_radiation_pattern_from_csv(far_field_csv_path, './results/far_field_3d_predicted.html')
            else:
                print(f"⚠️  远区场矩阵维度与预期不匹配: 期望({len(theta_values)}, {len(phi_values)}), 实际{far_field_matrix.shape}")

        result = {
            'input_dimensions': {'length': patch_length, 'width': patch_width},
            'predicted_s_parameters': s_params_pred,
            'predicted_far_field_pattern': far_field_pred,
            'predicted_far_field_matrix': far_field_matrix
        }

        print(f"✅ 预测完成!")
        print(f"   S参数预测维度: {s_params_pred.shape if s_params_pred is not None else 'None'}")
        print(f"   远区场预测维度: {far_field_pred.shape if far_field_pred is not None else 'None'}")

        #调用hfss计算
        hfss_results = []
        # 查找S11最小值及其对应的频率点
        s11_min_predict = np.min(s_params_pred)
        min_index = np.argmin(s_params_pred)
        freq_points = np.linspace(2.0, 3.0, len(s_params_pred))  # 201个频率点从2.0GHz到3.0GHz
        freq_at_s11_min_predict = freq_points[min_index]
        # HFSS仿真参数设置
        antenna_params = {
            "unit": "GHz",
            "patch_length": patch_length,
            "patch_width": patch_width,
            "patch_name": "Patch",
            "freq_step": "0.01GHz",
            "num_of_freq_points": 201,
            "start_frequency": 2,
            "stop_frequency": 3,
            "center_frequency": 2.5,
            "sweep_type": "Interpolating",
            "sub_length": 50,
            "sub_width": 60,
            "sub_high": 1.575,
            "feed_r1": 0.5,
            "feed_h": 1.575,
            "feed_center": 6.3,
            "lumpedport_r": 1.5,
            "lumpedport_D": 2.3 / 2,
        }
        # 调用HFSS计算
        train_model = False
        try:
            success, freq_at_s11_min, far_field_gain, s11_min, output_file, output_file_farfield = calculate_from_hfss_py(
                antenna_params, train_model
            )

            plot_3d_radiation_pattern_from_csv(output_file_farfield, './results/far_field_3d_hfss.html')

            if success and output_file:
                print(f"  HFSS计算成功!")
                print(f"  实际性能: S11={s11_min:.2f}dB, 频率={freq_at_s11_min:.2f}GHz, 增益={far_field_gain:.2f}dBi")
                print(f"  模型预测性能: S11={s11_min_predict:.2f}dB, "
                      f"频率={freq_at_s11_min_predict:.2f}GHz, "
                      f"增益={np.max(far_field_pred):.2f}dBi"
                      )

                # 保存结果
                hfss_results.append({
                    # 'design_index': i,
                    'patch_length': patch_length,
                    'patch_width': patch_width,
                    'predicted_s11': s11_min_predict,
                    'predicted_freq': freq_at_s11_min_predict,
                    'predicted_gain': far_field_pred,
                    'actual_s11': s11_min,
                    'actual_freq': freq_at_s11_min,
                    'actual_gain': far_field_gain,
                    'output_file': output_file
                })

                # 绘制S11对比图
                system.plot_s11_comparison_advanced(
                    float(patch_length), float(patch_width),
                    output_file, frequency_column=0, s11_column=1,
                    predict_s11_curve=s_params_pred
                )

                # 如果有预测结果和实际结果，进行评估
                if success and output_file and output_file_farfield and s_params_pred is not None:
                    try:
                        # 读取HFSS的实际数据
                        print(f"  读取HFSS的实际数据: {output_file}")
                        actual_s_params_df = pd.read_csv(output_file)
                        if 'dB(S(ground_T1,ground_T1)) []' in actual_s_params_df.columns:
                            actual_s_params = actual_s_params_df['dB(S(ground_T1,ground_T1)) []'].values
                        else:
                            actual_s_rameters = actual_s_params_df.values.flatten()

                        # 读取实际的远区场数据
                        print(f"  读取HFSS的实际远区场数据: {output_file_farfield}")
                        actual_far_field_df = pd.read_csv(output_file_farfield)
                        if 'Gain_dB' in actual_far_field_df.columns:
                            actual_far_field = actual_far_field_df['Gain_dB'].values
                        else:
                            actual_far_rametersfield = actual_far_field_df.values.flatten()

                        # 评估S参数

                        s_params_metrics = evaluate_s_parameters(s_params_pred, actual_s_params)
                        print("📈 S参数评估结果:")
                        for metric, value in s_params_metrics.items():
                            if metric == 'mape':
                                print(f"   {metric.upper()}: {value:.2f}%")
                            elif metric == 'ssim':
                                print(f"   {metric.upper()}: {value:.4f}")
                            else:
                                print(f"   {metric.upper()}: {value:.6f}")

                        # 评估远区场
                        far_field_metrics = evaluate_far_field_pattern(far_field_pred, actual_far_field)
                        print("📊 远区场评估结果:")
                        for metric, value in far_field_metrics.items():
                            if metric == 'mape':
                                print(f"   {metric.upper()}: {value:.2f}%")
                            elif metric == 'ssim':
                                print(f"   {metric.upper()}: {value:.4f}")
                            else:
                                print(f"   {metric.upper()}: {value:.6f}")

                        # 将评估结果添加到返回结果中
                        result['evaluation'] = {
                            's_parameters': s_params_metrics,
                            'far_field': far_field_metrics
                        }

                    except Exception as e:
                        print(f"评估过程出错: {e}")

            else:
                print(f"  HFSS计算失败")
                hfss_results.append({
                    # 'design_index': i,
                    'patch_length': patch_length,
                    'patch_width': patch_width,
                    'predicted_s11': s11_min_predict,
                    'predicted_freq': freq_at_s11_min_predict,
                    'predicted_gain': far_field_pred,
                    'actual_s11': None,
                    'actual_freq': None,
                    'actual_gain': None,
                    'output_file': None
                })
        except Exception as e:
            print(f"  HFSS计算出错: {e}")
            hfss_results.append({
                # 'design_index': i,
                    'patch_length': patch_length,
                    'patch_width': patch_width,
                'predicted_s11': s11_min_predict,
                'predicted_freq': freq_at_s11_min_predict,
                'predicted_gain': far_field_pred,
                'actual_s11': None,
                'actual_freq': None,
                'actual_gain': None,
                'output_file': None
            })

        # 6. 保存HFSS验证结果
        if hfss_results:
            hfss_df = pd.DataFrame(hfss_results)
            hfss_csv_path = 'results/hfss_validation_results.csv'
            hfss_df.to_csv(hfss_csv_path, index=False)
            print(f"\nHFSS验证结果已保存到 {hfss_csv_path}")

        # # 绘制预测结果
        # plot_predictions(s_params_pred, far_fiel_matrix, patch_length, patch_width)

        return result

    except Exception as e:
        print(f"❌ 预测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return {
            'input_dimensions': {'length': patch_length, 'width': patch_width},
            'predicted_s_parameters': None,
            'predicted_far_field_pattern': None,
            'error': str(e)
        }

def use_inverse_model(model_path, s_parameters=None, far_field_pattern=None):
    """
    使用训练好的逆向模型进行预测（根据性能参数预测天线尺寸）

    Args:
        model_path: 模型保存路径
        s_parameters: S参数曲线数据（201维）
        far_field_pattern: 远区场方向图数据（展平后的一维数组）

    Returns:
        dict: 包含预测结果的字典
    """
    # 构造完整模型和信息文件路径
    model_weights_path = model_path
    info_path = model_path.replace('.pth', '_info.npy')

    # 检查文件是否存在
    if not os.path.exists(model_weights_path):
        print(f"⚠️  模型权重文件不存在: {model_weights_path}")
        return {
            'input_performance': {'s_parameters': s_parameters, 'far_field_pattern': far_field_pattern},
            'predicted_dimensions': None,
            'error': '模型权重文件不存在'
        }

    if not os.path.exists(info_path):
        print(f"⚠️  模型信息文件不存在: {info_path}")
        return {
            'input_performance': {'s_parameters': s_parameters, 'far_field_pattern': far_field_pattern},
            'predicted_dimensions': None,
            'error': '模型信息文件不存在'
        }

    # 加载训练信息
    training_info = np.load(info_path, allow_pickle=True).item()

    # 初始化系统
    system = PatchAntennaDesignSystem()
    device = get_device()
    system.device = device

    # 恢复标准化器状态
    if 'scalers' in training_info:
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        # 重建 input_scaler (用于输入特征：S参数+远区场)
        system.inverse_input_scaler = MinMaxScaler()
        input_scaler_data = training_info['scalers']['input_scaler']
        system.inverse_input_scaler.scale_ = input_scaler_data['scale_']
        system.inverse_input_scaler.min_ = input_scaler_data['min_']
        system.inverse_input_scaler.data_min_ = input_scaler_data['data_min_']
        system.inverse_input_scaler.data_max_ = input_scaler_data['data_max_']
        system.inverse_input_scaler.data_range_ = input_scaler_data['data_range_']
        system.inverse_input_scaler.n_features_in_ = input_scaler_data.get('n_features_in_',
                                             len(input_scaler_data['scale_']) if 'scale_' in input_scaler_data else 0)
        system.inverse_input_scaler.n_samples_seen_ = input_scaler_data.get('n_samples_seen_', 1)

        # 重建 output_scaler (用于输出目标：天线尺寸)
        system.inverse_output_scaler = MinMaxScaler()
        output_scaler_data = training_info['scalers']['output_scaler']
        system.inverse_output_scaler.scale_ = output_scaler_data['scale_']
        system.inverse_output_scaler.min_ = output_scaler_data['min_']
        system.inverse_output_scaler.data_min_ = output_scaler_data['data_min_']
        system.inverse_output_scaler.data_max_ = output_scaler_data['data_max_']
        system.inverse_output_scaler.data_range_ = output_scaler_data['data_range_']
        system.inverse_output_scaler.n_features_in_ = output_scaler_data.get('n_features_in_',
                                              len(output_scaler_data['scale_']) if 'scale_' in output_scaler_data else 0)
        system.inverse_output_scaler.n_samples_seen_ = output_scaler_data.get('n_samples_seen_', 1)

    # 初始化逆向模型结构
    try:
        # 从训练信息中获取输入维度
        input_dim = training_info.get('X_train_shape', [0, 2911])[1]  # 默认201(S参数)+2710(远区场)
        output_dim = training_info.get('y_train_shape', [0, 2])[1]    # 默认2(长度+宽度)

        print(f"🔍 模型结构: 输入维度={input_dim}, 输出维度={output_dim}")

        # 创建逆向模型
        import torch.nn as nn
        system.inverse_model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        ).to(device)

        # 加载模型权重
        state_dict = torch.load(model_weights_path, map_location=device)
        system.inverse_model.load_state_dict(state_dict)
        system.inverse_model.eval()

        print(f"✅ 成功加载逆向模型: {model_weights_path}")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return {
            'input_performance': {'s_parameters': s_parameters, 'far_field_pattern': far_field_pattern},
            'predicted_dimensions': None,
            'error': f'模型加载失败: {str(e)}'
        }

    # 数据预处理和预测
    try:
        print(f"🔍 开始逆向预测...")

        # 检查输入数据
        if s_parameters is None or far_field_pattern is None:
            print("⚠️  输入数据为空，使用默认示例数据")
            # 生成示例数据用于演示
            s_parameters = np.random.uniform(-40, 0, 201)  # 201个S参数点
            far_field_pattern = np.random.uniform(0, 10, 37*73)  # 示例远区场数据

        # 确保数据维度正确
        if len(s_parameters) != 201:
            print(f"⚠️  S参数维度不匹配: 期望201，实际{len(s_parameters)}")
            # 尝试调整维度
            if len(s_parameters) > 201:
                s_parameters = s_parameters[:201]
            else:
                # 用最后一个值填充
                padding = np.full(201 - len(s_parameters), s_parameters[-1])
                s_parameters = np.concatenate([s_parameters, padding])

        # 合并输入特征
        input_features = np.concatenate([s_parameters, far_field_pattern])
        print(f"📊 输入特征维度: {input_features.shape}")

        # 标准化输入数据
        input_scaled = system.inverse_input_scaler.transform(input_features.reshape(1, -1))
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)

        # 模型预测
        generation_start_time = time.time()
        with torch.no_grad():
            predicted_scaled = system.inverse_model(input_tensor)
            predicted_scaled = predicted_scaled.cpu().numpy()

        # 反标准化得到实际尺寸
        predicted_dimensions = system.inverse_output_scaler.inverse_transform(predicted_scaled)[0]
        patch_length, patch_width = predicted_dimensions
        generation_time = time.time() - generation_start_time  # 计算生成耗时
        print(f"✅ 逆向预测完成，耗时: {generation_time:.2f} 秒")
        print(f"📊 预测结果:")
        print(f"   预测长度: {patch_length:.2f} mm")
        print(f"   预测宽度: {patch_width:.2f} mm")

        # 保存预测结果
        result = {
            'input_performance': {
                's_parameters': s_parameters,
                'far_field_pattern': far_field_pattern
            },
            'predicted_dimensions': {
                'length': patch_length,
                'width': patch_width
            }
        }

        # 创建结果目录
        if not os.path.exists('results'):
            os.makedirs('results')

        # 保存结果到CSV
        result_df = pd.DataFrame([{
            'predicted_length': patch_length,
            'predicted_width': patch_width,
            's_params_length': len(s_parameters),
            'far_field_length': len(far_field_pattern)
        }])
        result_csv_path = 'results/inverse_model_prediction.csv'
        result_df.to_csv(result_csv_path, index=False)
        print(f"💾 预测结果已保存到: {result_csv_path}")

        print(f"✅ 逆向预测完成!")
        return result

    except Exception as e:
        print(f"❌ 预测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return {
            'input_performance': {'s_parameters': s_parameters, 'far_field_pattern': far_field_pattern},
            'predicted_dimensions': None,
            'error': str(e)
        }


def use_inverse_model_with_target_specs(model_path, target_s11_min=-30, target_frequency=2.45, target_gain=7.0):
    """
    使用逆向模型，根据目标性能指标生成天线尺寸

    Args:
        model_path: 模型路径
        target_s11_min: 目标S11最小值 (dB)
        target_frequency: 目标频率 (GHz)
        target_gain: 目标增益 (dBi)

    Returns:
        dict: 预测结果
    """
    print(f"🔍 根据目标性能生成天线设计:")
    print(f"   目标S11最小值: {target_s11_min} dB")
    print(f"   目标频率: {target_frequency} GHz")
    print(f"   目标增益: {target_gain} dBi")

    # 生成符合目标的示例S参数曲线
    # 创建一个简单的S参数模型：在目标频率处有最小值
    frequencies = np.linspace(2.0, 3.0, 201)
    s_parameters = np.full(201, -10)  # 基础值-10dB

    # 在目标频率附近创建更深的凹陷
    target_idx = int((target_frequency - 2.0) / (3.0 - 2.0) * 200)
    for i in range(max(0, target_idx-10), min(201, target_idx+11)):
        distance = abs(i - target_idx)
        s_parameters[i] = target_s11_min + distance * 2  # 渐变效果

    # 生成示例远区场数据
    far_field_pattern = np.full(37*73, target_gain)

    # 添加一些变化使数据更真实
    far_field_pattern += np.random.normal(0, 0.5, len(far_field_pattern))

    # 调用逆向模型进行预测
    result = use_inverse_model(model_path, s_parameters, far_field_pattern)

    if result['predicted_dimensions']:
        length = result['predicted_dimensions']['length']
        width = result['predicted_dimensions']['width']
        print(f"🎯 推荐天线尺寸: {length:.2f} × {width:.2f} mm")

        # 可选：使用正向模型验证预测结果
        print(f"🔄 验证预测结果...")
        try:
            forward_result = use_multi_output_model('models/multi_output_trained_model.pth', length, width)
            if forward_result['predicted_s_parameters'] is not None:
                predicted_s11 = np.min(forward_result['predicted_s_parameters'])
                print(f"   验证S11最小值: {predicted_s11:.2f} dB")
        except Exception as e:
            print(f"   验证失败: {e}")

    return result


def plot_predictions(s_params_pred, far_field_pred, patch_length, patch_width):
    """
    绘制预测结果

    Args:
        s_params_pred: 预测的S参数数据
        far_field_pred: 预测的远区场一维数据
        patch_length: 贴片长度
        patch_width: 贴片宽度
    """
    # 创建保存目录
    if not os.path.exists('results'):
        os.makedirs('results')

    # 绘制S参数曲线
    if s_params_pred is not None:
        plt.figure(figsize=(12, 8))

        # 生成频率点
        frequencies = np.linspace(2.0, 3.0, len(s_params_pred))

        plt.plot(frequencies, s_params_pred, linewidth=2, color='blue', marker='o', markersize=4)
        plt.xlabel('频率 (GHz)')
        plt.ylabel('S11 (dB)')
        plt.title(f'预测S11参数曲线\n(贴片尺寸: {patch_length}×{patch_width}mm)')
        plt.grid(True, alpha=0.3)

        # 标注S11最小值
        s11_min = np.min(s_params_pred)
        min_freq_idx = np.argmin(s_params_pred)
        min_freq = frequencies[min_freq_idx]
        plt.annotate(f'最小值: {s11_min:.2f}dB\n频率: {min_freq:.3f}GHz',
                    xy=(min_freq, s11_min),
                    xytext=(min_freq+0.1, s11_min+5),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        plt.tight_layout()
        s_params_plot_path = f'results/s_params_prediction_{patch_length}x{patch_width}.png'
        plt.savefig(s_params_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 S参数预测图已保存到: {s_params_plot_path}")

    # 绘制远区场方向图
    if far_field_pred is not None:
        # 将一维远区场数据重新reshape为极坐标格式 (theta=37, phi=73)
        try:
            # 检查数据长度是否匹配
            expected_size = 37 * 73
            if len(far_field_pred) != expected_size:
                print(f"⚠️  远区场数据长度不匹配: 期望 {expected_size}, 实际 {len(far_field_pred)}")
                # 尝试使用实际长度进行reshape
                far_field_matrix = far_field_pred.reshape(37, -1)
            else:
                far_field_matrix = far_field_pred.reshape(37, 73)

            print(f"📊 远区场数据已reshape为: {far_field_matrix.shape}")

        except Exception as e:
            print(f"⚠️  远区场数据reshape失败: {e}")
            return

        # 创建单独的远区场方向图
        # 1. 水平面方向图 (theta=90°切面)
        plt.figure(figsize=(10, 6))
        phi_deg = np.linspace(0, 360, far_field_matrix.shape[1])
        # 取theta=90°的切面（大约在中间位置）
        theta_90_idx = far_field_matrix.shape[0] // 2
        horizontal_pattern = far_field_matrix[theta_90_idx, :]

        # 为了闭合图形，添加第一个点到末尾
        phi_deg_closed = np.append(phi_deg, 360)
        horizontal_pattern_closed = np.append(horizontal_pattern, horizontal_pattern[0])

        plt.polar(np.deg2rad(phi_deg_closed), horizontal_pattern_closed, linewidth=2)
        plt.title(f'水平面方向图 (θ=90°)\n(贴片尺寸: {patch_length}×{patch_width}mm)')
        plt.tight_layout()
        horizontal_plot_path = f'results/horizontal_pattern_{patch_length}x{patch_width}.png'
        plt.savefig(horizontal_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📡 水平面方向图已保存到: {horizontal_plot_path}")

        # 2. 垂直面方向图 (phi=0°和phi=90°切面)
        plt.figure(figsize=(10, 6))
        theta_deg = np.linspace(0, 180, far_field_matrix.shape[0])

        # phi=0°切面
        vertical_pattern_0 = far_field_matrix[:, 0]
        plt.plot(theta_deg, vertical_pattern_0, label='φ=0°', linewidth=2)

        # phi=90°切面
        phi_90_idx = min(18, far_field_matrix.shape[1]-1)
        vertical_pattern_90 = far_field_matrix[:, phi_90_idx]
        plt.plot(theta_deg, vertical_pattern_90, label='φ=90°', linewidth=2)

        plt.xlabel('θ (度)')
        plt.ylabel('增益 (dBi)')
        plt.title(f'垂直面方向图\n(贴片尺寸: {patch_length}×{patch_width}mm)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        vertical_plot_path = f'results/vertical_pattern_{patch_length}x{patch_width}.png'
        plt.savefig(vertical_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📡 垂直面方向图已保存到: {vertical_plot_path}")

        # 3. 3D方向图
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        theta = np.linspace(0, np.pi, far_field_matrix.shape[0])
        phi = np.linspace(0, 2*np.pi, far_field_matrix.shape[1])
        Theta, Phi = np.meshgrid(theta, phi, indexing='ij')

        # 转换为笛卡尔坐标
        R = far_field_matrix
        X = R * np.sin(Theta) * np.cos(Phi)
        Y = R * np.sin(Theta) * np.sin(Phi)
        Z = R * np.cos(Theta)

        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        # 设置坐标轴标签（使用极坐标描述）
        ax.set_xlabel('θ (极角) sin(θ)cos(φ)')
        ax.set_ylabel('θ (极角) sin(θ)sin(φ)')
        ax.set_zlabel('θ (极角) cos(θ)')

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # 设置标题
        ax.set_title(f'3D远区场方向图\n(贴片尺寸: {patch_length}×{patch_width}mm)')
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5)
        plt.tight_layout()
        far_field_3d_plot_path = f'results/far_field_3d_{patch_length}x{patch_width}.png'
        plt.savefig(far_field_3d_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📡 3D远区场方向图已保存到: {far_field_3d_plot_path}")

        import plotly.graph_objects as go
        import plotly.offline as pyo

        # 使用plotly创建交互式3D图
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])

        fig.update_layout(
            title=f'3D远区场方向图 (贴片尺寸: {patch_length}×{patch_width}mm)',
            scene=dict(
                # xaxis_title='X (径向方向)',
                # yaxis_title='Y (径向方向)',
                # zaxis_title='Z (径向方向)',
                xaxis_title='θ (极角) sin(θ)cos(φ)',
                yaxis_title='θ (极角) sin(θ)sin(φ)',
                zaxis_title='θ (极角) cos(θ)',
                camera_eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            width=800,
            height=600
        )

        # 在现有代码基础上添加参考线
        # 添加主要的极角参考线
        theta_lines = np.linspace(0, np.pi, 7)
        phi_lines = np.linspace(0, 2*np.pi, 13)

        for theta in theta_lines:
            for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
                r_vals = np.linspace(0, np.max(R), 50)
                x_line = r_vals * np.sin(theta) * np.cos(phi)
                y_line = r_vals * np.sin(theta) * np.sin(phi)
                z_line = r_vals * np.cos(theta)
                fig.add_trace(go.Scatter3d(
                    x=x_line, y=y_line, z=z_line,
                    mode='lines',
                    line=dict(color='red', width=2),
                    showlegend=False,
                    opacity=0.5
                ))
        # 使用更合适的颜色映射
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=R,  # 使用增益值作为颜色映射
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="增益 (dBi)")
        )])


        # 保存为HTML文件以支持完整交互
        plotly_plot_path = f'results/far_field_3d_interactive_{patch_length}x{patch_width}.html'
        pyo.plot(fig, filename=plotly_plot_path, auto_open=False)
        print(f"🌐 交互式3D方向图已保存到: {plotly_plot_path}")

    print("✅ 预测结果可视化完成！")


def plot_3d_radiation_pattern_from_csv(csv_file_path, output_html_path=None):
    """
    从CSV文件读取远区场数据并绘制3D方向图

    Args:
        csv_file_path: CSV文件路径，应包含 Theta(deg), Phi(deg), Gain_dB 三列
        output_html_path: 输出HTML文件路径（可选）
    """
    # 读取CSV数据
    df = pd.read_csv(csv_file_path)

    # 检查必需的列是否存在
    required_columns = ['Theta(deg)', 'Phi(deg)', 'Gain_dB']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"缺少必需的列: {col}")

    # 提取数据
    theta_deg = df['Theta(deg)'].values
    phi_deg = df['Phi(deg)'].values
    gain_db = df['Gain_dB'].values

    # 转换为弧度
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)

    # 转换为笛卡尔坐标
    # 假设增益值直接作为径向距离
    r = gain_db - np.min(gain_db) + 1  # 偏移以确保正值
    x = r * np.sin(theta_rad) * np.cos(phi_rad)
    y = r * np.sin(theta_rad) * np.sin(phi_rad)
    z = r * np.cos(theta_rad)

    # 创建3D散点图
    fig = go.Figure()

    # 添加散点数据
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=4,
            color=gain_db,
            colorscale='Viridis',
            colorbar=dict(title="增益 (dBi)"),
            showscale=True
        ),
        text=[f'θ: {t:.1f}°<br>φ: {p:.1f}°<br>Gain: {g:.2f} dBi'
              for t, p, g in zip(theta_deg, phi_deg, gain_db)],
        hoverinfo='text',
        name='远区场数据'
    ))

    # 如果数据点较少，可以创建插值表面
    if len(df) > 100:  # 数据点足够多时创建表面
        # 创建规则网格用于插值
        theta_grid = np.linspace(0, np.pi, 50)
        phi_grid = np.linspace(0, 2*np.pi, 100)
        Theta_grid, Phi_grid = np.meshgrid(theta_grid, phi_grid)

        # 插值到规则网格
        points = np.column_stack((theta_rad, phi_rad))
        values = gain_db
        grid_points = np.column_stack((
            Theta_grid.ravel(),
            Phi_grid.ravel()
        ))

        try:
            interpolated_gain = griddata(
                points, values, grid_points,
                method='cubic', fill_value=np.min(gain_db)
            ).reshape(Theta_grid.shape)

            # 转换插值数据到笛卡尔坐标
            R_grid = interpolated_gain - np.min(interpolated_gain) + 1
            X_grid = R_grid * np.sin(Theta_grid) * np.cos(Phi_grid)
            Y_grid = R_grid * np.sin(Theta_grid) * np.sin(Phi_grid)
            Z_grid = R_grid * np.cos(Theta_grid)

            # 添加插值表面
            fig.add_trace(go.Surface(
                x=X_grid, y=Y_grid, z=Z_grid,
                surfacecolor=interpolated_gain,
                colorscale='Viridis',
                opacity=0.7,
                showscale=False,
                name='插值表面'
            ))
        except Exception as e:
            print(f"插值失败: {e}")

    # 设置布局
    fig.update_layout(
        title=f'3D远区场方向图<br>数据来源: {csv_file_path}',
        scene=dict(
            xaxis_title='X (径向方向)',
            yaxis_title='Y (径向方向)',
            zaxis_title='Z (径向方向)',
            aspectmode='data'
        ),
        width=800,
        height=600
    )

    # 保存或显示图表
    if output_html_path:
        pyo.plot(fig, filename=output_html_path, auto_open=False)
        print(f"3D方向图已保存到: {output_html_path}")
    else:
        fig.show()

    return fig




if __name__ == "__main__":
    print("贴片天线GAN模型使用系统")
    print("=" * 70)

    # 使用已训练模型
    # model_info_path = 'models/trained_gan_model_info.npy'

    # target_specs = load_target_specs_from_csv('TEST_RESULT/data_dict_pandas_20251121_111221.csv')
    # use_trained_gan_model(model_info_path, target_specs)
    #
    # s11_min_predict, freq_at_s11_min_predict, s11_curve_predict, s11_min, freq_at_s11_min, far_field_gain = use_trained_gan_model_prediction_results()
    # # s11_min_predict, freq_at_s11_min_predict, s11_curve_predict, s11_min, freq_at_s11_min, far_field_gain = use_trained_gan_model_prediction_results(patch_lengths='40', patch_widths='40')
    # print("\n" + "=" * 70)
    # print(f"  实际性能: S11={s11_min:.2f}dB, 频率={freq_at_s11_min:.2f}GHz, 增益={far_field_gain:.2f}dBi")
    # print(f"  模型预测性能: S11={s11_min_predict:.2f}dB, "
    #       f"频率={freq_at_s11_min_predict:.2f}GHz, "
    #       # f"增益={far_field_gain_predict:.2f}dBi"
    #       )
    # print("\n" + "=" * 70)

    # 使用逆向模型
    print("\n" + "=" * 70)
    print("使用逆向模型（从性能参数预测尺寸）")
    print("=" * 70)

    # 方法1: 直接使用具体的性能数据
    target_specs_s11, patch_length, patch_width = load_target_specs_from_csv('../RESULT/data_dict_pandas_20251125_101258.csv')
    # print(f"目标性能数据:{target_specs_s11}")
    target_specs_gain = extract_gain_matrix_from_csv('../RESULT/data_dict_pandas_20251125_101258.csv')
    # print(f"目标增益数据:{target_specs_gain}")
    if target_specs_gain is not None:
        inverse_result = use_inverse_model('models/inverse_trained_model.pth',
                                           target_specs_s11[0] if target_specs_s11 else None,
                                           target_specs_gain)

    # # 方法2: 根据目标性能指标生成设计
    # inverse_result = use_inverse_model_with_target_specs(
    #     'models/inverse_trained_model.pth',
    #     target_s11_min=-15,  # 目标S11最小值
    #     target_frequency=2.45,  # 目标频率
    #     target_gain=8.0  # 目标增益
    # )

    if inverse_result['predicted_dimensions']:
        print(f"✅ 逆向设计完成!")
        print(f"       推荐尺寸: {inverse_result['predicted_dimensions']['length']:.2f} × "
              f"{inverse_result['predicted_dimensions']['width']:.2f} mm")
        print(f"目标性能对应尺寸: {patch_length:.2f} × {patch_width:.2f} mm")

        #使用模型预测结果
        print("使用模型预测结果:")
        model_info_path = 'models/multi_output_trained_model.npy'
        result = use_multi_output_model(model_info_path,
                                        float(inverse_result['predicted_dimensions']['length']),
                                        float(inverse_result['predicted_dimensions']['width']))

        print("\n" + "=" * 70)
        actual_length = inverse_result['predicted_dimensions']['length']
        actual_width = inverse_result['predicted_dimensions']['width']
        length_error = abs(patch_length - actual_length)
        width_error = abs(patch_width - actual_width)
        length_error_percent = (length_error / actual_length) * 100
        width_error_percent = (width_error / actual_width) * 100
        print(f"📏 尺寸误差分析:")
        print(f"   长度误差: {length_error:.2f} mm ({length_error_percent:.2f}%)")
        print(f"   宽度误差: {width_error:.2f} mm ({width_error_percent:.2f}%)")
        print(f"   总体误差: {length_error + width_error:.2f} mm")
        print("\n" + "=" * 70)
    else:
        print(f"❌ 逆向设计失败: {inverse_result.get('error', '未知错误')}")

    # print("\n" + "=" * 70)

    # print("\n" + "=" * 70)
    # model_info_path = 'models/multi_output_trained_model.npy'
    # result = use_multi_output_model(model_info_path, 39, 48.4)
    # result = use_multi_output_model(model_info_path, 35, 50)
    # print("\n" + "=" * 70)

    print("\n" + "=" * 70)
    print("模型使用完成！")
    print("=" * 70)
    print("\n您可以在 results 目录中查看生成的设计结果。")
