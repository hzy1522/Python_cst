"""
贴片天线设计系统 - 模型使用模块
Patch Antenna Design System - Model Usage Module
"""

import sys
import os
import numpy as np
import torch
import pandas as pd


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
    try:
        generated_designs, generated_performances = system.generate_antenna_designs(
            target_performances, num_samples=20
        )
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
        s11_curve_predict = system.predict_s11_from_dimensions(design[0], design[1])

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
            success, freq_at_s11_min, far_field_gain, s11_min, output_file = calculate_from_hfss_py(
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

    return target_specs


if __name__ == "__main__":
    print("贴片天线GAN模型使用系统")
    print("=" * 70)

    # 使用已训练模型
    model_info_path = 'models/trained_gan_model_info.npy'

    target_specs = load_target_specs_from_csv('TEST_RESULT/data_dict_pandas_20251121_111221.csv')
    use_trained_gan_model(model_info_path, target_specs)
    #
    use_trained_gan_model_prediction_results()
    # use_trained_gan_model_prediction_results(patch_lengths='40', patch_widths='40')

    print("\n" + "=" * 70)
    print("模型使用完成！")
    print("=" * 70)
    print("\n您可以在 results 目录中查看生成的设计结果。")
