"""
贴片天线设计系统完整使用示例
Patch Antenna Design System - Complete Usage Example

这个示例展示了如何在Python代码中完整使用贴片天线设计系统，
包括数据加载、模型训练、参数优化、结果分析等全流程。
"""

import sys
import os
import time
import numpy as np
import torch  # 补全torch导入
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from python_hfss import calculate_from_hfss as calculate_from_hfss_py

# 导入自定义模块
try:
    import calculate_by_hfss
    from patch_antenna_design import PatchAntennaDesignSystem
    from merge_csv_files import merge_single_line_csv_files  # 明确导入合并函数
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)

# 添加当前目录到系统路径（确保模块能被找到）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_device():
    """自动检测可用设备（优先GPU，没有则用CPU）"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"ℹ️  未检测到GPU，使用CPU训练（速度可能较慢）")
    return device

def gan_demo(create_antenna_data):
    """批量优化演示"""
    print("\n" + "=" * 70)
    print("GAN 模型")
    print("=" * 70)

    device = get_device()
    system = PatchAntennaDesignSystem()

    # 2. 加载数据
    print("\n2. 加载天线数据...")

    # 生成天线数据
    if create_antenna_data != 0:
        print(f"\n 生成{create_antenna_data}个天线数据...")
        calculate_by_hfss.Generate_test_data(create_antenna_data)

    print("=============================合并所有数据=============================")
    # input_pattern = "./RESULT/data_dict_pandas_*.csv"
    input_pattern = "./Train_data/data_dict_pandas_*.csv"
    output_file = "merged_detailed_antenna_data.csv"
    header_check_count = 40
    merge_single_line_csv_files(input_pattern, output_file, header_check_count)
    print(f"\n=============================合并完成！=============================")

    # 加载数据
    print("=============================加载数据=============================")
    # freq_points = np.linspace(2.0, 3.0, 201).tolist()
    # s11_names = [f'{freq:.3f}' for freq in freq_points]
    # try:
    #     X_scaled, y, X_original, y_original = system.load_csv_data(
    #         csv_file='./merged_detailed_antenna_data.csv',
    #         param_cols=['patch_length', 'patch_width'],
    #         perf_cols=['_最小值', 'Freq [GHz]', 'Gain_dB'] + s11_names
    #     )
    try:
        X_scaled, y, X_original, y_original = system.load_csv_data(
            csv_file='./merged_detailed_antenna_data.csv',
            param_cols=['patch_length', 'patch_width'],
            perf_cols=None  # 让函数自动检测列名
        )
        print(f"=============================数据加载完成: {X_original.shape[0]}个样本=============================")
    except Exception as e:
        print(f"=============================❌ 数据加载失败，使用合成数据: {e}=============================")
        X_scaled, y, X_original, y_original = system.generate_synthetic_data(num_samples=create_antenna_data)

    # 划分数据集并移到设备
    print("=============================划分数据集并移到设备=============================")
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def to_tensor_and_device(data, device):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        return data.to(device)

    X_train = to_tensor_and_device(X_train, device)
    y_train = to_tensor_and_device(y_train, device)
    X_val = to_tensor_and_device(X_val, device)
    y_val = to_tensor_and_device(y_val, device)

    # 训练模型
    print("=============================训练模型=============================")
    # 3. GAN工作流程
    print(f"\n2. GAN模型训练...")

    # 训练GAN
    # 同时训练正向和反向GAN
    history = system.train_gan(X_train, y_train, epochs=3000, batch_size=128, train_both=True)
    # # 或者保持原有功能，只训练正向GAN
    # history = system.train_gan(X_train, y_train, epochs=3000, batch_size=128, forward_gan=True)
    # # 或者保持原有功能，只训练反向GAN
    # history = system.train_gan(X_train, y_train, epochs=3000, batch_size=128, forward_gan=False)

    # 可视化GAN训练结果
    system.visualize_gan_results(history)

    # 定义设计目标
    target_performances = [
        [-35.0, 2.45, 7.0],  # WiFi 2.45GHz 高性能设计
        [-30.0, 2.4, 6.5],  # WiFi 2.4GHz 标准设计
        [-25.0, 2.5, 6.0],  # 低成本设计
        [-40.0, 2.42, 7.5]  # 超高性能设计
    ]

    # 使用GAN生成天线设计
    print(f"\n3. 使用GAN生成天线设计...")
    # generated_designs, generated_performances = system.generate_antenna_designs(
    #     target_performances, num_samples=20
    # )

    # 使用两种GAN生成设计
    generated_designs, generated_performances = system.generate_antenna_designs(
        target_performances, num_samples=20
    )

    # 可视化生成结果
    system.visualize_gan_results(history, generated_designs, generated_performances)
    # system.visualize_gan_results(gan_history_forward, generated_designs, generated_performances)

    # 保存生成的设计
    design_df = pd.DataFrame({
        'patch_length': generated_designs[:, 0],
        'patch_width': generated_designs[:, 1],
        's11_min': generated_performances[:, 0],
        'freq_at_s11_min': generated_performances[:, 1],
        'far_field_gain': generated_performances[:, 2]
    })
    design_df.to_csv('gan_generated_designs.csv', index=False)
    print(f"生成的天线设计已保存到 gan_generated_designs.csv")

    # 选择最佳设计进行HFSS验证 GAN模型验证
    if len(generated_designs) > 0:
        # best_design_idx = np.argmin(np.mean(np.abs(generated_performances - np.array(target_performances[0])), axis=1))
        # 修改为：
        if len(generated_performances) > 0:
            # 构造完整的目标性能向量（204维）
            full_target_perf = np.zeros(generated_performances.shape[1])  # 生成与generated_performances相同维度的零数组
            full_target_perf[0] = target_performances[0][0]  # S11最小值
            full_target_perf[1] = target_performances[0][1]  # 对应频率
            full_target_perf[2] = target_performances[0][2]  # 远区场增益
            # 其余201个S11点可以设为默认值或根据需要进行设置

            # 计算每个设计与完整目标性能的误差
            performance_errors = np.mean(np.abs(generated_performances - full_target_perf), axis=1)
            best_design_idx = np.argmin(performance_errors)
            best_design = generated_designs[best_design_idx]
            best_performance = generated_performances[best_design_idx]
        best_design = generated_designs[best_design_idx]
        best_performance = generated_performances[best_design_idx]

        print(f"\n4. HFSS仿真验证最佳设计...")
        print(f"最佳设计参数: 长度={best_design[0]:.2f}mm, 宽度={best_design[1]:.2f}mm")
        print(
            f"预测性能: S11={best_performance[0]:.2f}dB, 频率={best_performance[1]:.2f}GHz, 增益={best_performance[2]:.2f}dBi")

        # HFSS仿真
        # simulated_performance = system.hfss_interface(best_design)
        antenna_params_test_by_hfss = {"unit": "GHz",
                                       "patch_length": float(best_design[0]),
                                       "patch_width": float(best_design[1]),
                                       "patch_name": "Patch",
                                       "freq_step": "0.01GHz",
                                       "num_of_freq_points": 201,
                                       "start_frequency": 2,  # 起始工作频率 (GHz)
                                       "stop_frequency": 3,  # 截止频率
                                       "center_frequency": 2.5,  # 中心频率
                                       "sweep_type": "Interpolating",  # 扫描频率设置
                                       "sub_length": 50,  # 介质板长度(mm)
                                       "sub_width": 60,  # 介质板宽度(mm)
                                       "sub_high": 1.575,  # 介质板厚度(mm)
                                       "feed_r1": 0.5,
                                       "feed_h": 1.575,
                                       "feed_center": 6.3,
                                       "lumpedport_r": 1.5,
                                       "lumpedport_D": 2.3 / 2,
                                       }
        train_model = False
        success, freq_at_s11_min, far_field_gain, s11_min, output_file, output_file_farfield= calculate_from_hfss_py(antenna_params_test_by_hfss, train_model)

        system.plot_s11_comparison_advanced(float(best_design[0]), float(best_design[1]),
                                     output_file, frequency_column=0, s11_column=1)

        # 设计可行性分析
        print(f"\n5. 设计可行性分析:")
        is_feasible = True

        # if best_performance[0] > -15:
        if s11_min > -15:
            # print(f"  ⚠️  S11值 {best_performance[0]:.2f}dB 偏高")
            print(f"  ⚠️  S11值 {s11_min:.2f}dB 偏高")
            is_feasible = False
        else:
            # print(f"  ✓ S11值 {best_performance[0]:.2f}dB 满足要求")
            print(f"  ✓ S11值 {s11_min:.2f}dB 满足要求")

        # if not (2.4 <= best_performance[1] <= 2.5):
        if not (2.4 <= freq_at_s11_min <= 2.5):
            # print(f"  ⚠️  工作频率 {best_performance[1]:.2f}GHz 不在WiFi 2.4GHz频段内")
            print(f"  ⚠️  工作频率 {freq_at_s11_min:.2f}GHz 不在WiFi 2.4GHz频段内")
            is_feasible = False
        else:
            print(f"  ✓ 工作频率在WiFi 2.4GHz频段内")

        # if best_performance[2] < 5.0:
        if far_field_gain < 5.0:
            # print(f"  ⚠️  增益 {best_performance[2]:.2f}dBi 偏低")
            print(f"  ⚠️  增益 {far_field_gain:.2f}dBi 偏低")
            is_feasible = False
        else:
            # print(f"  ✓ 增益 {best_performance[2]:.2f}dBi 满足要求")
            print(f"  ✓ 增益 {far_field_gain:.2f}dBi 满足要求")

        if is_feasible:
            print("🎉 GAN生成的设计成功！满足所有要求。")
        else:
            print("⚠️ GAN生成的设计基本完成，但部分指标需要进一步优化。")

    # 保存批量设计结果
    if not os.path.exists('patch_antenna_results'):
        os.makedirs('patch_antenna_results')

    np.save('patch_antenna_results/gan_model_batch_design_results.npy', best_performance)
    print(f"\n批量设计完成！结果已保存到 gan_model_batch_design_results.npy")

    return best_performance


if __name__ == "__main__":
    print("欢迎使用贴片天线设计系统！")
    print("本系统专门用于贴片天线的深度学习设计和优化。")
    print("=" * 70)

    create_antenna_data = 0
    gan_demo(create_antenna_data)

    # 运行完整工作流程演示
    print("\n正在运行完整设计流程...")

    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    print("\n您可以在 patch_antenna_results 目录中查看详细的设计结果和可视化图表。")