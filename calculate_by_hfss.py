"""
更新后的天线数据生成器 - 支持详细的HFSS参数设置
"""

import numpy as np
import pandas as pd
import time
import os
from typing import List, Tuple, Dict
from python_hfss import calculate_from_hfss as calculate_from_hfss_py

class AntennaDataGenerator:
    def __init__(self):
        """初始化天线数据生成器"""
        self.param_names = [
            'unit',  # 单位设置
            "stop_frequency",  # 截止频率
            'center_frequency',  # 中心频率
            'sweep_type' , # 扫描频率设置
            'patch_length',  # 贴片长度(mm)             10-50
            'patch_width' , # 10-60
            'patch_name',
            'freq_step',
            'num_of_freq_points',
            'sub_length',  # 介质板长度(mm)
            'sub_width' , # 介质板宽度(mm)
            'sub_high', # 介质板厚度(mm)
            'feed_r1',
            'feed_h' ,
            'feed_center',
            'lumpedport_r',
            'lumpedport_D',
            # 'unit', 'start_frequency', 'stop_frequency', 'center_frequency',
            # 'sweep_type', 'ground_thickness', 'signal_layer_thickness',
            # 'patch_length', 'patch_width', 'patch_name', 'freq_step', 'num_of_freq_points'
        ]
        self.perf_names = ['s11_min', 'freq_at_s11_min', 'far_field_gain']

        # 参数范围设置 - 只保留需要随机生成的参数
        self.param_ranges = {
            # 物理尺寸参数 (mm)
            'patch_length': (10.0, 50.0),  # 贴片长度
            'patch_width': (10.0, 60.0),  # 贴片宽度
        }

        # 固定参数 - 包含所有不需要随机生成的参数
        self.fixed_params = {
            'unit': 'GHz',
            'sweep_type': 'Interpolating',
            'patch_name': 'Patch',
            'freq_step': '0.01GHz',
            'start_frequency': 2,
            'stop_frequency': 3,
            'center_frequency': 2.5,
            'sub_length': 50,  # 介质板长度(mm)
            'sub_width': 60,  # 介质板宽度(mm)
            'sub_high': 1.575,  # 介质板厚度(mm)
            'feed_r1': 0.5,
            'feed_h': 1.575,
            'feed_center': 6.3,
            'lumpedport_r': 1.5,
            'lumpedport_D': 2.3/2,
            'num_of_freq_points': 201,
            'ground_thickness': 0.035,  # 地板厚度(mm)
            'signal_layer_thickness': 0.035,  # 信号线厚度(mm)
        }

        print("天线数据生成器初始化完成")
        print("随机生成的参数范围:")
        for param, (min_val, max_val) in self.param_ranges.items():
            unit = 'GHz' if 'frequency' in param else 'mm' if param != 'num_of_freq_points' else ''
            print(f"  {param}: {min_val} - {max_val} {unit}")

        print("固定参数:")
        for param, value in self.fixed_params.items():
            print(f"  {param}: {value}")

    def calculate_from_hfss(self, antenna_params: Dict[str, any]) -> Tuple[float, float, float]:
        """
        更新版HFSS仿真计算函数 - 接受详细的参数字典

        参数:
        antenna_params: 包含所有HFSS参数的字典
            - unit: 单位设置
            - start_frequency: 起始工作频率 (GHz)
            - stop_frequency: 截止频率 (GHz)
            - center_frequency: 中心频率 (GHz)
            - sweep_type: 扫描频率设置
            - ground_thickness: 地板厚度 (mm)
            - signal_layer_thickness: 信号线厚度 (mm)
            - patch_length: 贴片长度 (mm)
            - patch_width: 贴片宽度 (mm)
            - patch_name: 贴片名称
            - freq_step: 频率步长
            - num_of_freq_points: 频率点数

        返回:
        S11最小值 (dB), 对应频率 (GHz), 远区场增益 (dBi)
        """
        print(f"\n=== HFSS仿真开始 ===")
        print(
            f"频率范围: {antenna_params['start_frequency']} - {antenna_params['stop_frequency']} {antenna_params['unit']}")
        print(f"中心频率: {antenna_params['center_frequency']} {antenna_params['unit']}")
        print(f"扫描类型: {antenna_params['sweep_type']}")
        print(f"频率点数: {antenna_params['num_of_freq_points']}")
        print(f"天线尺寸: 长度={antenna_params['patch_length']:.2f}mm, 宽度={antenna_params['patch_width']:.2f}mm")
        print(f"地板厚度: {antenna_params['ground_thickness']:.3f}mm")
        print(f"信号线厚度: {antenna_params['signal_layer_thickness']:.3f}mm")

        # 模拟HFSS计算过程
        simulation_time = np.random.uniform(0.2, 0.5)  # 模拟计算时间
        time.sleep(simulation_time)

        # 基于电磁学原理的计算模型
        c = 3e8  # 光速 (m/s)
        epsilon_r = 4.4  # FR4基板介电常数

        # 计算理论谐振频率
        patch_length_mm = antenna_params['patch_length']
        patch_width_mm = antenna_params['patch_width']

        # 有效介电常数
        epsilon_eff = (epsilon_r + 1) / 2 + (epsilon_r - 1) / 2 * (1 + 12 * (0.8 / patch_length_mm)) ** (-0.5)

        # 谐振频率计算
        theoretical_freq = c / (2 * patch_length_mm * 1e-3 * np.sqrt(epsilon_eff)) / 1e9

        # 考虑实际尺寸和频率范围的影响
        center_freq = antenna_params['center_frequency']
        freq_at_s11_min = theoretical_freq * (0.95 + 0.1 * np.random.random())

        # 确保频率在设置的范围内
        freq_at_s11_min = np.clip(
            freq_at_s11_min,
            antenna_params['start_frequency'],
            antenna_params['stop_frequency']
        )

        # 计算S11最小值 (考虑匹配质量)
        s11_min = -40 + 5 * np.random.random()  # 较好的匹配
        if abs(freq_at_s11_min - center_freq) > 1.0:
            s11_min += 10 * abs(freq_at_s11_min - center_freq)  # 偏离中心频率时匹配变差

        s11_min = np.clip(s11_min, -45, -10)

        # 计算远区场增益
        gain = 3.0 + 2 * np.random.random()  # 微带天线典型增益
        if patch_length_mm < 9.0 or patch_width_mm < 9.0:
            gain -= 1.0  # 尺寸较小时增益降低

        gain = np.clip(gain, 1.0, 8.0)

        print(f"=== HFSS仿真完成 ===")
        print(f"S11最小值: {s11_min:.2f} dB")
        print(f"对应频率: {freq_at_s11_min:.2f} {antenna_params['unit']}")
        print(f"远区场增益: {gain:.2f} dBi")

        return s11_min, freq_at_s11_min, gain

    def generate_antenna_params_dict(self) -> Dict[str, any]:
        """生成单个天线的参数字典"""
        params = {}

        # 添加固定参数
        params.update(self.fixed_params)

        # 只生成需要随机变化的参数
        for param, (min_val, max_val) in self.param_ranges.items():
            params[param] = np.random.uniform(min_val, max_val)

        # 确保频率范围的合理性（虽然现在是固定的，但保留此检查）
        if params['start_frequency'] >= params['stop_frequency']:
            params['start_frequency'], params['stop_frequency'] = params['stop_frequency'], params['start_frequency']

        if params['center_frequency'] < params['start_frequency']:
            params['center_frequency'] = params['start_frequency'] + (
                        params['stop_frequency'] - params['start_frequency']) * 0.3

        if params['center_frequency'] > params['stop_frequency']:
            params['center_frequency'] = params['stop_frequency'] - (
                        params['stop_frequency'] - params['start_frequency']) * 0.3

        return params

    def generate_patch_param_data(self, num_samples: int = 5000,
                                  save_to_csv: bool = False,
                                  output_file: str = "./antenna_data_detailed.csv") -> Tuple[List[Dict], np.ndarray]:
        """
        生成详细的天线参数数据

        参数:
        num_samples: 生成的样本数量
        save_to_csv: 是否保存到CSV文件
        output_file: 输出CSV文件名

        返回:
        params_list: 天线参数字典列表
        y: 性能指标数组 (num_samples, 3)
        """
        print(f"\n开始生成 {num_samples} 个天线样本...")
        start_time = time.time()

        params_list = []
        y = np.zeros((num_samples, len(self.perf_names)))

        # 生成进度条
        progress_interval = max(1, num_samples // 50)

        for i in range(num_samples):
            # 生成天线参数字典
            antenna_params = self.generate_antenna_params_dict()
            params_list.append(antenna_params)

            # # 调用HFSS计算性能
            # s11_min, freq_at_s11_min, far_field_gain = self.calculate_from_hfss(antenna_params)

#==========================================调用hfss计算性能==========================================
            train_model = True
            success, freq_at_s11_min, far_field_gain, s11_min = calculate_from_hfss_py(antenna_params, train_model)
#==========================================调用hfss计算性能==========================================
            y[i] = [s11_min, freq_at_s11_min, far_field_gain]

            # 显示进度
            if (i + 1) % progress_interval == 0 or i + 1 == num_samples:
                progress = (i + 1) / num_samples * 100
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time / (i + 1) * num_samples
                remaining_time = estimated_total_time - elapsed_time

                print(f"\r进度: {i + 1}/{num_samples} ({progress:.1f}%), "
                      f"耗时: {elapsed_time:.1f}s, 剩余: {remaining_time:.1f}s", end="")

        print(f"\n\n数据生成完成！总耗时: {time.time() - start_time:.2f}秒")

        # 显示统计信息
        self.print_statistics(params_list, y)

        # 保存到CSV
        if save_to_csv:
            self.save_to_csv(params_list, y, output_file)

        return params_list, y

    def print_statistics(self, params_list: List[Dict], y: np.ndarray):
        """打印数据统计信息"""
        print(f"\n=== 参数统计信息 ===")

        # 转换为DataFrame便于统计
        df_params = pd.DataFrame(params_list)

        # 参数统计 - 只显示随机生成的参数
        for param in self.param_ranges.keys():
            values = df_params[param].values
            print(f"{param}: 均值={values.mean():.3f}, 标准差={values.std():.3f}")

        # 性能指标统计
        print(f"\n=== 性能指标统计 ===")
        for i, perf_name in enumerate(self.perf_names):
            print(f"{perf_name}: 均值={y[:, i].mean():.3f}, 标准差={y[:, i].std():.3f}")

    def save_to_csv(self, params_list: List[Dict], y: np.ndarray, output_file: str):
        """保存数据到CSV文件"""
        # 创建完整的数据字典
        data = []
        for i, params in enumerate(params_list):
            row = params.copy()
            row.update({
                's11_min': y[i, 0],
                'freq_at_s11_min': y[i, 1],
                'far_field_gain': y[i, 2]
            })
            data.append(row)

        # 保存到CSV
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

        print(f"\n数据已保存到: {output_file}")
        print(f"文件大小: {os.path.getsize(output_file) / 1024:.2f} KB")
        print(f"包含列数: {len(df.columns)}")

def Generate_test_data(num_samples):
    """生成天线训练数据"""
    # 创建生成器
    generator = AntennaDataGenerator()

    # 生成测试样本
    print("\n=== 生成测试数据 ===")
    params_list, y = generator.generate_patch_param_data(
        num_samples=num_samples,
        # save_to_csv=True,
        # output_file="./antenna_test_data_detailed.csv"
    )

    # 显示前3个样本的详细信息
    print(f"\n=== 前3个样本详细信息 ===")
    for i in range(min(3, len(params_list))):
        print(f"\n样本 {i + 1}:")
        params = params_list[i]
        # 只显示随机生成的参数和关键固定参数
        print(f"  随机参数:")
        for key in generator.param_ranges.keys():
            print(f"    {key}: {params[key]:.2f}")
        print(f"  关键固定参数:")
        print(f"    center_frequency: {params['center_frequency']} {params['unit']}")
        print(f"    sub_high: {params['sub_high']}mm")
        print(f"    feed_center: {params['feed_center']}mm")
        print(f"  性能指标: S11={y[i, 0]:.2f}dB, 频率={y[i, 1]:.2f}GHz, 增益={y[i, 2]:.2f}dBi")


def main():
    """演示更新后的功能"""
    # 创建生成器
    generator = AntennaDataGenerator()

    # 生成10个测试样本
    print("\n=== 生成测试数据 ===")
    params_list, y = generator.generate_patch_param_data(
        num_samples=10,
        # save_to_csv=True,
        # output_file="./antenna_test_data_detailed.csv"
    )

    # 显示前3个样本的详细信息
    print(f"\n=== 前3个样本详细信息 ===")
    for i in range(min(3, len(params_list))):
        print(f"\n样本 {i + 1}:")
        params = params_list[i]
        # 只显示随机生成的参数和关键固定参数
        print(f"  随机参数:")
        for key in generator.param_ranges.keys():
            print(f"    {key}: {params[key]:.2f}mm")
        print(f"  关键固定参数:")
        print(f"    center_frequency: {params['center_frequency']} {params['unit']}")
        print(f"    sub_high: {params['sub_high']}mm")
        print(f"    feed_center: {params['feed_center']}mm")
        print(f"  性能指标: S11={y[i, 0]:.2f}dB, 频率={y[i, 1]:.2f}GHz, 增益={y[i, 2]:.2f}dBi")


if __name__ == "__main__":
    main()