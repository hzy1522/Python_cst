"""
基于PyTorch的贴片天线设计系统 - GAN增强版
专门针对:
- 输入: 贴片长宽
- 输出: S11最小值、对应频率、远区场增益
新增功能: GAN模型用于生成新的天线设计方案
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from python_hfss import *
import time
import os
from sklearn.metrics import r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 在 patch_antenna_design.py 中添加注意力模块
class AttentionModule(nn.Module):
    def __init__(self, input_dim, attention_dim=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        attention_weights = torch.softmax(self.attention(x), dim=1)
        weighted_output = torch.sum(x * attention_weights, dim=1)
        return weighted_output, attention_weights

class PatchAntennaDesignSystem:
    def __init__(self, device=None):
        """初始化贴片天线设计系统"""
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"使用设备: {self.device}")

        # 系统参数
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.input_dim = 2  # 贴片长度和宽度
        # self.output_dim = 204  # 201个S11点 + S11最小值 + 对应频率 + 远区场增益
        self.output_dim = 201  # 201个S11点
        self.noise_dim = 64  # GAN噪声维度

        # 参数和性能指标名称
        self.param_names = ['贴片长度(mm)', '贴片宽度(mm)']
        self.freq_points = np.linspace(2.0, 3.0, 201).tolist()
        s11_names = [f'{freq:.3f}' for freq in self.freq_points]
        # self.perf_names = ['S11最小值(dB)', '对应频率(GHz)', '远区场增益(dBi)'] + s11_names
        self.perf_names = s11_names

        # GAN相关属性
        self.generator = None
        self.discriminator = None
        self.performance_predictor = None
        self.gan_optimizers = None

        self.forward_gan_optimizers = None
        self.forward_discriminator = None
        self.forward_generator = None

    def plot_s11_comparison_advanced(self, patch_length, patch_width, csv_file_path,
                                     frequency_column=None,
                                     s11_column=None,
                                     # predict_s11_min=None,
                                     # predict_freq=None,
                                     # predict_gain=None,
                                     predict_s11_curve=None):
        """
        高级版本的S11对比绘制函数，可以指定频率和S11列

        参数:
        self: PatchAntennaDesignSystem实例
        patch_length: 贴片长度(mm)
        patch_width: 贴片宽度(mm)
        csv_file_path: 包含实际S11数据的CSV文件路径
        frequency_column: 频率列的名称或索引（可选）
        s11_column: S11数据列的名称或索引（可选）

        # predict_s11_min=None,           可选：
        # predict_freq=None,              给出这四个参数，则直接绘制对比图
        # predict_gain=None,
        predict_s11_curve=None,
        """
        # 1. 使用GAN模型预测S11曲线
        if predict_s11_curve is None:
            s11_curve = self.predict_s11_from_dimensions(patch_length,patch_width)
            # s11_curve, s11_min, freq_at_s11_min, far_field_gain = self.predict_s11_from_dimensions(patch_length, patch_width)
        else:
            s11_curve = predict_s11_curve

        # 2. 从CSV文件读取实际的S11数据
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file_path)
            print(f"CSV文件列名: {list(df.columns)}")

            # 如果指定了列名或索引
            if s11_column is not None:
                if isinstance(s11_column, str):
                    actual_s11_data = df[s11_column].values
                else:
                    actual_s11_data = df.iloc[:, s11_column].values
            else:
                # 默认使用第二列（索引为1）
                actual_s11_data = df.iloc[:, 1].values

            # 处理频率数据
            if frequency_column is not None:
                if isinstance(frequency_column, str):
                    frequencies = df[frequency_column].values
                else:
                    frequencies = df.iloc[:, frequency_column].values
            else:
                # 使用系统默认频率点
                frequencies = self.freq_points if hasattr(self, 'freq_points') else np.linspace(2.0, 3.0,
                                                                                                    len(actual_s11_data))

        except Exception as e:
            print(f"读取CSV文件失败: {e}")
            return

        # 3. 绘制对比图
        plt.figure(figsize=(12, 8))

        # 绘制预测的S11曲线
        pred_line = plt.plot(self.freq_points, s11_curve,
                             label=f'预测S11 (尺寸: {patch_length}×{patch_width}mm)',
                             linewidth=2, color='blue', marker='o', markersize=4)

        # 绘制CSV文件中的实际S11数据
        actual_line = plt.plot(frequencies[:len(actual_s11_data)], actual_s11_data,
                               label='实际S11 (CSV数据)',
                               linewidth=2, color='red', marker='s', markersize=4)

        # 图表设置
        plt.xlabel('频率 (GHz)')
        plt.ylabel('S11 (dB)')
        plt.title('预测S11 vs 实际S11对比')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # # 添加统计信息
        # plt.text(0.05, 0.95,
        #          f'预测S11最小值: {s11_min:.2f}dB\n对应频率: {freq_at_s11_min:.2f}GHz\n增益: {far_field_gain:.2f}dBi',
        #          transform=plt.gca().transAxes, verticalalignment='top',
        #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()

        # 计算误差
        min_length = min(len(s11_curve), len(actual_s11_data))
        mse = np.mean((s11_curve[:min_length] - actual_s11_data[:min_length]) ** 2)
        rmse = np.sqrt(mse)

        print(f"天线尺寸: {patch_length}mm × {patch_width}mm")
        print(f"CSV数据点数: {len(actual_s11_data)}")
        print(f"预测曲线点数: {len(s11_curve)}")
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")

    def load_csv_data(self, csv_file, param_cols=None, perf_cols=None):
        """
        从CSV文件加载贴片天线数据，适配不同格式的S11列名
        """
        print(f"从CSV文件加载数据: {csv_file}")

        # 读取CSV文件
        df = pd.read_csv(csv_file)
        print(f"数据形状: {df.shape}")
        print(f"列名数量: {len(df.columns)}")

        # 默认参数列名
        if param_cols is None:
            param_cols = ['patch_length', 'patch_width']

        # 构建性能列名 - 适配不同的S11列名格式 - 只保留S参数列
        if perf_cols is None:
            # 识别S11频率列（通过尝试转换为浮点数）
            s11_cols = []
            for col in df.columns:
                try:
                    col_value = float(col)
                    # 检查是否在2.0-3.0GHz范围内
                    if 2.0 <= col_value <= 3.0:
                        s11_cols.append(col)
                except (ValueError, TypeError):
                    continue

            # 按数值大小排序S11列
            s11_cols = sorted(s11_cols, key=lambda x: float(x))

            # 确保只有201个S参数点
            print(f"自动检测到 {len(s11_cols)} 个S11频率列")
            if len(s11_cols) > 201:
                s11_cols = s11_cols[:201]  # 取前201个

            perf_cols = s11_cols
            print(f"使用的性能列: {len(s11_cols)}个S参数列")

        # 验证参数列名
        for col in param_cols:
            if col not in df.columns:
                raise ValueError(f"参数列 '{col}' 不在CSV文件中")

        # 提取数据
        X_original = df[param_cols].values

        # 检查性能列
        available_perf_cols = [col for col in perf_cols if col in df.columns]
        missing_perf_cols = [col for col in perf_cols if col not in df.columns]

        if missing_perf_cols:
            print(f"警告: 以下性能列未找到: {missing_perf_cols[:10]}...")
            print(f"实际找到 {len(available_perf_cols)} 个性能列")

        if len(available_perf_cols) < 100:  # 至少应该有足够多的S11点
            raise ValueError(f"找到的性能列过少 ({len(available_perf_cols)})，请检查CSV文件格式")

        y_original = df[available_perf_cols].values

        # 更新output_dim以匹配实际数据
        self.output_dim = y_original.shape[1]
        print(f"更新output_dim为: {self.output_dim}")

        # 数据归一化
        X_scaled = self.scaler.fit_transform(X_original)
        y_scaled = self.target_scaler.fit_transform(y_original)

        print(f"参数数据形状: {X_original.shape}")
        print(f"性能数据形状: {y_original.shape}")

        return (torch.tensor(X_scaled, dtype=torch.float32),
                torch.tensor(y_scaled, dtype=torch.float32),
                X_original, y_original)

    # 在 patch_antenna_design.py 的 validate_training_data 方法中，需要考虑数据已经被归一化的情况
    def validate_training_data(self, X, y):
        """
        验证训练数据的质量，只检查S参数范围（已归一化）
        """
        print("数据质量检查:")

        # 检查是否为201个S参数点
        if y.shape[1] == 201:
            print(f"  S参数范围: [{np.min(y):.2f}, {np.max(y):.2f}] (归一化值)")

            # 对于归一化数据使用宽松的验证条件
            # 归一化后的数据通常在[0,1]范围内，但可能略有超出
            valid_indices = np.ones(len(y), dtype=bool)

            # 检查归一化范围是否合理（允许小范围超出）
            valid_indices &= (np.min(y, axis=1) >= -0.1) & (np.max(y, axis=1) <= 1.1)

            # 检查数据是否有明显异常（如全为相同值）
            std_vals = np.std(y, axis=1)
            valid_indices &= std_vals > 1e-6  # 标准差不能接近0

            print(f"有效数据点: {np.sum(valid_indices)}/{len(y)}")

            # 如果过滤后没有数据，发出警告并返回原始数据
            if np.sum(valid_indices) == 0:
                print("警告: 数据验证过滤掉了所有数据，将使用原始数据")
                return X, y

            return X[valid_indices], y[valid_indices]

        return X, y

    def generate_synthetic_data(self, num_samples=10000):
        """
        生成更真实的合成贴片天线数据

        参数:
        num_samples: 样本数量

        返回:
        X_scaled: 归一化的天线参数
        y_scaled: 归一化的天线性能指标
        X_original: 原始天线参数
        y_original: 原始性能指标
        """
        np.random.seed(42)
        print(f"生成合成贴片天线数据，样本数: {num_samples}")

        # 贴片天线参数
        patch_length = np.random.uniform(10, 50, num_samples)  # 10-50mm
        patch_width = np.random.uniform(10, 60, num_samples)   # 10-60mm

        X_original = np.column_stack([patch_length, patch_width])

        # 性能指标计算（基于电磁学原理）
        c = 3e8  # 光速
        epsilon_r = 4.4  # FR4介电常数
        h = 0.035e-3  # 标准GND厚度

        # 有效介电常数
        epsilon_eff = (epsilon_r + 1) / 2 + (epsilon_r - 1) / 2 * (1 + 12 * h / patch_width * 1e-3) ** (-0.5)

        # 谐振频率
        delta_l = 0.412 * h * (epsilon_eff + 0.3) * (patch_width * 1e-3 / h + 0.264) / \
                  ((epsilon_eff - 0.258) * (patch_width * 1e-3 / h + 0.8))
        L_eff = (patch_length * 1e-3) + 2 * delta_l
        freq = c / (2 * L_eff * np.sqrt(epsilon_eff)) / 1e9
        freq += np.random.normal(0, 0.05, num_samples)

        # S11最小值
        Z0 = 50
        Z_antenna = 377 * patch_width * 1e-3 / (2 * h * np.sqrt(epsilon_eff))
        reflection_coeff = (Z_antenna - Z0) / (Z_antenna + Z0)
        s11_min = 20 * np.log10(np.abs(reflection_coeff))
        s11_min += np.random.normal(0, 1.5, num_samples)
        s11_min = np.clip(s11_min, -40, -5)

        # 远区场增益
        gain = 2.15 + 0.01 * (patch_length + patch_width) + 0.5 * np.log10(patch_length * patch_width) + \
               np.random.normal(0, 0.4, num_samples)
        gain = np.clip(gain, 1, 12)

        # 生成201个频率点的S11数据
        s11_curves = []
        frequencies = np.array(self.freq_points)

        for i in range(num_samples):
            # 生成每个样本的S11曲线，以谐振频率为中心
            resonant_freq = freq[i]

            # 生成S11曲线，中心在谐振频率处
            s11_curve = []
            for f in frequencies:
                # 简化的S11模型：在谐振频率处最小，离谐振点越远值越大
                distance_from_resonance = abs(f - resonant_freq)
                # 使用洛伦兹函数形状模拟S11曲线
                s11_value = s11_min[i] + 20 * (distance_from_resonance / 0.1) ** 2
                s11_value = min(s11_value, 0)  # S11通常为负值或0
                s11_curve.append(s11_value)

            s11_curves.append(s11_curve)

        s11_curves = np.array(s11_curves)

        # 组合所有性能指标：201个S11点 + S11最小值 + 频率 + 增益
        y_original = np.column_stack([s11_curves, s11_min, freq, gain])

        # 数据归一化
        X_scaled = self.scaler.fit_transform(X_original)
        y_scaled = self.target_scaler.fit_transform(y_original)

        print(f"合成数据生成完成")
        print(f"参数数据形状: {X_original.shape}")
        print(f"性能数据形状: {y_original.shape}")

        return (torch.tensor(X_scaled, dtype=torch.float32),
                torch.tensor(y_scaled, dtype=torch.float32),
                X_original, y_original)

    def create_performance_predictor(self):
        """创建性能预测器网络"""
        class PerformancePredictor(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),

                    nn.Linear(128, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.4),

                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),

                    nn.Linear(128, output_dim)
                )

            def forward(self, x):
                return self.network(x)

        return PerformancePredictor(self.input_dim, self.output_dim).to(self.device)

    def create_gan_models(self):
        """创建GAN模型（生成器和判别器）"""

        # 生成器：输入噪声和目标性能，输出天线参数
        class Generator(nn.Module):
            def __init__(self, noise_dim, target_dim, output_dim):
                super().__init__()
                self.noise_dim = noise_dim
                self.target_dim = target_dim

                # 使用更深的网络结构和残差连接
                self.network = nn.Sequential(
                    nn.Linear(noise_dim + target_dim, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.3),

                    nn.Linear(512, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),

                    # 添加残差连接
                    nn.Linear(1024, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),

                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.3),

                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.2),

                    nn.Linear(256, output_dim)
                )
                # 为不同的输出维度定义激活函数
                self.s11_activation = nn.Tanh()  # S11: [-40, 0] -> 映射到 [-1, 1]
                self.freq_activation = nn.Sigmoid()  # 频率: [2.0, 3.0] -> 映射到 [0, 1]
                self.gain_activation = nn.Sigmoid()  # 增益: [0, 10] -> 映射到 [0, 1]

            def forward(self, noise, targets):
                # 拼接噪声和目标性能
                input_data = torch.cat([noise, targets], dim=1)
                output = self.network(input_data)
                # 对输出施加物理约束
                # 假设输出的前3个维度是 [S11, freq, gain]
                if output.shape[1] >= 3:
                    # S11 (dB): 从 [-1, 1] 映射到 [-40, 0]
                    output[:, 0] = self.s11_activation(output[:, 0]) * 20 - 20
                    # 频率 (GHz): 从 [-1, 1] 映射到 [1.5, 3.5]（放宽范围）
                    output[:, 1] = torch.sigmoid(output[:, 1]) * 2.0 + 1.5
                    # 增益 (dBi): 从 [-1, 1] 映射到 [-2, 12]（放宽范围）
                    output[:, 2] = torch.sigmoid(output[:, 2]) * 14 - 2

                    # 对于S11曲线部分(第4到204维度)，保持在合理范围内
                    if output.shape[1] > 3:
                        output[:, 3:] = torch.clamp(output[:, 3:], -80, 10)

                return output
                # return self.network(input_data)

        # 判别器：判断天线参数是否真实，并预测性能
        class Discriminator(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.input_dim = input_dim

                # 特征提取网络 - 使用更深的网络
                self.feature_extractor = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),

                    nn.Linear(256, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.4),

                    nn.Linear(512, 1024),
                    nn.BatchNorm1d(1024),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.4),

                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),

                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.2)
                )
                # 真实性判断头
                self.realness_head = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )

                # 性能预测头
                self.performance_head = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, output_dim)
                )

            def forward(self, x):
                features = self.feature_extractor(x)
                realness = self.realness_head(features)
                performance = self.performance_head(features)
                return realness, performance

        # 创建模型
        self.generator = Generator(
            noise_dim=self.noise_dim,
            target_dim=self.output_dim,
            output_dim=self.input_dim
        ).to(self.device)

        self.discriminator = Discriminator(
            input_dim=self.input_dim,
            output_dim=self.output_dim
        ).to(self.device)

        # 创建优化器
        self.gan_optimizers = {
            'generator': optim.AdamW(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4),
            'discriminator': optim.AdamW(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-4)
        }

        print("GAN模型创建完成:")
        print(f"生成器参数量: {sum(p.numel() for p in self.generator.parameters() if p.requires_grad):,}")
        print(f"判别器参数量: {sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad):,}")

    def create_forward_gan_models(self):
        """创建正向预测的GAN模型（参数->性能）"""

        # 生成器：输入天线参数，输出性能指标
        class ForwardGenerator(nn.Module):
            def __init__(self, input_dim, noise_dim, output_dim):
                super().__init__()
                # 使用注意力机制增强特征表达
                self.feature_extractor = nn.Sequential(
                    nn.Linear(input_dim + noise_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),

                    nn.Linear(256, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.4),

                    nn.Linear(512, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.4)
                )

                # 添加注意力模块
                self.attention = AttentionModule(1024)

                # 多头输出结构
                self.main_output = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.3),

                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.2),

                    nn.Linear(256, output_dim)
                )

            def forward(self, params, noise):
                input_data = torch.cat([params, noise], dim=1)
                features = self.feature_extractor(input_data)
                # 应用注意力机制
                attended_features, _ = self.attention(features)
                # 注意：这里需要调整，因为注意力机制会改变维度
                # 可能需要修改为直接使用 features 而不应用注意力
                output = self.main_output(features)
                return output

        # 判别器：判断(参数,性能)对是否真实
        class ForwardDiscriminator(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                # 确保输入维度计算正确
                # 输入维度应该是参数维度(2) + 性能维度(201) = 203

                self.network = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),

                    nn.Linear(512, 1024),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.4),

                    nn.Linear(1024, 1024),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.4),

                    nn.Linear(1024, 512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),

                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.2),

                    nn.Linear(256, output_dim),
                    nn.Sigmoid()
                )

            def forward(self, params, performances):
                # 确保输入维度正确
                input_data = torch.cat([params, performances], dim=1)
                return self.network(input_data)

        # 创建模型
        self.forward_generator = ForwardGenerator(
            input_dim=self.input_dim,
            noise_dim=self.noise_dim,
            output_dim=self.output_dim
        ).to(self.device)

        self.forward_discriminator = ForwardDiscriminator(
            input_dim = self.input_dim + self.output_dim,  # 2 + 201 = 203
            output_dim = 1
        ).to(self.device)

        # 创建优化器
        self.forward_gan_optimizers = {
            'generator': optim.AdamW(self.forward_generator.parameters(),
                                    lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4),
            'discriminator': optim.AdamW(self.forward_discriminator.parameters(),
                                       lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-4)
        }
        print("正向GAN模型创建完成:")
        print(f"生成器参数量: {sum(p.numel() for p in self.forward_generator.parameters() if p.requires_grad):,}")
        print(f"判别器参数量: {sum(p.numel() for p in self.forward_discriminator.parameters() if p.requires_grad):,}")

    def train_performance_predictor(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=128):
        """训练性能预测器"""
        print("\n训练性能预测器...")

        self.performance_predictor = self.create_performance_predictor()

        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.performance_predictor.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            self.performance_predictor.train()
            train_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # 前向传播
                outputs = self.performance_predictor(inputs)
                loss = criterion(outputs, targets)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.performance_predictor.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)

            # 验证
            self.performance_predictor.eval()
            val_loss = 0.0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.performance_predictor(inputs)
                    val_loss += criterion(outputs, targets).item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)
            history['val_loss'].append(val_loss)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.performance_predictor.state_dict(), 'best_performance_predictor.pth')
                torch.save(self.performance_predictor.state_dict(),'./models/best_performance_predictor.pth')
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        print(f"性能预测器训练完成！最佳验证损失: {best_val_loss:.6f}")
        return history

    def augment_geometric_parameters(self, X_original, y_original, augmentation_factor=2):
        """
        对几何参数进行扰动增强
        """
        augmented_X = []
        augmented_y = []

        for i in range(len(X_original)):
            patch_length, patch_width = X_original[i]

            # 原始数据
            augmented_X.append([patch_length, patch_width])
            augmented_y.append(y_original[i])

            # 生成增强样本
            for _ in range(augmentation_factor):
                # 添加小幅度随机扰动 (±5%)
                new_length = patch_length * (1 + np.random.normal(0, 0.05))
                new_width = patch_width * (1 + np.random.normal(0, 0.05))

                # 保持物理合理性约束
                new_length = np.clip(new_length, 10, 50)
                new_width = np.clip(new_width, 10, 60)

                augmented_X.append([new_length, new_width])

                # 对应性能指标也需要调整
                augmented_performance = self._adjust_performance_for_augmented_params(
                    y_original[i], patch_length, patch_width, new_length, new_width
                )
                augmented_y.append(augmented_performance)

        return np.array(augmented_X), np.array(augmented_y)

    def _adjust_performance_for_augmented_params(self, original_perf, old_length, old_width, new_length, new_width):
        """
        根据几何参数变化调整性能指标
        """
        # 复制原始性能数据
        adjusted_perf = np.copy(original_perf)

        # 确保数组长度正确
        if len(adjusted_perf) != self.output_dim:
            # 重新创建正确大小的数组
            new_adjusted_perf = np.zeros(self.output_dim)
            # 复制能容纳的元素
            copy_size = min(len(adjusted_perf), self.output_dim)
            new_adjusted_perf[:copy_size] = adjusted_perf[:copy_size]
            adjusted_perf = new_adjusted_perf

        # 基于尺寸变化调整谐振频率
        freq_ratio = np.sqrt((old_length * old_width) / (new_length * new_width))
        adjusted_perf[1] *= freq_ratio  # freq_at_s11_min

        # 轻微调整S11最小值和增益
        adjusted_perf[0] += np.random.normal(0, 1.0)  # s11_min
        adjusted_perf[2] += np.random.normal(0, 0.5)  # far_field_gain

        # 重新生成S11曲线（确保目标数组有足够空间）
        if len(adjusted_perf) > 3:
            s11_curve = self._generate_s11_curve(adjusted_perf[0], adjusted_perf[1])
            # 确保S11曲线长度正确
            if len(s11_curve) == self.output_dim - 3:
                adjusted_perf[3:] = s11_curve
            else:
                # 如果长度不匹配，进行截断或填充
                if len(s11_curve) > self.output_dim - 3:
                    adjusted_perf[3:] = s11_curve[:self.output_dim - 3]
                else:
                    adjusted_perf[3:3+len(s11_curve)] = s11_curve
                    # 填充剩余位置
                    adjusted_perf[3+len(s11_curve):] = 0.0

        return adjusted_perf


    def augment_with_physical_parameters(self, X_original, y_original):
        """
        考虑材料参数变化的数据增强
        """
        augmented_X = []
        augmented_y = []

        # FR4材料参数范围
        epsilon_r_range = [4.2, 4.6]  # 介电常数范围
        substrate_thickness_range = [0.03, 0.04]  # 基板厚度范围(mm)

        for i in range(len(X_original)):
            patch_length, patch_width = X_original[i]
            original_perf = y_original[i]

            # 原始样本
            augmented_X.append([patch_length, patch_width])
            augmented_y.append(original_perf)

            # 生成不同材料参数的样本
            for _ in range(3):
                # 随机材料参数
                epsilon_r = np.random.uniform(epsilon_r_range[0], epsilon_r_range[1])
                substrate_thickness = np.random.uniform(substrate_thickness_range[0], substrate_thickness_range[1])

                # 基于新参数重新计算性能
                new_performance = self._recalculate_performance_with_material_params(
                    patch_length, patch_width, epsilon_r, substrate_thickness, original_perf
                )

                augmented_X.append([patch_length, patch_width])
                augmented_y.append(new_performance)

        return np.array(augmented_X), np.array(augmented_y)

    def _recalculate_performance_with_material_params(self, length, width, epsilon_r, thickness, original_perf):
        """
        基于材料参数重新计算性能
        """
        # 计算有效介电常数
        epsilon_eff = (epsilon_r + 1) / 2 + (epsilon_r - 1) / 2 * (1 + 12 * thickness / width) ** (-0.5)

        # 调整谐振频率
        delta_l = 0.412 * thickness * (epsilon_eff + 0.3) * (width / thickness + 0.264) / \
                  ((epsilon_eff - 0.258) * (width / thickness + 0.8))
        L_eff = (length * 1e-3) + 2 * delta_l
        freq = 3e8 / (2 * L_eff * np.sqrt(epsilon_eff)) / 1e9

        # 调整其他参数
        new_perf = np.copy(original_perf)
        new_perf[1] = freq  # 更新频率
        new_perf[0] += np.random.normal(0, 0.8)  # 调整S11
        new_perf[2] += np.random.normal(0, 0.3)  # 调整增益

        # 重新生成S11曲线
        if len(new_perf) > 3:
            new_perf[3:] = self._generate_s11_curve(new_perf[0], new_perf[1])

        return new_perf

    def augment_boundary_exploration(self, X_original, y_original):
        """
        在参数边界生成极端但合理的样本
        """
        augmented_X = []
        augmented_y = []

        # 参数边界
        length_bounds = [10, 50]
        width_bounds = [10, 60]

        for i in range(len(X_original)):
            augmented_X.append(X_original[i])
            augmented_y.append(y_original[i])

        # 生成边界样本
        boundary_samples = [
            [length_bounds[0], width_bounds[0]],  # 最小尺寸
            [length_bounds[1], width_bounds[1]],  # 最大尺寸
            [length_bounds[0], width_bounds[1]],  # 长短组合
            [length_bounds[1], width_bounds[0]],  # 短长组合
            [(length_bounds[0] + length_bounds[1]) / 2, width_bounds[0]],  # 中等长度，最小宽度
            [(length_bounds[0] + length_bounds[1]) / 2, width_bounds[1]],  # 中等长度，最大宽度
            [length_bounds[0], (width_bounds[0] + width_bounds[1]) / 2],   # 最小长度，中等宽度
            [length_bounds[1], (width_bounds[0] + width_bounds[1]) / 2],   # 最大长度，中等宽度
        ]

        for params in boundary_samples:
            length, width = params
            # 基于最近邻样本生成性能数据
            nearest_idx = self._find_nearest_sample(X_original, length, width)
            base_performance = y_original[nearest_idx]

            # 调整性能参数
            adjusted_performance = self._adjust_performance_for_boundary(
                base_performance, length, width, X_original[nearest_idx][0], X_original[nearest_idx][1]
            )

            augmented_X.append([length, width])
            augmented_y.append(adjusted_performance)

        return np.array(augmented_X), np.array(augmented_y)

    def _find_nearest_sample(self, X_original, target_length, target_width):
        """
        找到最近邻的样本
        """
        distances = np.sqrt((X_original[:, 0] - target_length) ** 2 + (X_original[:, 1] - target_width) ** 2)
        return np.argmin(distances)

    def _adjust_performance_for_boundary(self, base_perf, new_length, new_width, old_length, old_width):
        """
        调整边界样本的性能参数
        """
        adjusted_perf = np.copy(base_perf)

        # 基于尺寸比例调整频率
        size_ratio = np.sqrt((new_length * new_width) / (old_length * old_width))
        adjusted_perf[1] *= size_ratio

        # 添加合理的噪声
        adjusted_perf[0] += np.random.normal(0, 1.5)
        adjusted_perf[2] += np.random.normal(0, 0.8)

        # 重新生成S11曲线
        if len(adjusted_perf) > 3:
            adjusted_perf[3:] = self._generate_s11_curve(adjusted_perf[0], adjusted_perf[1])

        return adjusted_perf

    def train_gan(self, X_train, y_train, epochs=3000, batch_size=128, forward_gan=True, train_both=False):
        """训练GAN模型"""

        def compute_gradient_penalty(discriminator, real_samples, fake_samples, forward_gan=True):
            # 处理元组参数
            if isinstance(real_samples, tuple):
                batch_size = real_samples[0].size(0)
            else:
                batch_size = real_samples.size(0)

            alpha = torch.rand(batch_size, 1).to(self.device)

            if forward_gan:
                # 对于正向GAN，需要拼接参数和性能
                if isinstance(real_samples, tuple) and isinstance(fake_samples, tuple):
                    interpolates = alpha * torch.cat([real_samples[0], real_samples[1]], dim=1) + \
                                  (1 - alpha) * torch.cat([fake_samples[0], fake_samples[1]], dim=1)
                else:
                    interpolates = alpha * torch.cat([real_samples[0], real_samples[1]], dim=1) + \
                                  (1 - alpha) * torch.cat([fake_samples[0], fake_samples[1]], dim=1)
            else:
                if isinstance(real_samples, tuple):
                    interpolates = alpha * real_samples[0] + (1 - alpha) * fake_samples[0]
                else:
                    interpolates = alpha * real_samples + (1 - alpha) * fake_samples

            interpolates = interpolates.requires_grad_(True)

            if forward_gan:
                if isinstance(real_samples, tuple):
                    disc_interpolates = discriminator(interpolates[:, :real_samples[0].size(1)],
                                                    interpolates[:, real_samples[0].size(1):])
                else:
                    disc_interpolates = discriminator(interpolates[:, :real_samples[0].size(1)],
                                                    interpolates[:, real_samples[0].size(1):])
            else:
                disc_interpolates = discriminator(interpolates)[0]

            gradients = torch.autograd.grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones(interpolates.size(0), 1).to(self.device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            return gradient_penalty


        # 如果需要同时训练两个模型或者指定训练正向GAN
        if train_both or forward_gan:
            print("正向GAN模型训练...")
            if self.forward_generator is None or self.forward_discriminator is None:
                self.create_forward_gan_models()

        # 如果需要同时训练两个模型或者指定训练反向GAN
        if train_both or not forward_gan:
            print("反向GAN模型训练...")
            if self.generator is None or self.discriminator is None:
                self.create_gan_models()

        # 训练性能预测器（如果尚未训练）
        if self.performance_predictor is None:
            # 先训练性能预测器
            X_tr, X_vl, y_tr, y_vl = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            self.train_performance_predictor(X_tr, y_tr, X_vl, y_vl)

        print("\n开始训练GAN模型...")

        # 在 train_gan 方法中，创建数据加载器之前添加：
        print("验证训练数据质量...")
        X_train_np = X_train.cpu().numpy() if torch.is_tensor(X_train) else X_train
        y_train_np = y_train.cpu().numpy() if torch.is_tensor(y_train) else y_train

        X_train_valid, y_train_valid = self.validate_training_data(X_train_np, y_train_np)

        # 重新创建张量
        X_train = torch.tensor(X_train_valid, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y_train_valid, dtype=torch.float32, device=self.device)
        # 在 train_gan 方法中，数据清洗后添加检查
        print(f"清洗后数据量: {X_train.shape[0]}")

        if X_train.shape[0] == 0:
            raise ValueError("数据清洗后没有剩余数据，请检查数据验证逻辑或原始数据质量")

        # 如果数据量过少，可以考虑使用合成数据补充
        if X_train.shape[0] < 50:  # 设置一个最小阈值
            print(f"警告: 清洗后数据量较少 ({X_train.shape[0]}), 补充合成数据")
            X_synthetic, y_synthetic, _, _ = self.generate_synthetic_data(num_samples=1000)
            X_train = torch.cat([X_train, torch.tensor(X_synthetic, dtype=torch.float32, device=self.device)], dim=0)
            y_train = torch.cat([y_train, torch.tensor(y_synthetic, dtype=torch.float32, device=self.device)], dim=0)
            print(f"补充合成数据后数据量: {X_train.shape[0]}")

        # 创建数据加载器
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 损失函数
        adversarial_loss = nn.BCELoss()
        performance_loss = nn.MSELoss()

        # 真实和虚假标签
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # 训练历史
        history = {
            'forward': {
                'generator_loss': [],
                'discriminator_loss': [],
                'adversarial_loss': [],
                'performance_loss': []
            },
            'reverse': {
                'generator_loss': [],
                'discriminator_loss': [],
                'adversarial_loss': [],
                'performance_loss': []
            }
        }

        # 如果同时训练两个模型
        if train_both:
            print("同时训练正向和反向GAN模型...")

            # 分别获取模型和优化器
            forward_generator = self.forward_generator
            forward_discriminator = self.forward_discriminator
            forward_optimizers = self.forward_gan_optimizers

            reverse_generator = self.generator
            reverse_discriminator = self.discriminator
            reverse_optimizers = self.gan_optimizers

            for epoch in range(epochs):
                for i, (real_params, real_perfs) in enumerate(dataloader):
                    batch_size = real_params.size(0)

                    # 调整标签大小
                    current_real_labels = real_labels[:batch_size]
                    current_fake_labels = fake_labels[:batch_size]

                    # ---------------------
                    #  训练正向GAN
                    # ---------------------
                    # 训练正向判别器
                    forward_discriminator.train()
                    forward_generator.eval()

                    real_params_f = real_params.to(self.device)
                    real_perfs_f = real_perfs.to(self.device)

                    # 判别器对真实数据的预测
                    real_pred_f = forward_discriminator(real_params_f, real_perfs_f)

                    # 生成虚假数据
                    noise_f = torch.randn(batch_size, self.noise_dim, device=self.device)
                    fake_perfs_f = forward_generator(real_params_f, noise_f)

                    # 判别器对虚假数据的预测
                    fake_pred_f = forward_discriminator(real_params_f, fake_perfs_f.detach())

                    # 添加梯度惩罚
                    d_loss_real_f = adversarial_loss(real_pred_f, current_real_labels)
                    d_loss_fake_f = adversarial_loss(fake_pred_f, current_fake_labels)
                    # 添加梯度惩罚
                    gradient_penalty = compute_gradient_penalty(forward_discriminator,
                                                                (real_params_f, real_perfs_f),
                                                                (real_params_f, fake_perfs_f.detach()),
                                                                forward_gan=True)
                    d_loss_f = (d_loss_real_f + d_loss_fake_f) * 0.5 + 10 * gradient_penalty
                    # 优化正向判别器
                    forward_optimizers['discriminator'].zero_grad()
                    d_loss_f.backward()
                    torch.nn.utils.clip_grad_norm_(forward_discriminator.parameters(), max_norm=1.0)
                    forward_optimizers['discriminator'].step()

                    # 训练正向生成器
                    forward_generator.train()
                    forward_discriminator.eval()

                    # 生成新的虚假数据
                    noise_f = torch.randn(batch_size, self.noise_dim, device=self.device)
                    fake_perfs_f = forward_generator(real_params_f, noise_f)

                    # 判别器对虚假数据的预测
                    fake_pred_f = forward_discriminator(real_params_f, fake_perfs_f)

                    g_loss_adv_f = adversarial_loss(fake_pred_f, current_real_labels)
                    # 使用加权损失函数，更重视关键性能指标
                    weights = torch.ones_like(real_perfs_f)
                    weights[:, 0] = 3.0  # S11最小值权重更高
                    weights[:, 1] = 1.5  # 频率权重中等
                    weights[:, 2] = 2.0  # 增益权重较高

                    weighted_perf_loss = torch.mean(weights * (fake_perfs_f - real_perfs_f) ** 2)
                    g_loss_f = g_loss_adv_f + weighted_perf_loss * 2.0

                    # 优化正向生成器
                    forward_optimizers['generator'].zero_grad()
                    g_loss_f.backward()
                    torch.nn.utils.clip_grad_norm_(forward_generator.parameters(), max_norm=1.0)
                    forward_optimizers['generator'].step()

                    # ---------------------
                    #  训练反向GAN
                    # ---------------------
                    # 训练反向判别器
                    reverse_discriminator.train()
                    reverse_generator.eval()

                    # 真实数据
                    real_params_r = real_params.to(self.device)
                    real_perfs_r = real_perfs.to(self.device)

                    # 判别器对真实数据的预测
                    real_pred_r, pred_perfs_real_r = reverse_discriminator(real_params_r)

                    # 生成虚假数据
                    noise_r = torch.randn(batch_size, self.noise_dim, device=self.device)
                    fake_params_r = reverse_generator(noise_r, real_perfs_r)

                    # 判别器对虚假数据的预测
                    fake_pred_r, pred_perfs_fake_r = reverse_discriminator(fake_params_r.detach())

                    # 计算判别器损失
                    d_loss_real_r = adversarial_loss(real_pred_r, current_real_labels)
                    d_loss_fake_r = adversarial_loss(fake_pred_r, current_fake_labels)
                    d_loss_perf_real_r = performance_loss(pred_perfs_real_r, real_perfs_r)
                    d_loss_perf_fake_r = performance_loss(pred_perfs_fake_r, real_perfs_r)

                    d_loss_r = (d_loss_real_r + d_loss_fake_r) * 0.5 + (d_loss_perf_real_r + d_loss_perf_fake_r) * 0.5

                    # 优化反向判别器
                    reverse_optimizers['discriminator'].zero_grad()
                    d_loss_r.backward()
                    torch.nn.utils.clip_grad_norm_(reverse_discriminator.parameters(), max_norm=1.0)
                    reverse_optimizers['discriminator'].step()

                    # 训练反向生成器
                    reverse_generator.train()
                    reverse_discriminator.eval()

                    # 生成新的虚假数据
                    noise_r = torch.randn(batch_size, self.noise_dim, device=self.device)
                    fake_params_r = reverse_generator(noise_r, real_perfs_r)

                    # 判别器对虚假数据的预测
                    fake_pred_r, pred_perfs_r = reverse_discriminator(fake_params_r)

                    # 使用性能预测器评估生成的参数
                    gen_perfs_r = self.performance_predictor(fake_params_r)

                    # 计算生成器损失
                    g_loss_adv_r = adversarial_loss(fake_pred_r, current_real_labels)
                    g_loss_perf_r = performance_loss(gen_perfs_r, real_perfs_r)
                    g_loss_r = g_loss_adv_r + g_loss_perf_r * 2.0  # 更重视性能匹配

                    # 优化反向生成器
                    reverse_optimizers['generator'].zero_grad()
                    g_loss_r.backward()
                    torch.nn.utils.clip_grad_norm_(reverse_generator.parameters(), max_norm=1.0)
                    reverse_optimizers['generator'].step()

                # 记录历史 - 正向GAN
                # history['forward']['generator_loss'].append(g_loss_f.item())
                # history['forward']['discriminator_loss'].append(d_loss_f.item())
                # history['forward']['adversarial_loss'].append(g_loss_adv_f.item())
                # history['forward']['performance_loss'].append(g_loss_perf_f.item())
                history['forward']['generator_loss'].append(g_loss_f.item())
                history['forward']['discriminator_loss'].append(d_loss_f.item())
                history['forward']['adversarial_loss'].append(g_loss_adv_f.item())
                history['forward']['performance_loss'].append(weighted_perf_loss.item())  # 使用正确的变量
                # 记录历史 - 反向GAN
                history['reverse']['generator_loss'].append(g_loss_r.item())
                history['reverse']['discriminator_loss'].append(d_loss_r.item())
                history['reverse']['adversarial_loss'].append(g_loss_adv_r.item())
                history['reverse']['performance_loss'].append(g_loss_perf_r.item())

                # 打印进度
                if (epoch + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}]")
                    print(f"  正向GAN - G Loss: {g_loss_f.item():.6f}, D Loss: {d_loss_f.item():.6f}")
                    print(f"  反向GAN - G Loss: {g_loss_r.item():.6f}, D Loss: {d_loss_r.item():.6f}")

            # 保存模型
            # 保存正向GAN模型
            torch.save(self.forward_generator.state_dict(), 'forward_gan_generator.pth')
            torch.save(self.forward_discriminator.state_dict(), 'forward_gan_discriminator.pth')
            torch.save(self.forward_generator.state_dict(), './models/forward_gan_generator.pth')
            torch.save(self.forward_discriminator.state_dict(), './models/forward_gan_discriminator.pth')

            # 保存反向GAN模型
            torch.save(self.generator.state_dict(), 'gan_generator.pth')
            torch.save(self.discriminator.state_dict(), 'gan_discriminator.pth')
            torch.save(self.generator.state_dict(), './models/gan_generator.pth')
            torch.save(self.discriminator.state_dict(), './models/gan_discriminator.pth')

            print("正向和反向GAN模型训练完成并保存！")

        else:
            # 原有的单向训练逻辑
            # 根据训练类型选择正确的模型和优化器
            if forward_gan:
                generator = self.forward_generator
                discriminator = self.forward_discriminator
                optimizers = self.forward_gan_optimizers
                model_type = "正向"
            else:
                generator = self.generator
                discriminator = self.discriminator
                optimizers = self.gan_optimizers
                model_type = "反向"

            for epoch in range(epochs):
                for i, (real_params, real_perfs) in enumerate(dataloader):
                    batch_size = real_params.size(0)

                    # 调整标签大小
                    current_real_labels = real_labels[:batch_size]
                    current_fake_labels = fake_labels[:batch_size]

                    if not forward_gan:
                        # ---------------------
                        #  训练反向GAN判别器
                        # ---------------------
                        discriminator.train()
                        generator.eval()

                        # 真实数据
                        real_params = real_params.to(self.device)
                        real_perfs = real_perfs.to(self.device)

                        # 判别器对真实数据的预测
                        real_pred, pred_perfs_real = discriminator(real_params)

                        # 生成虚假数据
                        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                        fake_params = generator(noise, real_perfs)

                        # 判别器对虚假数据的预测
                        fake_pred, pred_perfs_fake = discriminator(fake_params.detach())

                        # 计算判别器损失
                        d_loss_real = adversarial_loss(real_pred, current_real_labels)
                        d_loss_fake = adversarial_loss(fake_pred, current_fake_labels)
                        d_loss_perf_real = performance_loss(pred_perfs_real, real_perfs)
                        d_loss_perf_fake = performance_loss(pred_perfs_fake, real_perfs)

                        d_loss = (d_loss_real + d_loss_fake) * 0.5 + (d_loss_perf_real + d_loss_perf_fake) * 0.5

                        # 优化判别器
                        optimizers['discriminator'].zero_grad()
                        d_loss.backward()
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                        optimizers['discriminator'].step()

                        # ---------------------
                        #  训练反向GAN生成器
                        # ---------------------
                        generator.train()
                        discriminator.eval()

                        # 生成新的虚假数据
                        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                        fake_params = generator(noise, real_perfs)

                        # 判别器对虚假数据的预测
                        fake_pred, pred_perfs = discriminator(fake_params)

                        # 使用性能预测器评估生成的参数
                        gen_perfs = self.performance_predictor(fake_params)

                        # 计算生成器损失
                        g_loss_adv = adversarial_loss(fake_pred, current_real_labels)
                        g_loss_perf = performance_loss(gen_perfs, real_perfs)
                        g_loss = g_loss_adv + g_loss_perf * 2.0  # 更重视性能匹配

                        # 优化生成器
                        optimizers['generator'].zero_grad()
                        g_loss.backward()
                        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                        optimizers['generator'].step()

                        g_loss_adv_item = g_loss_adv.item()
                        g_loss_perf_item = g_loss_perf.item()

                    else:
                        # ---------------------
                        #  训练正向GAN判别器
                        # ---------------------
                        discriminator.train()
                        generator.eval()

                        real_params = real_params.to(self.device)
                        real_perfs = real_perfs.to(self.device)

                        # 判别器对真实数据的预测
                        real_pred = discriminator(real_params, real_perfs)

                        # 生成虚假数据
                        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                        fake_perfs = generator(real_params, noise)

                        # 判别器对虚假数据的预测
                        fake_pred = discriminator(real_params, fake_perfs.detach())

                        # 计算判别器损失
                        d_loss_real = adversarial_loss(real_pred, current_real_labels)
                        d_loss_fake = adversarial_loss(fake_pred, current_fake_labels)
                        d_loss = (d_loss_real + d_loss_fake) * 0.5

                        # 优化判别器
                        optimizers['discriminator'].zero_grad()
                        d_loss.backward()
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                        optimizers['discriminator'].step()

                        # ---------------------
                        #  训练正向GAN生成器
                        # ---------------------
                        generator.train()
                        discriminator.eval()

                        # 生成新的虚假数据
                        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                        fake_perfs = generator(real_params, noise)

                        # 判别器对虚假数据的预测
                        fake_pred = discriminator(real_params, fake_perfs)

                        # 计算生成器损失
                        g_loss_adv = adversarial_loss(fake_pred, current_real_labels)
                        g_loss_perf = performance_loss(fake_perfs, real_perfs)
                        g_loss = g_loss_adv + g_loss_perf

                        # 优化生成器
                        optimizers['generator'].zero_grad()
                        g_loss.backward()
                        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                        optimizers['generator'].step()

                        g_loss_adv_item = g_loss_adv.item()
                        g_loss_perf_item = g_loss_perf.item()

                # 记录历史
                if forward_gan:
                    history['forward']['generator_loss'].append(g_loss.item())
                    history['forward']['discriminator_loss'].append(d_loss.item())
                    history['forward']['adversarial_loss'].append(g_loss_adv_item)
                    history['forward']['performance_loss'].append(g_loss_perf_item)
                else:
                    history['reverse']['generator_loss'].append(g_loss.item())
                    history['reverse']['discriminator_loss'].append(d_loss.item())
                    history['reverse']['adversarial_loss'].append(g_loss_adv_item)
                    history['reverse']['performance_loss'].append(g_loss_perf_item)

                # 打印进度
                if (epoch + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], "
                          f"G Loss: {g_loss.item():.6f}, "
                          f"D Loss: {d_loss.item():.6f}, "
                          f"Adv Loss: {g_loss_adv_item:.6f}, "
                          f"Perf Loss: {g_loss_perf_item:.6f}")

            # 保存GAN模型
            if forward_gan:
                torch.save(self.forward_generator.state_dict(), 'forward_gan_generator.pth')
                torch.save(self.forward_discriminator.state_dict(), 'forward_gan_discriminator.pth')
                torch.save(self.forward_generator.state_dict(), './models/forward_gan_generator.pth')
                torch.save(self.forward_discriminator.state_dict(), './models/forward_gan_discriminator.pth')
                print(f"{model_type}GAN模型训练完成并保存！")
            else:
                torch.save(self.generator.state_dict(), 'gan_generator.pth')
                torch.save(self.discriminator.state_dict(), 'gan_discriminator.pth')
                torch.save(self.generator.state_dict(), './models/gan_generator.pth')
                torch.save(self.discriminator.state_dict(), './models/gan_discriminator.pth')
                print(f"{model_type}GAN模型训练完成并保存！")

        return history

    def generate_antenna_designs(self, target_performances, num_samples=10):
        """
        使用GAN生成符合目标性能的天线设计，精确处理所有性能指标

        参数:
        target_performances: 目标性能指标，可以是：
                            1. [[S11, freq, gain], ...] - 简化形式
                            2. [[full_performance_vector], ...] - 完整204维形式
        num_samples: 每个目标生成的样本数量

        返回:
        generated_designs: 生成的天线设计参数
        predicted_performances: 预测的完整性能指标
        """
        # 检查预处理器是否已拟合
        try:
            # 尝试访问预处理器的属性来验证是否已拟合
            _ = self.scaler.scale_
            _ = self.target_scaler.scale_
        except AttributeError:
            raise RuntimeError("数据预处理器未就绪，请确保已加载训练时的预处理器状态")

        # 确保模型已加载
        if self.generator is None:
            try:
                self.create_gan_models()
                state_dict = torch.load('gan_generator.pth', map_location=self.device)
                self.generator.load_state_dict(state_dict)
                print("成功加载预训练的反向GAN生成器")
            except Exception as e:
                print(f"加载反向GAN模型失败: {e}")

        if self.forward_generator is None:
            try:
                self.create_forward_gan_models()
                state_dict = torch.load('forward_gan_generator.pth', map_location=self.device)
                self.forward_generator.load_state_dict(state_dict)
                print("成功加载预训练的正向GAN生成器")
            except Exception as e:
                print(f"加载正向GAN模型失败: {e}")

        if self.performance_predictor is None:
            try:
                self.performance_predictor = self.create_performance_predictor()
                state_dict = torch.load('best_performance_predictor.pth', map_location=self.device)
                self.performance_predictor.load_state_dict(state_dict)
                print("成功加载预训练的性能预测器")
            except Exception as e:
                print(f"加载性能预测器失败: {e}")

        print(f"\n使用GAN生成天线设计（精确处理）...")
        print(f"目标性能数量: {len(target_performances)}")
        print(f"每个目标生成样本数: {num_samples}")

        # 确保模型处于评估模式
        if self.generator is not None:
            self.generator.eval()
        if self.forward_generator is not None:
            self.forward_generator.eval()
        if self.performance_predictor is not None:
            self.performance_predictor.eval()

        all_designs = []
        all_performances = []

        with torch.no_grad():
            for target_perf in target_performances:
                # 处理不同形式的目标性能
                if len(target_perf) == 3:
                    # 简化形式：[S11, freq, gain]
                    print(f"\n生成目标性能: S11={target_perf[0]:.2f}dB, freq={target_perf[1]:.2f}GHz, gain={target_perf[2]:.2f}dBi")

                    # 构造完整的性能向量（204维）
                    full_target_perf = np.zeros(self.output_dim)
                    full_target_perf[0] = target_perf[0]  # S11最小值
                    full_target_perf[1] = target_perf[1]  # 对应频率
                    full_target_perf[2] = target_perf[2]  # 远区场增益

                    # 智能生成S11曲线（基于主要指标）
                    full_target_perf[3:] = self._generate_s11_curve(target_perf[0], target_perf[1])

                elif len(target_perf) == self.output_dim:
                    # 完整形式：204维性能向量
                    print(f"\n生成完整目标性能（{self.output_dim}维）")
                    full_target_perf = np.array(target_perf)
                else:
                    print(f"警告: 目标性能维度不匹配，期望3或{self.output_dim}维，实际{len(target_perf)}维")
                    continue

                # 使用反向GAN生成天线参数
                target_tensor = torch.tensor(
                    self.target_scaler.transform([full_target_perf]),
                    dtype=torch.float32,
                    device=self.device
                ).repeat(num_samples, 1)

                # 生成噪声
                noise = torch.randn(num_samples, self.noise_dim, device=self.device)

                # 生成天线参数（使用反向GAN）
                if self.generator is not None:
                    generated_params = self.generator(noise, target_tensor)
                else:
                    print("警告: 反向GAN生成器未初始化，无法生成设计")
                    return np.array([]), np.array([])

                # 反归一化参数到原始范围
                generated_params_np = generated_params.cpu().numpy()
                param_min = self.scaler.mean_ - 2 * self.scaler.scale_
                param_max = self.scaler.mean_ + 2 * self.scaler.scale_
                generated_params_denorm = (generated_params_np + 1) * (param_max - param_min) / 2 + param_min

                # 使用正向GAN验证生成参数的性能
                if self.forward_generator is not None:
                    params_normalized = self.scaler.transform(generated_params_denorm)
                    params_tensor = torch.tensor(params_normalized, dtype=torch.float32, device=self.device)
                    forward_noise = torch.randn(num_samples, self.noise_dim, device=self.device)
                    forward_predicted_perfs = self.forward_generator(params_tensor, forward_noise)
                    forward_predicted_perfs_denorm = self.target_scaler.inverse_transform(forward_predicted_perfs.cpu().numpy())
                else:
                    print("警告: 正向GAN生成器未初始化，使用性能预测器代替")
                    if self.performance_predictor is not None:
                        params_normalized = self.scaler.transform(generated_params_denorm)
                        params_tensor = torch.tensor(params_normalized, dtype=torch.float32, device=self.device)
                        forward_predicted_perfs = self.performance_predictor(params_tensor)
                        forward_predicted_perfs_denorm = self.target_scaler.inverse_transform(forward_predicted_perfs.cpu().numpy())
                    else:
                        forward_predicted_perfs_denorm = np.tile(full_target_perf, (num_samples, 1))

                # 精确计算性能误差（使用完整性能向量）
                target_perf_array = full_target_perf
                performance_errors = np.mean(np.abs(forward_predicted_perfs_denorm - target_perf_array), axis=1)
                best_indices = np.argsort(performance_errors)[:3]

                print(f"生成完成！最佳3个设计:")
                for j, idx in enumerate(best_indices):
                    design = generated_params_denorm[idx]
                    perf = forward_predicted_perfs_denorm[idx]
                    error = performance_errors[idx]
                    print(f"  设计 {j+1}: 长度={design[0]:.2f}mm, 宽度={design[1]:.2f}mm, "
                          f"S11={perf[0]:.2f}dB, 频率={perf[1]:.2f}GHz, 增益={perf[2]:.2f}dBi, 误差={error:.3f}")

                all_designs.extend(generated_params_denorm[best_indices])
                # 保存完整性能指标
                all_performances.extend(forward_predicted_perfs_denorm[best_indices])

        return np.array(all_designs), np.array(all_performances)

    def _generate_s11_curve(self, s11_min, resonant_freq):
        """
        根据S11最小值和共振频率生成完整的S11曲线

        参数:
        s11_min: S11最小值
        resonant_freq: 共振频率

        返回:
        s11_curve: 正确数量的频率点的S11值
        """
        frequencies = np.array(self.freq_points)

        # 确保生成正确的点数
        if len(frequencies) != self.output_dim - 3:
            # 重新生成频率点
            frequencies = np.linspace(2.0, 3.0, self.output_dim - 3)

        # 主谐振响应 (洛伦兹函数)
        Q_factor = 25.0 + np.random.normal(0, 5)  # 添加一些变化
        Q_factor = max(Q_factor, 10)  # 限制最小Q值

        delta_f = frequencies - resonant_freq
        lorentzian = 1.0 / (1.0 + (2 * Q_factor * delta_f / resonant_freq) ** 2)

        # 转换为dB，以s11_min为参考
        s11_curve = s11_min + 10 * np.log10(lorentzian + 1e-12)

        # 添加高频衰减特性
        frequency_factor = 1 + 0.5 * (frequencies - 2.0)  # 高频衰减
        s11_curve = s11_curve - 2 * (frequencies - resonant_freq) ** 2 * frequency_factor

        # 限制范围并添加微小噪声
        s11_curve = np.clip(s11_curve, -60, 0)
        noise = np.random.normal(0, 0.2, len(s11_curve))
        s11_curve = s11_curve + noise

        return np.clip(s11_curve, -60, 0)

    def visualize_gan_results(self, gan_history, generated_designs=None, generated_performances=None):
        """可视化GAN训练结果"""
        os.makedirs('patch_antenna_results', exist_ok=True)

        # 检查 gan_history 结构并相应处理
        if 'forward' in gan_history or 'reverse' in gan_history:
            # 新的双模型结构
            has_forward = 'forward' in gan_history and len(gan_history['forward']['generator_loss']) > 0
            has_reverse = 'reverse' in gan_history and len(gan_history['reverse']['generator_loss']) > 0
        else:
            # 旧的单模型结构
            has_forward = False
            has_reverse = len(gan_history.get('generator_loss', [])) > 0

        # GAN训练损失曲线
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        if has_forward:
            plt.plot(gan_history['forward']['generator_loss'], label='正向生成器损失')
            plt.plot(gan_history['forward']['discriminator_loss'], label='正向判别器损失')
        if has_reverse:
            if not has_forward:
                plt.plot(gan_history['reverse']['generator_loss'], label='反向生成器损失')
                plt.plot(gan_history['reverse']['discriminator_loss'], label='反向判别器损失')
            else:
                plt.plot(gan_history['reverse']['generator_loss'], label='反向生成器损失', linestyle='--')
                plt.plot(gan_history['reverse']['discriminator_loss'], label='反向判别器损失', linestyle='--')
        elif not has_forward and not has_reverse:
            # 兼容旧结构
            if 'generator_loss' in gan_history:
                plt.plot(gan_history['generator_loss'], label='生成器损失')
            if 'discriminator_loss' in gan_history:
                plt.plot(gan_history['discriminator_loss'], label='判别器损失')

        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('GAN训练损失曲线')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        if has_forward:
            plt.plot(gan_history['forward']['adversarial_loss'], label='正向对抗损失')
        if has_reverse:
            if not has_forward:
                plt.plot(gan_history['reverse']['adversarial_loss'], label='反向对抗损失')
            else:
                plt.plot(gan_history['reverse']['adversarial_loss'], label='反向对抗损失', linestyle='--')
        elif not has_forward and not has_reverse and 'adversarial_loss' in gan_history:
            plt.plot(gan_history['adversarial_loss'], label='对抗损失')

        plt.xlabel('Epoch')
        plt.ylabel('对抗损失')
        plt.title('对抗损失演变')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        if has_forward:
            plt.plot(gan_history['forward']['performance_loss'], label='正向性能损失')
        if has_reverse:
            if not has_forward:
                plt.plot(gan_history['reverse']['performance_loss'], label='反向性能损失')
            else:
                plt.plot(gan_history['reverse']['performance_loss'], label='反向性能损失', linestyle='--')
        elif not has_forward and not has_reverse and 'performance_loss' in gan_history:
            plt.plot(gan_history['performance_loss'], label='性能损失')

        plt.xlabel('Epoch')
        plt.ylabel('性能损失')
        plt.title('性能损失演变')
        plt.legend()
        plt.grid(True)

        # GAN收敛指标
        plt.subplot(2, 2, 4)
        if has_forward:
            forward_convergence = np.array(gan_history['forward']['generator_loss']) / np.array(gan_history['forward']['discriminator_loss'])
            plt.plot(forward_convergence, label='正向G/D损失比')
        if has_reverse:
            reverse_convergence = np.array(gan_history['reverse']['generator_loss']) / np.array(gan_history['reverse']['discriminator_loss'])
            if has_forward:
                plt.plot(reverse_convergence, label='反向G/D损失比', linestyle='--')
            else:
                plt.plot(reverse_convergence, label='反向G/D损失比')
        elif not has_forward and not has_reverse:
            # 兼容旧结构
            if 'generator_loss' in gan_history and 'discriminator_loss' in gan_history:
                convergence = np.array(gan_history['generator_loss']) / np.array(gan_history['discriminator_loss'])
                plt.plot(convergence, label='G/D损失比')

        plt.xlabel('Epoch')
        plt.ylabel('G/D 损失比')
        plt.title('GAN收敛指标')
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('patch_antenna_results/gan_training_monitor.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 如果有生成的设计，可视化生成结果
        if generated_designs is not None and generated_performances is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 生成的参数分布
            ax1 = axes[0, 0]
            ax1.scatter(generated_designs[:, 0], generated_designs[:, 1], alpha=0.6, s=20)
            ax1.set_xlabel('贴片长度 (mm)')
            ax1.set_ylabel('贴片宽度 (mm)')
            ax1.set_title('生成的天线参数分布')
            ax1.grid(True)

            # S11分布
            ax2 = axes[0, 1]
            ax2.hist(generated_performances[:, 0], bins=30, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('S11最小值 (dB)')
            ax2.set_ylabel('频次')
            ax2.set_title('生成设计的S11分布')
            ax2.grid(True)

            # 频率分布
            ax3 = axes[1, 0]
            ax3.hist(generated_performances[:, 1], bins=30, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('对应频率 (GHz)')
            ax3.set_ylabel('频次')
            ax3.set_title('生成设计的频率分布')
            ax3.grid(True)

            # 增益分布
            ax4 = axes[1, 1]
            ax4.hist(generated_performances[:, 2], bins=30, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('远区场增益 (dBi)')
            ax4.set_ylabel('频次')
            ax4.set_title('生成设计的增益分布')
            ax4.grid(True)

            plt.tight_layout()
            plt.savefig('patch_antenna_results/gan_generated_designs.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("GAN可视化结果已保存到 patch_antenna_results 目录")

    def predict_s11_from_dimensions(self, patch_length, patch_width):
        """
        使用训练好的GAN模型根据天线尺寸预测S11结果

        参数:
        system: PatchAntennaDesignSystem实例
        patch_length: 贴片长度(mm)
        patch_width: 贴片宽度(mm)

        返回:
        predicted_s11_curve: 201个频率点的S11值
        # s11_min: S11最小值
        # freq_at_s11_min: 对应频率
        # far_field_gain: 远区场增益
        """
        """
            根据贴片尺寸预测S11曲线和关键参数
        """
        # 确保模型处于评估模式
        if self.forward_generator is not None:
            self.forward_generator.eval()
        if self.performance_predictor is not None:
            self.performance_predictor.eval()

        # 输入预处理
        dimensions = np.array([[patch_length, patch_width]], dtype=np.float32)
        scaled_dimensions = self.scaler.transform(dimensions)

        # 转换为张量
        input_tensor = torch.tensor(scaled_dimensions, dtype=torch.float32).to(self.device)

        # 模型预测
        with torch.no_grad():
            #生成噪声向量
            noise = torch.randn(1, self.noise_dim, device=self.device)
            prediction = self.forward_generator(input_tensor, noise)
            # prediction = self.forward_generator(input_tensor)

        # # 反归一化
        # prediction_np = prediction.cpu().numpy()
        # unscaled_prediction = self.target_scaler.inverse_transform(prediction_np)
        # 反归一化
        prediction_np = prediction.cpu().numpy()
        s11_curve = self.target_scaler.inverse_transform(prediction_np)[0]

        # 确保输出维度正确
        if len(s11_curve) != 201:
            # 如果维度不匹配，只取前201个点或用0填充
            if len(s11_curve) > 201:
                s11_curve = s11_curve[:201]
            else:
                padded_curve = np.zeros(201)
                padded_curve[:len(s11_curve)] = s11_curve
                s11_curve = padded_curve
        print("预测的S11曲线:", s11_curve)
        return s11_curve

if __name__ == "__main__":
    # 演示使用
    system = PatchAntennaDesignSystem()

    # 检查命令行参数
    import sys
    use_gan = False
    csv_file = None

    for arg in sys.argv[1:]:
        if arg.endswith('.csv'):
            csv_file = arg
        elif arg == '--gan':
            use_gan = True

    if csv_file:
        print(f"使用CSV文件: {csv_file}")
        system.design_workflow(
            csv_file=csv_file,
            model_type='resnet',
            epochs=300,
            use_gan=use_gan
        )
    else:
        print("使用合成数据进行演示")
        print("使用方法:")
        print("  python patch_antenna_design.py [CSV文件路径] [--gan]")
        print("  --gan: 使用GAN模型生成天线设计")
        print()

        # 使用GAN模型
        system.design_workflow(
            model_type='resnet',
            epochs=300,
            use_synthetic_data=True,
            use_gan=use_gan
        )

    print("\n设计流程全部完成！")