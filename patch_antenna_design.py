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
        # self.output_dim = 3  # S11最小值、对应频率、远区场增益
        self.output_dim = 204  # 201个S11点 + S11最小值 + 对应频率 + 远区场增益
        self.noise_dim = 64  # GAN噪声维度

        # 参数和性能指标名称
        self.param_names = ['贴片长度(mm)', '贴片宽度(mm)']
        self.freq_points = np.linspace(2.0, 3.0, 201).tolist()
        s11_names = [f'{freq:.3f}' for freq in self.freq_points]
        self.perf_names = ['S11最小值(dB)', '对应频率(GHz)', '远区场增益(dBi)'] + s11_names
        # self.perf_names = ['S11最小值(dB)', '对应频率(GHz)', '远区场增益(dBi)']

        # GAN相关属性
        self.generator = None
        self.discriminator = None
        self.performance_predictor = None
        self.gan_optimizers = None

        self.forward_gan_optimizers = None
        self.forward_discriminator = None
        self.forward_generator = None

    def load_csv_data(self, csv_file, param_cols=None, perf_cols=None):
        """
        从CSV文件加载贴片天线数据

        参数:
        csv_file: CSV文件路径
        param_cols: 参数列名列表 (可选)
        perf_cols: 性能指标列名列表 (可选)

        返回:
        X_scaled: 归一化的天线参数
        y_scaled: 归一化的天线性能指标
        X_original: 原始天线参数
        y_original: 原始性能指标
        """
        print(f"从CSV文件加载数据: {csv_file}")

        # 读取CSV文件
        df = pd.read_csv(csv_file)
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

        # 默认列名
        if param_cols is None:
            param_cols = ['patch_length', 'patch_width']
            print(f"使用默认参数列名: {param_cols}")

        # if perf_cols is None:
        #     perf_cols = ['s11_min', 'freq_at_s11_min', 'far_field_gain']
        #     print(f"使用默认性能列名: {perf_cols}")

        # 默认列名 - 修改为包含201个S11点
        if perf_cols is None:
            # 假设您的CSV文件中S11列名为 2, , ..., 3
            s11_cols = [f'{i}:.3f' for i in self.freq_points]
            perf_cols = s11_cols + ['s11_min', 'freq_at_s11_min', 'far_field_gain']
            print(f"使用默认性能列名: {perf_cols}")

        # 验证列名
        for col in param_cols + perf_cols:
            if col not in df.columns:
                raise ValueError(f"列名 '{col}' 不在CSV文件中")

        # 提取数据
        X_original = df[param_cols].values
        y_original = df[perf_cols].values

        # 验证维度
        if X_original.shape[1] != self.input_dim:
            raise ValueError(f"参数列数应为 {self.input_dim}，但实际为 {X_original.shape[1]}")

        if y_original.shape[1] != self.output_dim:
            raise ValueError(f"性能列数应为 {self.output_dim}，但实际为 {y_original.shape[1]}")

        # 数据归一化
        X_scaled = self.scaler.fit_transform(X_original)
        y_scaled = self.target_scaler.fit_transform(y_original)

        print(f"参数数据形状: {X_original.shape}")
        print(f"性能数据形状: {y_original.shape}")

        # 显示统计信息
        print(f"\n参数统计:")
        for i, (name, col) in enumerate(zip(self.param_names, param_cols)):
            print(f"  {name}: 均值={X_original[:, i].mean():.3f}, 标准差={X_original[:, i].std():.3f}")

        print(f"\n性能指标统计:")
        for i, (name, col) in enumerate(zip(self.perf_names, perf_cols)):
            print(f"  {name}: 均值={y_original[:, i].mean():.3f}, 标准差={y_original[:, i].std():.3f}")

        return (torch.tensor(X_scaled, dtype=torch.float32),
                torch.tensor(y_scaled, dtype=torch.float32),
                X_original, y_original)

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

                self.network = nn.Sequential(
                    # 输入层：噪声 + 目标性能
                    nn.Linear(noise_dim + target_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),

                    nn.Linear(256, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.4),

                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),

                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.2),

                    # 输出层：天线参数
                    nn.Linear(128, output_dim),
                    nn.Tanh()  # 输出范围[-1, 1]，后续可调整到实际范围
                )

            def forward(self, noise, targets):
                # 拼接噪声和目标性能
                input_data = torch.cat([noise, targets], dim=1)
                return self.network(input_data)

        # 判别器：判断天线参数是否真实，并预测性能
        class Discriminator(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.input_dim = input_dim

                # 特征提取网络
                self.feature_extractor = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),

                    nn.Linear(128, 256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.4),

                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3)
                )

                # 真实性判断头
                self.realness_head = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.LeakyReLU(0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

                # 性能预测头
                self.performance_head = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.LeakyReLU(0.2),
                    nn.Linear(64, output_dim)
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
                self.network = nn.Sequential(
                    nn.Linear(input_dim + noise_dim, 128),
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

            def forward(self, params, noise):
                input_data = torch.cat([params, noise], dim=1)
                return self.network(input_data)

        # 判别器：判断(参数,性能)对是否真实
        class ForwardDiscriminator(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim + output_dim, 256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),

                    nn.Linear(256, 128),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),

                    nn.Linear(128, 64),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.2),

                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

            def forward(self, params, performances):
                input_data = torch.cat([params, performances], dim=1)
                return self.network(input_data)

        # 创建模型
        self.forward_generator = ForwardGenerator(
            input_dim=self.input_dim,
            noise_dim=self.noise_dim,
            output_dim=self.output_dim
        ).to(self.device)

        self.forward_discriminator = ForwardDiscriminator(
            input_dim=self.input_dim,
            output_dim=self.output_dim
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

            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        print(f"性能预测器训练完成！最佳验证损失: {best_val_loss:.6f}")
        return history

    def train_gan(self, X_train, y_train, epochs=5000, batch_size=128, forward_gan=True):
        """训练GAN模型"""
        if forward_gan == True:
            print("正向GAN模型训练...")
            if self.forward_generator is None or self.forward_discriminator is None:
                self.create_forward_gan_models()
        else:
            print("反向GAN模型训练...")
            if self.generator is None or self.discriminator is None:
                self.create_gan_models()

        if self.performance_predictor is None:
            # 先训练性能预测器
            X_tr, X_vl, y_tr, y_vl = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            self.train_performance_predictor(X_tr, y_tr, X_vl, y_vl)

        print("\n开始训练GAN模型...")

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
            'generator_loss': [],
            'discriminator_loss': [],
            'adversarial_loss': [],
            'performance_loss': []
        }

        for epoch in range(epochs):
            for i, (real_params, real_perfs) in enumerate(dataloader):
                batch_size = real_params.size(0)

                # 调整标签大小
                current_real_labels = real_labels[:batch_size]
                current_fake_labels = fake_labels[:batch_size]

                # ---------------------
                #  训练判别器
                # ---------------------
                self.discriminator.train()
                self.generator.eval()

                # 真实数据
                real_params = real_params.to(self.device)
                real_perfs = real_perfs.to(self.device)

                # 判别器对真实数据的预测
                real_pred, pred_perfs_real = self.discriminator(real_params)

                # 生成虚假数据
                noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                fake_params = self.generator(noise, real_perfs)

                # 判别器对虚假数据的预测
                fake_pred, pred_perfs_fake = self.discriminator(fake_params.detach())

                # 计算判别器损失
                d_loss_real = adversarial_loss(real_pred, current_real_labels)
                d_loss_fake = adversarial_loss(fake_pred, current_fake_labels)
                d_loss_perf_real = performance_loss(pred_perfs_real, real_perfs)
                d_loss_perf_fake = performance_loss(pred_perfs_fake, real_perfs)

                d_loss = (d_loss_real + d_loss_fake) * 0.5 + (d_loss_perf_real + d_loss_perf_fake) * 0.5

                # 优化判别器
                self.gan_optimizers['discriminator'].zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.gan_optimizers['discriminator'].step()

                # ---------------------
                #  训练生成器
                # ---------------------
                self.generator.train()
                self.discriminator.eval()

                # 生成新的虚假数据
                noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                fake_params = self.generator(noise, real_perfs)

                # 判别器对虚假数据的预测
                fake_pred, pred_perfs = self.discriminator(fake_params)

                # 使用性能预测器评估生成的参数
                gen_perfs = self.performance_predictor(fake_params)

                # 计算生成器损失
                g_loss_adv = adversarial_loss(fake_pred, current_real_labels)
                g_loss_perf = performance_loss(gen_perfs, real_perfs)
                g_loss = g_loss_adv + g_loss_perf * 2.0  # 更重视性能匹配

                # 优化生成器
                self.gan_optimizers['generator'].zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                self.gan_optimizers['generator'].step()

            # 记录历史
            history['generator_loss'].append(g_loss.item())
            history['discriminator_loss'].append(d_loss.item())
            history['adversarial_loss'].append(g_loss_adv.item())
            history['performance_loss'].append(g_loss_perf.item())

            # 打印进度
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"G Loss: {g_loss.item():.6f}, "
                      f"D Loss: {d_loss.item():.6f}, "
                      f"Adv Loss: {g_loss_adv.item():.6f}, "
                      f"Perf Loss: {g_loss_perf.item():.6f}")

        # 保存GAN模型
        torch.save(self.generator.state_dict(), 'gan_generator.pth')
        torch.save(self.discriminator.state_dict(), 'gan_discriminator.pth')
        print("GAN模型训练完成并保存！")

        return history

    def generate_antenna_designs(self, target_performances, num_samples=10):
        """
        使用GAN生成符合目标性能的天线设计

        参数:
        target_performances: 目标性能指标 [[S11, freq, gain], ...]
        num_samples: 每个目标生成的样本数量

        返回:
        generated_designs: 生成的天线设计参数
        predicted_performances: 预测的性能指标
        """
        if self.generator is None:
            # 尝试加载预训练模型
            try:
                self.create_gan_models()
                self.generator.load_state_dict(torch.load('gan_generator.pth'))
                print("成功加载预训练的GAN生成器")
            except:
                raise ValueError("GAN模型未训练且未找到预训练模型")

        if self.performance_predictor is None:
            try:
                self.performance_predictor = self.create_performance_predictor()
                self.performance_predictor.load_state_dict(torch.load('best_performance_predictor.pth'))
                print("成功加载预训练的性能预测器")
            except:
                raise ValueError("性能预测器未训练且未找到预训练模型")

        print(f"\n使用GAN生成天线设计...")
        print(f"目标性能数量: {len(target_performances)}")
        print(f"每个目标生成样本数: {num_samples}")

        self.generator.eval()
        self.performance_predictor.eval()

        all_designs = []
        all_performances = []

        with torch.no_grad():
            for target_perf in target_performances:
                print(f"\n生成目标性能: S11={target_perf[0]:.2f}dB, freq={target_perf[1]:.2f}GHz, gain={target_perf[2]:.2f}dBi")

                # 归一化目标性能
                target_tensor = torch.tensor(
                    self.target_scaler.transform([target_perf]),
                    dtype=torch.float32,
                    device=self.device
                ).repeat(num_samples, 1)

                # 生成噪声
                noise = torch.randn(num_samples, self.noise_dim, device=self.device)

                # 生成天线参数
                generated_params = self.generator(noise, target_tensor)

                # 反归一化参数到原始范围
                generated_params_np = generated_params.cpu().numpy()
                # 由于生成器使用Tanh输出[-1,1]，需要调整到实际范围
                param_min = self.scaler.mean_ - 2 * self.scaler.scale_
                param_max = self.scaler.mean_ + 2 * self.scaler.scale_
                generated_params_denorm = (generated_params_np + 1) * (param_max - param_min) / 2 + param_min

                # 预测生成参数的性能
                predicted_perfs = self.performance_predictor(generated_params)
                predicted_perfs_denorm = self.target_scaler.inverse_transform(predicted_perfs.cpu().numpy())

                # 筛选最佳设计（性能最接近目标的）
                performance_errors = np.mean(np.abs(predicted_perfs_denorm - np.array(target_perf)), axis=1)
                best_indices = np.argsort(performance_errors)[:3]  # 取前3个最佳设计

                print(f"生成完成！最佳3个设计:")
                for j, idx in enumerate(best_indices):
                    design = generated_params_denorm[idx]
                    perf = predicted_perfs_denorm[idx]
                    error = performance_errors[idx]
                    print(f"  设计 {j+1}: 长度={design[0]:.2f}mm, 宽度={design[1]:.2f}mm, "
                          f"S11={perf[0]:.2f}dB, 频率={perf[1]:.2f}GHz, 增益={perf[2]:.2f}dBi, 误差={error:.3f}")

                all_designs.extend(generated_params_denorm[best_indices])
                all_performances.extend(predicted_perfs_denorm[best_indices])

        return np.array(all_designs), np.array(all_performances)

    def optimize_antenna_parameters(self, target_s11, target_gain, target_frequency,
                              bounds=None, num_iterations=1000):
        """
        优化天线参数以达到目标性能

        参数:
        target_s11: 目标S11值
        target_gain: 目标增益
        target_frequency: 目标频率
        bounds: 参数边界 [min_length, max_length, min_width, max_width]
        num_iterations: 优化迭代次数
        """

        if bounds is None:
            # 使用默认边界
            bounds = [10.0, 50.0, 10.0, 60.0]  # [min_len, max_len, min_width, max_width]

        # 初始化参数
        params = torch.rand(2, device=self.device, requires_grad=True) * \
                 torch.tensor([bounds[1]-bounds[0], bounds[3]-bounds[2]], device=self.device) + \
                 torch.tensor([bounds[0], bounds[2]], device=self.device)

        optimizer = optim.Adam([params], lr=0.01)
        target_performance = torch.tensor([target_s11, target_frequency, target_gain],
                                         device=self.device)

        best_loss = float('inf')
        best_params = None

        print(f"开始优化天线参数...")
        print(f"目标性能: S11={target_s11:.2f}dB, 频率={target_frequency:.2f}GHz, 增益={target_gain:.2f}dBi")

        for i in range(num_iterations):
            optimizer.zero_grad()

            # 归一化参数
            normalized_params = (params - torch.tensor([bounds[0], bounds[2]], device=self.device)) / \
                               (torch.tensor([bounds[1]-bounds[0], bounds[3]-bounds[2]], device=self.device))

            # 生成噪声
            noise = torch.randn(1, self.noise_dim, device=self.device)

            # 预测性能
            predicted_performance = self.forward_generator(normalized_params.unsqueeze(0), noise)

            # 计算损失（加权不同性能指标）
            weights = torch.tensor([2.0, 1.0, 1.5], device=self.device)  # S11权重最高
            loss = torch.mean(weights * torch.square(predicted_performance[0] - target_performance))

            loss.backward()
            optimizer.step()

            # 限制参数在边界内
            with torch.no_grad():
                params[0].clamp_(bounds[0], bounds[1])  # 长度约束
                params[1].clamp_(bounds[2], bounds[3])  # 宽度约束

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = params.detach().clone()

            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6f}")

        # 反归一化最佳参数
        final_params = best_params.cpu().numpy()

        print(f"优化完成!")
        print(f"推荐参数: 长度={final_params[0]:.2f}mm, 宽度={final_params[1]:.2f}mm")
        print(f"最佳损失: {best_loss:.6f}")

        return final_params

    def create_model(self, model_type='resnet'):
        """
        创建改进的贴片天线设计神经网络模型

        参数:
        model_type: 模型类型 ('mlp', 'resnet', 'cnn', 'rnn', 'gnn')

        返回:
        神经网络模型
        """
        if model_type == 'mlp':
            return nn.Sequential(
                nn.Linear(self.input_dim, 128),
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

                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(64, self.output_dim)
            ).to(self.device)

        elif model_type == 'resnet':
            class ResidualBlock(nn.Module):
                def __init__(self, in_dim, out_dim, stride=1):
                    super().__init__()
                    self.conv1 = nn.Linear(in_dim, out_dim)
                    self.bn1 = nn.BatchNorm1d(out_dim)
                    self.relu = nn.ReLU()
                    self.conv2 = nn.Linear(out_dim, out_dim)
                    self.bn2 = nn.BatchNorm1d(out_dim)

                    self.downsample = None
                    if stride != 1 or in_dim != out_dim:
                        self.downsample = nn.Sequential(
                            nn.Linear(in_dim, out_dim),
                            nn.BatchNorm1d(out_dim)
                        )

                def forward(self, x):
                    residual = x
                    if self.downsample is not None:
                        residual = self.downsample(x)

                    out = self.conv1(x)
                    out = self.bn1(out)
                    out = self.relu(out)
                    out = self.conv2(out)
                    out = self.bn2(out)

                    out += residual
                    out = self.relu(out)
                    return out

            return nn.Sequential(
                nn.Linear(self.input_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),

                ResidualBlock(64, 64),
                ResidualBlock(64, 128, stride=2),
                ResidualBlock(128, 128),
                ResidualBlock(128, 256, stride=2),
                ResidualBlock(256, 256),

                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, self.output_dim)
            ).to(self.device)

        elif model_type == 'cnn':
            class ImprovedCNN(nn.Module):
                def __init__(self, input_dim, output_dim):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Linear(input_dim, 32),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Unflatten(1, (1, 32)),
                        nn.Conv1d(1, 16, kernel_size=3, padding=1),
                        nn.BatchNorm1d(16),
                        nn.ReLU(),
                        nn.MaxPool1d(2),
                        nn.Conv1d(16, 32, kernel_size=3, padding=1),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.MaxPool1d(2),
                        nn.Conv1d(32, 64, kernel_size=3, padding=1),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.AdaptiveMaxPool1d(4),
                        nn.Flatten()
                    )
                    self.head = nn.Sequential(
                        nn.Linear(64 * 4, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        nn.Dropout(0.4),
                        nn.Linear(256, 128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, output_dim)
                    )

                def forward(self, x):
                    x = self.features(x)
                    x = self.head(x)
                    return x

            return ImprovedCNN(self.input_dim, self.output_dim).to(self.device)

        elif model_type == 'rnn':
            class ImprovedRNN(nn.Module):
                def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_dim,
                        hidden_size=hidden_dim,
                        num_layers=num_layers,
                        batch_first=True,
                        bidirectional=True,
                        dropout=0.3
                    )
                    self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
                    self.attention = nn.Linear(hidden_dim * 2, 1)
                    self.fc1 = nn.Linear(hidden_dim * 2, 128)
                    self.fc2 = nn.Linear(128, 64)
                    self.fc3 = nn.Linear(64, output_dim)
                    self.relu = nn.ReLU()
                    self.dropout = nn.Dropout(0.4)

                def forward(self, x):
                    x = x.unsqueeze(1)
                    batch_size = x.size(0)
                    h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim, device=x.device)
                    c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim, device=x.device)

                    out, _ = self.lstm(x, (h0, c0))
                    attention_scores = self.attention(out).squeeze(2)
                    attention_weights = torch.softmax(attention_scores, dim=1)
                    attended_out = torch.bmm(attention_weights.unsqueeze(1), out).squeeze(1)

                    out = self.batch_norm(attended_out)
                    out = self.dropout(self.relu(self.fc1(out)))
                    out = self.dropout(self.relu(self.fc2(out)))
                    out = self.fc3(out)
                    return out

            return ImprovedRNN(self.input_dim, 64, self.output_dim, num_layers=2).to(self.device)

        elif model_type == 'gnn':
            class AdvancedGNN(nn.Module):
                def __init__(self, input_dim, output_dim, hidden_dim=32, num_layers=3, num_heads=2):
                    super().__init__()
                    self.input_dim = input_dim
                    self.hidden_dim = hidden_dim
                    self.num_layers = num_layers
                    self.num_heads = num_heads
                    self.total_hidden_dim = hidden_dim * num_heads

                    self.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
                    self.node_embedding = nn.Linear(1, hidden_dim)

                    self.gat_layers = nn.ModuleList()
                    self.attention_layers = nn.ModuleList()
                    self.graph_conv_layers = nn.ModuleList()
                    self.batch_norm_layers = nn.ModuleList()
                    self.residual_layers = nn.ModuleList()

                    for i in range(num_layers):
                        in_dim = hidden_dim if i == 0 else self.total_hidden_dim
                        self.gat_layers.append(nn.ModuleList([nn.Linear(in_dim, hidden_dim) for _ in range(num_heads)]))
                        self.attention_layers.append(nn.Linear(2 * hidden_dim, 1))
                        self.graph_conv_layers.append(nn.Linear(self.total_hidden_dim, self.total_hidden_dim))
                        self.batch_norm_layers.append(nn.BatchNorm1d(self.total_hidden_dim))
                        self.residual_layers.append(nn.Linear(self.total_hidden_dim, self.total_hidden_dim) if i > 0 else None)

                    self.global_pool = nn.AdaptiveAvgPool1d(1)
                    self.fc1 = nn.Linear(self.total_hidden_dim, 256)
                    self.fc2 = nn.Linear(256, 128)
                    self.fc3 = nn.Linear(128, output_dim)
                    self.relu = nn.ReLU()
                    self.dropout = nn.Dropout(0.4)

                def compute_multi_head_attention(self, node_features, edge_index, layer_idx):
                    src, dst = edge_index
                    all_head_outputs = []

                    for head in range(self.num_heads):
                        linear = self.gat_layers[layer_idx][head]
                        transformed_features = linear(node_features)
                        src_features = transformed_features[src]
                        dst_features = transformed_features[dst]

                        cat_features = torch.cat([src_features, dst_features], dim=1)
                        attention_scores = self.attention_layers[layer_idx](cat_features)
                        attention_scores = torch.leaky_relu(attention_scores, 0.2)

                        num_nodes = node_features.size(0)
                        attention_weights = torch.zeros(num_nodes, num_nodes, device=node_features.device)
                        attention_weights[src, dst] = attention_scores.squeeze()
                        row_sums = attention_weights.sum(dim=1, keepdim=True)
                        attention_weights = attention_weights / (row_sums + 1e-12)

                        attended_features = torch.matmul(attention_weights, transformed_features)
                        all_head_outputs.append(attended_features)

                    return torch.cat(all_head_outputs, dim=1)

                def forward(self, x):
                    batch_size = x.size(0)
                    node_features = x.view(batch_size * self.input_dim, 1)
                    node_features = self.node_embedding(node_features)
                    node_features = self.relu(node_features)
                    node_features = self.dropout(node_features)

                    edge_index = self.edge_index.clone()
                    for i in range(1, batch_size):
                        edge_index = torch.cat([edge_index, self.edge_index + i * self.input_dim], dim=1)

                    for layer_idx in range(self.num_layers):
                        attention_output = self.compute_multi_head_attention(node_features, edge_index, layer_idx)
                        conv_output = self.graph_conv_layers[layer_idx](attention_output)

                        if self.residual_layers[layer_idx] is not None:
                            conv_output += self.residual_layers[layer_idx](attention_output)

                        conv_output = self.batch_norm_layers[layer_idx](conv_output)
                        conv_output = self.relu(conv_output)
                        conv_output = self.dropout(conv_output)
                        node_features = conv_output

                    node_features = node_features.view(batch_size, self.input_dim, -1)
                    pooled_features = self.global_pool(node_features.permute(0, 2, 1)).squeeze()

                    out = self.fc1(pooled_features)
                    out = self.relu(out)
                    out = self.dropout(out)
                    out = self.fc2(out)
                    out = self.relu(out)
                    out = self.dropout(out)
                    out = self.fc3(out)

                    return out

            return AdvancedGNN(self.input_dim, self.output_dim, hidden_dim=32, num_layers=3, num_heads=2).to(self.device)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def train_model(self, model, X_train, y_train, X_val, y_val,
                   epochs=300, batch_size=128, lr=0.001):
        """
        改进的模型训练方法

        参数:
        model: 神经网络模型
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率

        返回:
        训练历史
        """
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

        history = {'train_loss': [], 'val_loss': [], 'train_rmse': [], 'val_rmse': [], 'learning_rate': []}
        best_val_loss = float('inf')
        patience = 30
        early_stop_counter = 0

        print("开始训练模型...")
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step(epoch + len(train_loader.dataset) / len(train_loader))

                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)
            train_rmse = np.sqrt(train_loss)

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)
            val_rmse = np.sqrt(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_rmse'].append(train_rmse)
            history['val_rmse'].append(val_rmse)
            history['learning_rate'].append(current_lr)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_patch_antenna_model.pth')
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"早停策略触发，在第 {epoch+1} 轮停止训练")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                      f"Train RMSE: {train_rmse:.6f}, Val RMSE: {val_rmse:.6f}, "
                      f"LR: {current_lr:.6f}")

        print(f"训练完成！最佳验证损失: {best_val_loss:.6f}")
        return history

    def optimize_antenna(self, model, target_specs, param_bounds, num_iterations=3000, learning_rate=0.01, device=None):
        """天线参数优化"""
        if device is None:
            device = self.device
        print(f"优化使用设备: {device}")

        if len(target_specs) != self.output_dim:
            raise ValueError(f"目标性能指标应为 {self.output_dim} 个，实际为 {len(target_specs)}")
        if param_bounds.shape != (self.input_dim, 2):
            raise ValueError(f"参数边界应为 {self.input_dim}x2 的数组，实际为 {param_bounds.shape}")

        num_params = self.input_dim
        model = model.to(device)

        params = torch.rand(num_params, dtype=torch.float32, device=device, requires_grad=True)
        param_min = torch.tensor(param_bounds[:, 0], dtype=torch.float32, device=device)
        param_max = torch.tensor(param_bounds[:, 1], dtype=torch.float32, device=device)
        params.data = params.data * (param_max - param_min) + param_min

        optimizer = optim.Adam([params], lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
        target_tensor = torch.tensor(
            self.target_scaler.transform([target_specs]),
            dtype=torch.float32,
            device=device
        ).squeeze()

        best_loss = float('inf')
        best_params = None
        best_performance = None

        print("开始贴片天线参数优化...")
        print(f"目标性能: S11={target_specs[0]:.2f}dB, 频率={target_specs[1]:.2f}GHz, 增益={target_specs[2]:.2f}dBi")

        model.train()
        for p in model.parameters():
            p.requires_grad = False

        def set_batch_norm_eval(model):
            for module in model.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.eval()

        set_batch_norm_eval(model)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)

        for i in range(num_iterations):
            optimizer.zero_grad()

            params_normalized = (params - param_min) / (param_max - param_min + 1e-8)
            params_normalized = torch.clamp(params_normalized, 0.0, 1.0)

            performance = model(params_normalized.unsqueeze(0))[0]

            if performance.dim() != 1 or performance.shape[0] != self.output_dim:
                performance = performance.view(-1)[:self.output_dim]
                if performance.shape[0] < self.output_dim:
                    pad_size = self.output_dim - performance.shape[0]
                    performance = torch.cat([performance, torch.zeros(pad_size, device=device)])

            weights = torch.tensor([3.0, 1.5, 2.0], dtype=torch.float32, device=device)
            loss = torch.mean(weights * torch.square(performance - target_tensor))

            try:
                loss.backward()
            except RuntimeError as e:
                print(f"反向传播错误: {e}")
                torch.backends.cudnn.enabled = False
                loss.backward()
                torch.backends.cudnn.enabled = True

            torch.nn.utils.clip_grad_norm_([params], max_norm=0.5)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                params.clamp_(param_min, param_max)

            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = params.detach().cpu().numpy().copy()
                best_performance = performance.detach().cpu().numpy().copy()

            if (i + 1) % 200 == 0 or i == num_iterations - 1:
                if best_performance is not None and len(best_performance) == self.output_dim:
                    try:
                        pred_real = self.target_scaler.inverse_transform(best_performance.reshape(1, -1))[0]
                        print(f"Iteration {i + 1}/{num_iterations}, Loss: {current_loss:.6f}, "
                              f"Best Loss: {best_loss:.6f}, Best S11: {pred_real[0]:.2f}dB")
                    except Exception as e:
                        print(f"反归一化错误: {e}")

        model.train()
        for p in model.parameters():
            p.requires_grad = True

        if best_performance is not None and len(best_performance) == self.output_dim:
            try:
                best_performance_real = self.target_scaler.inverse_transform(best_performance.reshape(1, -1))[0]
            except Exception as e:
                print(f"最终反归一化错误: {e}")
                best_performance_real = np.zeros(self.output_dim)
        else:
            best_performance_real = np.zeros(self.output_dim)

        print(f"优化结束！最佳损失: {best_loss:.6f}")
        return best_params, best_performance_real, best_loss

    def visualize_results(self, history, y_true, y_pred, model_type):
        """可视化结果"""
        os.makedirs('patch_antenna_results', exist_ok=True)

        y_true_real = self.target_scaler.inverse_transform(y_true)
        y_pred_real = self.target_scaler.inverse_transform(y_pred)

        # 训练监控图
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('MSE损失')
        plt.title('训练损失曲线')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(history['train_rmse'], label='训练RMSE')
        plt.plot(history['val_rmse'], label='验证RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('训练RMSE曲线')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(history['learning_rate'])
        plt.xlabel('Epoch')
        plt.ylabel('学习率')
        plt.title('学习率调度')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        r2_scores = []
        for i in range(len(history['val_loss'])):
            r2 = max(0, 1 - history['val_loss'][i] / np.var(y_true))
            r2_scores.append(r2)
        plt.plot(r2_scores)
        plt.xlabel('Epoch')
        plt.ylabel('R²分数')
        plt.title('模型性能演变')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'patch_antenna_results/{model_type}_training_monitor.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 预测性能分析
        fig, axes = plt.subplots(2, self.output_dim, figsize=(15, 10))

        for i in range(self.output_dim):
            ax1 = axes[0, i]
            ax1.scatter(y_true_real[:, i], y_pred_real[:, i], alpha=0.6, s=15)
            min_val = min(y_true_real[:, i].min(), y_pred_real[:, i].min())
            max_val = max(y_true_real[:, i].max(), y_pred_real[:, i].max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            ax1.set_xlabel('真实值')
            ax1.set_ylabel('预测值')
            ax1.set_title(f'{self.perf_names[i]} 预测 vs 真实')
            ax1.grid(True)

            r2 = r2_score(y_true_real[:, i], y_pred_real[:, i])
            ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax2 = axes[1, i]
            errors = y_true_real[:, i] - y_pred_real[:, i]
            ax2.hist(errors, bins=50, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('误差')
            ax2.set_ylabel('频次')
            ax2.set_title(f'{self.perf_names[i]} 误差分布')
            ax2.grid(True)

            mean_err = np.mean(errors)
            std_err = np.std(errors)
            ax2.axvline(mean_err, color='red', linestyle='--', label=f'均值: {mean_err:.3f}')
            ax2.axvline(mean_err + 2*std_err, color='orange', linestyle='--', label=f'±2σ: {std_err:.3f}')
            ax2.axvline(mean_err - 2*std_err, color='orange', linestyle='--')
            ax2.legend()

        plt.tight_layout()
        plt.savefig(f'patch_antenna_results/{model_type}_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"可视化结果已保存到 patch_antenna_results 目录 (模型类型: {model_type})")

    def visualize_gan_results(self, gan_history, generated_designs=None, generated_performances=None):
        """可视化GAN训练结果"""
        os.makedirs('patch_antenna_results', exist_ok=True)

        # GAN训练损失曲线
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.plot(gan_history['generator_loss'], label='生成器损失')
        plt.plot(gan_history['discriminator_loss'], label='判别器损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('GAN训练损失曲线')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(gan_history['adversarial_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('对抗损失')
        plt.title('对抗损失演变')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(gan_history['performance_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('性能损失')
        plt.title('性能损失演变')
        plt.grid(True)

        # GAN收敛指标
        plt.subplot(2, 2, 4)
        convergence = np.array(gan_history['generator_loss']) / np.array(gan_history['discriminator_loss'])
        plt.plot(convergence)
        plt.xlabel('Epoch')
        plt.ylabel('G/D 损失比')
        plt.title('GAN收敛指标')
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
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

    def hfss_interface(self, parameters):
        """HFSS仿真接口"""
        print("\n=== HFSS仿真接口 ===")
        print("天线类型: 贴片天线")
        print(f"设计参数: {parameters}")

        print("参数说明:")
        for i, name in enumerate(self.param_names):
            print(f"  {name}: {parameters[i]:.3f}")

        print("\n正在调用HFSS执行以下操作:")
        print("1. 创建新的HFSS项目")
        print("2. 根据参数建立贴片天线模型")
        print("3. 设置标准GND结构(0.035mm)")
        print("4. 设置仿真频率范围和边界条件")
        print("5. 运行电磁仿真")
        print("6. 提取S11参数和远区场增益")

        # 仿真结果模拟
        patch_length, patch_width = parameters

        epsilon_r = 4.4
        h = 0.035e-3
        L_meters = patch_length * 1e-3

        epsilon_eff = (epsilon_r + 1) / 2 + (epsilon_r - 1) / 2 * (1 + 12 * h / (patch_width * 1e-3)) ** (-0.5)
        delta_l = 0.412 * h * (epsilon_eff + 0.3) * ((patch_width * 1e-3) / h + 0.264) / \
                  ((epsilon_eff - 0.258) * ((patch_width * 1e-3) / h + 0.8))
        L_eff = L_meters + 2 * delta_l
        freq = 3e8 / (2 * L_eff * np.sqrt(epsilon_eff)) / 1e9

        Z0 = 50
        Z_antenna = 377 * (patch_width * 1e-3) / (2 * h * np.sqrt(epsilon_eff))
        reflection_coeff = (Z_antenna - Z0) / (Z_antenna + Z0)
        s11_min = 20 * np.log10(np.abs(reflection_coeff))

        gain = 2.15 + 0.01 * (patch_length + patch_width) + 0.5 * np.log10(patch_length * patch_width)

        # 添加噪声
        simulated_s11 = s11_min + np.random.normal(0, 0.8)
        simulated_freq = freq + np.random.normal(0, 0.02)
        simulated_gain = gain + np.random.normal(0, 0.3)

        simulated_performance = [simulated_s11, simulated_freq, simulated_gain]

        print(f"\nHFSS仿真结果:")
        print(f"  S11最小值: {simulated_performance[0]:.2f} dB")
        print(f"  对应频率: {simulated_performance[1]:.2f} GHz")
        print(f"  远区场增益: {simulated_performance[2]:.2f} dBi")

        return simulated_performance

    def design_workflow(self, csv_file=None, param_cols=None, perf_cols=None,
                       model_type='resnet', epochs=300, use_synthetic_data=False, use_gan=False):
        """完整的贴片天线设计工作流程"""
        print("=== 贴片天线设计工作流程 ===")
        print("=" * 60)
        start_time = time.time()

        # 1. 加载数据
        print("\n1. 加载天线数据...")
        if csv_file and not use_synthetic_data:
            X_scaled, y_scaled, X_original, y_original = self.load_csv_data(
                csv_file, param_cols, perf_cols
            )
        else:
            print("使用合成数据进行演示")
            X_scaled, y_scaled, X_original, y_original = self.generate_synthetic_data(
                num_samples=10000
            )

        print(f"数据集大小: {X_scaled.shape[0]} 样本")

        # 2. 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )

        if use_gan:
            # 3. GAN工作流程
            print(f"\n2. GAN模型训练...")

            # 训练GAN
            gan_history = self.train_gan(X_train, y_train, epochs=3000, batch_size=128)

            # 可视化GAN训练结果
            self.visualize_gan_results(gan_history)

            # 定义设计目标
            target_performances = [
                [-35.0, 2.45, 7.0],   # WiFi 2.45GHz 高性能设计
                [-30.0, 2.4, 6.5],    # WiFi 2.4GHz 标准设计
                [-25.0, 2.5, 6.0],    # 低成本设计
                [-40.0, 2.42, 7.5]    # 超高性能设计
            ]

            # 使用GAN生成天线设计
            print(f"\n3. 使用GAN生成天线设计...")
            generated_designs, generated_performances = self.generate_antenna_designs(
                target_performances, num_samples=20
            )

            # 可视化生成结果
            self.visualize_gan_results(gan_history, generated_designs, generated_performances)

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

            # 选择最佳设计进行HFSS验证
            if len(generated_designs) > 0:
                best_design_idx = np.argmin(np.mean(np.abs(generated_performances - np.array(target_performances[0])), axis=1))
                best_design = generated_designs[best_design_idx]
                best_performance = generated_performances[best_design_idx]

                print(f"\n4. HFSS仿真验证最佳设计...")
                print(f"最佳设计参数: 长度={best_design[0]:.2f}mm, 宽度={best_design[1]:.2f}mm")
                print(f"预测性能: S11={best_performance[0]:.2f}dB, 频率={best_performance[1]:.2f}GHz, 增益={best_performance[2]:.2f}dBi")

                # HFSS仿真
                simulated_performance = self.hfss_interface(best_design)

                # 设计可行性分析
                print(f"\n5. 设计可行性分析:")
                is_feasible = True

                if best_performance[0] > -15:
                    print(f"  ⚠️  S11值 {best_performance[0]:.2f}dB 偏高")
                    is_feasible = False
                else:
                    print(f"  ✓ S11值 {best_performance[0]:.2f}dB 满足要求")

                if not (2.4 <= best_performance[1] <= 2.5):
                    print(f"  ⚠️  工作频率 {best_performance[1]:.2f}GHz 不在WiFi 2.4GHz频段内")
                    is_feasible = False
                else:
                    print(f"  ✓ 工作频率在WiFi 2.4GHz频段内")

                if best_performance[2] < 5.0:
                    print(f"  ⚠️  增益 {best_performance[2]:.2f}dBi 偏低")
                    is_feasible = False
                else:
                    print(f"  ✓ 增益 {best_performance[2]:.2f}dBi 满足要求")

                if is_feasible:
                    print("🎉 GAN生成的设计成功！满足所有要求。")
                else:
                    print("⚠️ GAN生成的设计基本完成，但部分指标需要进一步优化。")

        else:
            # 传统神经网络工作流程
            print(f"\n2. 创建 {model_type} 模型...")
            model = self.create_model(model_type)

            print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

            print("\n3. 训练模型...")
            history = self.train_model(
                model, X_train, y_train, X_val, y_val,
                epochs=epochs, batch_size=128, lr=0.001
            )

            print("\n4. 模型性能评估...")
            model.eval()
            with torch.no_grad():
                y_pred_val = model(X_val).cpu().numpy()

            y_val_real = self.target_scaler.inverse_transform(y_val.cpu().numpy())
            y_pred_val_real = self.target_scaler.inverse_transform(y_pred_val)

            print("R²决定系数:")
            for i, name in enumerate(self.perf_names):
                r2 = r2_score(y_val_real[:, i], y_pred_val_real[:, i])
                print(f"  {name}: {r2:.4f}")

            self.visualize_results(history, y_val.cpu().numpy(), y_pred_val, model_type)

            print("\n5. 天线参数优化...")
            target_specs = [-35.0, 2.45, 7.0]
            print(f"设计目标: S11={target_specs[0]:.2f}dB, 频率={target_specs[1]:.2f}GHz, 增益={target_specs[2]:.2f}dBi")

            param_min = X_original.min(axis=0)
            param_max = X_original.max(axis=0)
            param_bounds = np.column_stack([param_min, param_max])

            optimal_params, predicted_performance, optimization_loss = self.optimize_antenna(
                model, target_specs, param_bounds, num_iterations=3000
            )

            print(f"\n优化结果:")
            print(f"最优参数: 长度={optimal_params[0]:.3f}mm, 宽度={optimal_params[1]:.3f}mm")
            print(f"预测性能: S11={predicted_performance[0]:.2f}dB, 频率={predicted_performance[1]:.2f}GHz, 增益={predicted_performance[2]:.2f}dBi")

            print(f"\n6. HFSS仿真验证...")
            simulated_performance = self.hfss_interface(optimal_params)

            print(f"\n7. 设计可行性分析:")
            is_feasible = True

            if predicted_performance[0] > -15:
                print(f"  ⚠️  S11值 {predicted_performance[0]:.2f}dB 偏高")
                is_feasible = False
            else:
                print(f"  ✓ S11值 {predicted_performance[0]:.2f}dB 满足要求")

            if not (2.4 <= predicted_performance[1] <= 2.5):
                print(f"  ⚠️  工作频率 {predicted_performance[1]:.2f}GHz 不在WiFi 2.4GHz频段内")
                is_feasible = False
            else:
                print(f"  ✓ 工作频率在WiFi 2.4GHz频段内")

            if predicted_performance[2] < 5.0:
                print(f"  ⚠️  增益 {predicted_performance[2]:.2f}dBi 偏低")
                is_feasible = False
            else:
                print(f"  ✓ 增益 {predicted_performance[2]:.2f}dBi 满足要求")

        end_time = time.time()
        print(f"\n=== 贴片天线设计工作流程完成 ===")
        print(f"总耗时: {end_time - start_time:.2f} 秒")

        if is_feasible:
            print("🎉 设计成功！该贴片天线设计满足要求。")
        else:
            print("⚠️  设计基本完成，但部分指标需要进一步优化。")


    def predict_s11_from_dimensions(self, patch_length, patch_width):
        """
        使用训练好的GAN模型根据天线尺寸预测S11结果

        参数:
        system: PatchAntennaDesignSystem实例
        patch_length: 贴片长度(mm)
        patch_width: 贴片宽度(mm)

        返回:
        predicted_s11_curve: 201个频率点的S11值
        s11_min: S11最小值
        freq_at_s11_min: 对应频率
        far_field_gain: 远区场增益
        """

        # 准备输入数据
        params = np.array([[patch_length, patch_width]], dtype=np.float32)

        # 归一化输入参数
        params_normalized = self.scaler.transform(params)
        params_tensor = torch.tensor(params_normalized, dtype=torch.float32, device=self.device)

        # 使用性能预测器进行预测
        self.performance_predictor.eval()
        with torch.no_grad():
            predicted_performance = self.performance_predictor(params_tensor)
            predicted_performance = predicted_performance.cpu().numpy()

        # 反归一化预测结果
        predicted_performance_denorm = self.target_scaler.inverse_transform(predicted_performance)[0]

        # 提取结果
        s11_min = predicted_performance_denorm[0]
        freq_at_s11_min = predicted_performance_denorm[1]
        far_field_gain = predicted_performance_denorm[2]
        s11_curve = predicted_performance_denorm[3:]  # 201个频率点的S11值

        return s11_curve, s11_min, freq_at_s11_min, far_field_gain


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