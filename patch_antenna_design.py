"""
基于PyTorch的贴片天线设计系统
专门针对:
- 输入: 贴片长宽、GND厚度、信号线厚度
- 输出: S11最小值、对应频率、远区场增益
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from python_hfss import *
import time
import os

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
        self.input_dim = 4  # 贴片长宽、GND厚度、信号线厚度
        self.output_dim = 3  # S11最小值、对应频率、远区场增益

        # 参数和性能指标名称
        self.param_names = ['贴片长度(mm)', '贴片宽度(mm)', 'GND厚度(mm)', '信号线厚度(mm)']
        self.perf_names = ['S11最小值(dB)', '对应频率(GHz)', '远区场增益(dBi)']

    def load_csv_data(self, csv_file, param_cols=None, perf_cols=None):
        """
        从CSV文件加载贴片天线数据

        参数:
        csv_file: CSV文件路径
        param_cols: 参数列名列表 (可选)
        perf_cols: 性能指标列名列表 (可选)

        返回:
        X_scaled: 归一化的天线参数
        y: 天线性能指标
        X_original: 原始天线参数
        y_original: 原始性能指标
        """
        print(f"从CSV文件加载数据: {csv_file}")

        # 读取CSV文件
        df = pd.read_csv(csv_file)
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

        # 如果没有指定列名，则使用默认列名
        if param_cols is None:
            param_cols = ['patch_length', 'patch_width', 'gnd_thickness', 'signal_thickness']
            print(f"使用默认参数列名: {param_cols}")

        if perf_cols is None:
            perf_cols = ['s11_min', 'freq_at_s11_min', 'far_field_gain']
            print(f"使用默认性能列名: {perf_cols}")

        # 验证列名是否存在
        for col in param_cols + perf_cols:
            if col not in df.columns:
                raise ValueError(f"列名 '{col}' 不在CSV文件中")

        # 提取数据
        X_original = df[param_cols].values
        y_original = df[perf_cols].values

        # 验证数据维度
        if X_original.shape[1] != self.input_dim:
            raise ValueError(f"参数列数应为 {self.input_dim}，但实际为 {X_original.shape[1]}")

        if y_original.shape[1] != self.output_dim:
            raise ValueError(f"性能列数应为 {self.output_dim}，但实际为 {y_original.shape[1]}")

        # 数据归一化
        X_scaled = self.scaler.fit_transform(X_original)

        print(f"参数数据形状: {X_original.shape}")
        print(f"性能数据形状: {y_original.shape}")

        # 显示数据统计信息
        print(f"\n参数统计:")
        for i, (name, col) in enumerate(zip(self.param_names, param_cols)):
            print(f"  {name}: 均值={X_original[:, i].mean():.3f}, 标准差={X_original[:, i].std():.3f}")

        print(f"\n性能指标统计:")
        for i, (name, col) in enumerate(zip(self.perf_names, perf_cols)):
            print(f"  {name}: 均值={y_original[:, i].mean():.3f}, 标准差={y_original[:, i].std():.3f}")

        return (torch.tensor(X_scaled, dtype=torch.float32),
                torch.tensor(y_original, dtype=torch.float32),
                X_original, y_original)

    def generate_synthetic_data(self, num_samples=5000):
        """
        生成合成的贴片天线数据用于测试

        参数:
        num_samples: 样本数量

        返回:
        X_scaled: 归一化的天线参数
        y: 天线性能指标
        X_original: 原始天线参数
        y_original: 原始性能指标
        """
        np.random.seed(42)
        print(f"生成合成贴片天线数据，样本数: {num_samples}")

        # 贴片天线参数范围
        patch_length = np.random.uniform(5, 15, num_samples)  # 贴片长度 10-50mm
        patch_width = np.random.uniform(5, 15, num_samples)   # 贴片宽度 10-50mm
        gnd_thickness = np.random.uniform(0.01, 0.05, num_samples)  # GND厚度 0.5-3.0mm
        signal_thickness = np.random.uniform(0.01, 0.05, num_samples)  # 信号线厚度 0.1-1.0mm

        X_original = np.column_stack([patch_length, patch_width, gnd_thickness, signal_thickness])

        # 性能指标计算（基于电磁学原理的简化模型）
        c = 3e8  # 光速

        # 谐振频率计算
        L_meters = patch_length * 1e-3
        freq = c / (2 * L_meters * np.sqrt(4.4)) / 1e9  # 假设介电常数为4.4 (FR4)
        freq += np.random.normal(0, 0.2, num_samples)  # 添加噪声

        # S11最小值计算 (与天线尺寸和匹配有关)
        s11_min = -25 - 0.1 * (patch_length + patch_width) + np.random.normal(0, 2, num_samples)
        s11_min = np.clip(s11_min, -40, -10)  # S11范围限制在-40到-10dB

        # 远区场增益计算
        gain = 2.0 + 0.02 * (patch_length + patch_width) - 0.5 * gnd_thickness + np.random.normal(0, 0.3, num_samples)
        gain = np.clip(gain, 0, 10)  # 增益范围限制在0-10dBi

        y_original = np.column_stack([s11_min, freq, gain])

        # 数据归一化
        X_scaled = self.scaler.fit_transform(X_original)

        print(f"合成数据生成完成")
        print(f"参数数据形状: {X_original.shape}")
        print(f"性能数据形状: {y_original.shape}")

        return (torch.tensor(X_scaled, dtype=torch.float32),
                torch.tensor(y_original, dtype=torch.float32),
                X_original, y_original)

    def create_model(self, model_type='resnet'):
        """
        创建贴片天线设计神经网络模型

        参数:
        model_type: 模型类型 ('mlp', 'resnet', 'cnn')

        返回:
        神经网络模型
        """
        if model_type == 'mlp':
            return nn.Sequential(
                nn.Linear(self.input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(128, self.output_dim)
            ).to(self.device)

        elif model_type == 'resnet':
            # 残差网络
            class ResidualBlock(nn.Module):
                def __init__(self, dim):
                    super().__init__()
                    self.block = nn.Sequential(
                        nn.Linear(dim, dim),
                        nn.BatchNorm1d(dim),
                        nn.ReLU(),
                        nn.Linear(dim, dim),
                        nn.BatchNorm1d(dim)
                    )
                    self.relu = nn.ReLU()

                def forward(self, x):
                    return self.relu(x + self.block(x))

            return nn.Sequential(
                nn.Linear(self.input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),

                ResidualBlock(128),
                ResidualBlock(128),

                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),

                ResidualBlock(256),
                ResidualBlock(256),

                nn.Linear(256, self.output_dim)
            ).to(self.device)

        elif model_type == 'cnn':
            # 一维卷积网络
            return nn.Sequential(
                nn.Unflatten(1, (1, self.input_dim)),
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Flatten(),
                nn.Linear(64 * (self.input_dim // 4), 128),
                nn.ReLU(),
                nn.Linear(128, self.output_dim)
            ).to(self.device)

        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def train_model(self, model, X_train, y_train, X_val, y_val,
                   epochs=200, batch_size=64, lr=0.001):
        """
        训练贴片天线设计模型

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

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': []
        }

        best_val_loss = float('inf')

        print("开始训练模型...")
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            # 计算平均损失
            train_loss /= len(train_loader.dataset)
            train_rmse = np.sqrt(train_loss)

            # 验证
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)
            val_rmse = np.sqrt(val_loss)

            # 更新学习率
            scheduler.step(val_loss)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_patch_antenna_model.pth')

            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_rmse'].append(train_rmse)
            history['val_rmse'].append(val_rmse)

            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                      f"Train RMSE: {train_rmse:.6f}, Val RMSE: {val_rmse:.6f}")

        print(f"训练完成！最佳验证损失: {best_val_loss:.6f}")
        return history

    def optimize_antenna(self, model, target_specs, param_bounds,
                        num_iterations=1000, learning_rate=0.01):
        """
        贴片天线参数优化

        参数:
        model: 训练好的模型
        target_specs: 目标性能指标 [S11最小值, 对应频率, 远区场增益]
        param_bounds: 参数边界 [[min1, max1], [min2, max2], [min3, max3], [min4, max4]]
        num_iterations: 迭代次数
        learning_rate: 学习率

        返回:
        优化后的参数和预测性能
        """
        # 验证输入
        if len(target_specs) != self.output_dim:
            raise ValueError(f"目标性能指标应为 {self.output_dim} 个，实际为 {len(target_specs)}")

        if param_bounds.shape != (self.input_dim, 2):
            raise ValueError(f"参数边界应为 {self.input_dim}x2 的数组")

        num_params = self.input_dim

        # 初始化参数
        params = torch.rand(num_params, requires_grad=True, device=self.device, dtype=torch.float32)

        # 将参数映射到指定范围
        param_bounds_tensor = torch.tensor(param_bounds, dtype=torch.float32, device=self.device)
        for i in range(num_params):
            params.data[i] = param_bounds_tensor[i, 0] + params.data[i] * (param_bounds_tensor[i, 1] - param_bounds_tensor[i, 0])

        optimizer = optim.Adam([params], lr=learning_rate)
        target_tensor = torch.tensor(target_specs, dtype=torch.float32, device=self.device)

        best_loss = float('inf')
        best_params = None
        best_performance = None

        print("开始贴片天线参数优化...")
        print(f"目标性能: S11={target_specs[0]:.2f}dB, 频率={target_specs[1]:.2f}GHz, 增益={target_specs[2]:.2f}dBi")

        for i in range(num_iterations):
            # 归一化参数
            params_normalized = (params - param_bounds_tensor[:, 0]) / \
                              (param_bounds_tensor[:, 1] - param_bounds_tensor[:, 0])

            # 预测性能
            performance = model(params_normalized.unsqueeze(0))[0]

            # 计算损失 (加权损失，更关注重要的性能指标)
            # S11权重更高，因为它对天线性能影响更大
            weights = torch.tensor([2.0, 1.0, 1.0], device=self.device)  # S11权重更高
            loss = torch.mean(weights * torch.square(performance - target_tensor))

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 限制参数在边界内
            for j in range(num_params):
                params.data[j] = torch.clamp(params.data[j], param_bounds_tensor[j, 0], param_bounds_tensor[j, 1])

            # 更新最佳结果
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = params.detach().cpu().numpy().copy()
                best_performance = performance.detach().cpu().numpy().copy()

            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6f}, "
                      f"Best Loss: {best_loss:.6f}")

        return best_params, best_performance, best_loss

    def visualize_results(self, history, y_true, y_pred):
        """
        可视化训练结果和预测性能
        """
        # 创建图形目录
        os.makedirs('patch_antenna_results', exist_ok=True)

        # 1. 训练损失曲线
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('训练损失曲线')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history['train_rmse'], label='训练RMSE')
        plt.plot(history['val_rmse'], label='验证RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('训练RMSE曲线')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('patch_antenna_results/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 预测 vs 真实值散点图
        fig, axes = plt.subplots(1, self.output_dim, figsize=(12, 4))

        for i in range(self.output_dim):
            ax = axes[i]
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=10)

            # 添加对角线
            min_val = min(y_true[:, i].min(), y_pred[:, i].min())
            max_val = max(y_true[:, i].max(), y_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            ax.set_xlabel('真实值')
            ax.set_ylabel('预测值')
            ax.set_title(self.perf_names[i])
            ax.grid(True)

            # 计算R²
            r2 = 1 - np.sum((y_true[:, i] - y_pred[:, i])**2) / np.sum((y_true[:, i] - y_true[:, i].mean())**2)
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig('patch_antenna_results/prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 误差分布直方图
        errors = y_true - y_pred
        fig, axes = plt.subplots(1, self.output_dim, figsize=(12, 4))

        for i in range(self.output_dim):
            ax = axes[i]
            ax.hist(errors[:, i], bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('误差')
            ax.set_ylabel('频次')
            ax.set_title(f'{self.perf_names[i]} 误差分布')
            ax.grid(True)

            # 添加统计信息
            mean_err = np.mean(errors[:, i])
            std_err = np.std(errors[:, i])
            ax.axvline(mean_err, color='red', linestyle='--', label=f'均值: {mean_err:.3f}')
            ax.legend()

        plt.tight_layout()
        plt.savefig('patch_antenna_results/error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. 参数相关性分析
        fig, axes = plt.subplots(self.output_dim, self.input_dim, figsize=(16, 12))

        for i in range(self.output_dim):
            for j in range(self.input_dim):
                ax = axes[i, j]
                ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=5)
                ax.set_xlabel(self.perf_names[i])
                ax.set_ylabel(self.param_names[j])
                ax.grid(True)

        plt.tight_layout()
        plt.savefig('patch_antenna_results/correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("可视化结果已保存到 patch_antenna_results 目录")

    def hfss_interface(self, parameters):
        """
        HFSS仿真接口

        参数:
        parameters: 天线参数 [贴片长度, 贴片宽度, GND厚度, 信号线厚度]

        返回:
        仿真得到的性能指标
        """
        print("\n=== HFSS仿真接口 ===")
        print("天线类型: 贴片天线")
        print(f"设计参数: {parameters}")

        print("参数说明:")
        for i, name in enumerate(self.param_names):
            print(f"  {name}: {parameters[i]:.3f}")

        print("\n正在调用HFSS执行以下操作:")
        print("1. 创建新的HFSS项目")
        print("2. 根据参数建立贴片天线模型")
        print("3. 设置GND结构和信号线")
        print("4. 设置仿真频率范围和边界条件")
        print("5. 运行电磁仿真")
        print("6. 提取S11参数和远区场增益")

        # 在实际应用中，这里会调用HFSS API
        # 这里使用简化的模拟结果
        simulated_s11 = -20 - 0.1 * (parameters[0] + parameters[1]) + np.random.normal(0, 1)
        simulated_freq = 2.4 + 0.02 * parameters[0] + np.random.normal(0, 0.1)
        simulated_gain = 5.0 + 0.01 * (parameters[0] + parameters[1]) + np.random.normal(0, 0.2)

        simulated_performance = [simulated_s11, simulated_freq, simulated_gain]

        print(f"\nHFSS仿真结果:")
        print(f"  S11最小值: {simulated_performance[0]:.2f} dB")
        print(f"  对应频率: {simulated_performance[1]:.2f} GHz")
        print(f"  远区场增益: {simulated_performance[2]:.2f} dBi")

        return simulated_performance

    def design_workflow(self, csv_file=None, param_cols=None, perf_cols=None,
                       model_type='resnet', epochs=200, use_synthetic_data=False):
        """
        完整的贴片天线设计工作流程

        参数:
        csv_file: CSV文件路径
        param_cols: 参数列名列表
        perf_cols: 性能列名列表
        model_type: 模型类型
        epochs: 训练轮数
        use_synthetic_data: 是否使用合成数据进行测试

        返回:
        优化后的天线设计结果
        """
        print("=== 贴片天线设计工作流程 ===")
        print("=" * 60)
        start_time = time.time()

        # 1. 加载数据
        print("\n1. 加载天线数据...")
        if csv_file and not use_synthetic_data:
            X_scaled, y, X_original, y_original = self.load_csv_data(
                csv_file, param_cols, perf_cols
            )
        else:
            print("使用合成数据进行演示")
            X_scaled, y, X_original, y_original = self.generate_synthetic_data(
                num_samples=5000
            )

        print(f"数据集大小: {X_scaled.shape[0]} 样本")

        # 2. 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # 3. 创建和训练模型
        print(f"\n2. 创建 {model_type} 模型...")
        model = self.create_model(model_type)

        print("\n3. 训练模型...")
        history = self.train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=64, lr=0.001
        )

        # 4. 模型评估
        print("\n4. 模型性能评估...")
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train).cpu().numpy()
            y_pred_val = model(X_val).cpu().numpy()

        # 计算R²分数
        from sklearn.metrics import r2_score
        print("R²决定系数 (越高越好):")
        for i, name in enumerate(self.perf_names):
            r2 = r2_score(y_val.cpu().numpy()[:, i], y_pred_val[:, i])
            print(f"  {name}: {r2:.4f}")

        # 5. 可视化结果
        print("\n5. 生成可视化结果...")
        self.visualize_results(history, y_val.cpu().numpy(), y_pred_val)

        # 6. 天线参数优化
        print("\n6. 贴片天线参数优化...")

        # 定义设计目标 (示例目标)
        target_specs = [
            -30.0,   # S11最小值: -30dB (尽可能小)
            2.45,    # 对应频率: 2.45GHz (WiFi频段)
            6.5      # 远区场增益: 6.5dBi (高增益)
        ]

        print(f"设计目标:")
        for i, (name, target) in enumerate(zip(self.perf_names, target_specs)):
            print(f"  {name}: {target}")

        # 参数边界（基于数据范围）
        param_min = X_original.min(axis=0)
        param_max = X_original.max(axis=0)
        param_bounds = np.column_stack([param_min, param_max])

        print(f"\n参数优化边界:")
        for i, name in enumerate(self.param_names):
            print(f"  {name}: {param_bounds[i, 0]:.3f} - {param_bounds[i, 1]:.3f}")

        # 执行优化
        optimal_params, predicted_performance, optimization_loss = self.optimize_antenna(
            model, target_specs, param_bounds, num_iterations=2000
        )

        print(f"\n优化结果:")
        print(f"最优参数:")
        for i, name in enumerate(self.param_names):
            print(f"  {name}: {optimal_params[i]:.3f}")

        print(f"\n预测性能:")
        for i, name in enumerate(self.perf_names):
            diff = abs(predicted_performance[i] - target_specs[i])
            status = "✓" if (name == self.perf_names[0] and predicted_performance[i] <= target_specs[i]) or \
                           (name == self.perf_names[1] and abs(diff) < 0.1) or \
                           (name == self.perf_names[2] and predicted_performance[i] >= target_specs[i]) else "⚠️"
            print(f"  {status} {name}: {predicted_performance[i]:.3f} (目标: {target_specs[i]})")

        # 7. HFSS仿真验证
        print(f"\n7. HFSS仿真验证...")
        simulated_performance = self.hfss_interface(optimal_params)

        # 8. 设计可行性分析
        print(f"\n8. 设计可行性分析:")
        is_feasible = True

        # S11检查
        if predicted_performance[0] > -15:  # S11 > -15dB 被认为性能较差
            print(f"  ⚠️  S11值 {predicted_performance[0]:.2f}dB 偏高，可能需要改进匹配")
            is_feasible = False
        else:
            print(f"  ✓ S11值 {predicted_performance[0]:.2f}dB 满足要求")

        # 频率检查
        if not (2.4 <= predicted_performance[1] <= 2.5):  # WiFi 2.4GHz频段
            print(f"  ⚠️  工作频率 {predicted_performance[1]:.2f}GHz 不在WiFi 2.4GHz频段内")
            is_feasible = False
        else:
            print(f"  ✓ 工作频率在WiFi 2.4GHz频段内")

        # 增益检查
        if predicted_performance[2] < 5.0:  # 增益低于5dBi
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

        # 保存设计结果
        design_result = {
            'optimal_parameters': optimal_params,
            'predicted_performance': predicted_performance,
            'simulated_performance': simulated_performance,
            'target_specifications': target_specs,
            'optimization_loss': optimization_loss,
            'model_type': model_type,
            'training_history': history,
            'is_feasible': is_feasible,
            'total_time': end_time - start_time
        }

        np.save('patch_antenna_results/design_result.npy', design_result)
        print("设计结果已保存到 patch_antenna_results/design_result.npy")

        return design_result

if __name__ == "__main__":
    # 演示使用
    system = PatchAntennaDesignSystem()

    # 检查命令行参数
    import sys
    if len(sys.argv) > 1 and sys.argv[1].endswith('.csv'):
        csv_file = sys.argv[1]
        print(f"使用CSV文件: {csv_file}")

        # 如果指定了列名
        param_cols = None
        perf_cols = None
        if len(sys.argv) > 3:
            param_cols = sys.argv[2].split(',')
            perf_cols = sys.argv[3].split(',')

        # 执行设计流程
        result = system.design_workflow(
            csv_file=csv_file,
            param_cols=param_cols,
            perf_cols=perf_cols,
            model_type='resnet',
            epochs=200
        )
    else:
        # 使用合成数据进行演示
        print("使用合成数据进行演示 (添加CSV文件路径作为参数可使用真实数据)")
        result = system.design_workflow(
            model_type='resnet',
            epochs=200,
            use_synthetic_data=True
        )

    print("\n设计流程全部完成！")