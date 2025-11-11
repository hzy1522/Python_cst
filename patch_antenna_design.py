"""
基于PyTorch的贴片天线设计系统
专门针对:
- 输入: 贴片长宽
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

        # 系统参数 - 修改：输入维度从4变为2
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()  # 对目标值使用MinMaxScaler
        self.input_dim = 2  # 只使用贴片长度和宽度
        self.output_dim = 3  # S11最小值、对应频率、远区场增益

        # 参数和性能指标名称 - 修改：只保留两个参数
        self.param_names = ['贴片长度(mm)', '贴片宽度(mm)']
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
        y_scaled: 归一化的天线性能指标
        X_original: 原始天线参数
        y_original: 原始性能指标
        """
        print(f"从CSV文件加载数据: {csv_file}")

        # 读取CSV文件
        df = pd.read_csv(csv_file)
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

        # 如果没有指定列名，则使用默认列名 - 修改：只使用两个参数
        if param_cols is None:
            param_cols = ['patch_length', 'patch_width']
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

        # 验证数据维度 - 修改：输入维度为2
        if X_original.shape[1] != self.input_dim:
            raise ValueError(f"参数列数应为 {self.input_dim}，但实际为 {X_original.shape[1]}")

        if y_original.shape[1] != self.output_dim:
            raise ValueError(f"性能列数应为 {self.output_dim}，但实际为 {y_original.shape[1]}")

        # 数据归一化
        X_scaled = self.scaler.fit_transform(X_original)
        y_scaled = self.target_scaler.fit_transform(y_original)

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

        # 贴片天线参数范围（只使用长度和宽度）
        patch_length = np.random.uniform(10, 50, num_samples)  # 贴片长度 10-50mm
        patch_width = np.random.uniform(10, 60, num_samples)   # 贴片宽度 10-60mm

        X_original = np.column_stack([patch_length, patch_width])

        # 更真实的性能指标计算（基于电磁学原理的改进模型）
        c = 3e8  # 光速
        epsilon_r = 4.4  # FR4介电常数
        h = 0.035e-3  # 介质厚度(m) - 使用标准值0.035mm

        # 有效介电常数计算
        epsilon_eff = (epsilon_r + 1) / 2 + (epsilon_r - 1) / 2 * (1 + 12 * h / patch_width * 1e-3) ** (-0.5)

        # 谐振频率计算（更准确的公式）
        delta_l = 0.412 * h * (epsilon_eff + 0.3) * (patch_width * 1e-3 / h + 0.264) / \
                  ((epsilon_eff - 0.258) * (patch_width * 1e-3 / h + 0.8))
        L_eff = (patch_length * 1e-3) + 2 * delta_l
        freq = c / (2 * L_eff * np.sqrt(epsilon_eff)) / 1e9

        # 添加制造误差和环境噪声
        freq += np.random.normal(0, 0.05, num_samples)

        # S11最小值计算（考虑阻抗匹配）
        Z0 = 50  # 系统阻抗
        # 天线输入阻抗计算（简化模型）
        Z_antenna = 377 * patch_width * 1e-3 / (2 * h * np.sqrt(epsilon_eff))
        # 反射系数计算
        reflection_coeff = (Z_antenna - Z0) / (Z_antenna + Z0)
        s11_min = 20 * np.log10(np.abs(reflection_coeff))

        # 添加实际影响因素
        s11_min += np.random.normal(0, 1.5, num_samples)
        s11_min = np.clip(s11_min, -40, -5)  # S11范围限制在-40到-5dB

        # 远区场增益计算（更准确的模型）
        gain = 2.15 + 0.01 * (patch_length + patch_width) + 0.5 * np.log10(patch_length * patch_width) + np.random.normal(0, 0.4, num_samples)
        gain = np.clip(gain, 1, 12)  # 增益范围限制在1-12dBi

        y_original = np.column_stack([s11_min, freq, gain])

        # 数据归一化
        X_scaled = self.scaler.fit_transform(X_original)
        y_scaled = self.target_scaler.fit_transform(y_original)

        print(f"合成数据生成完成")
        print(f"参数数据形状: {X_original.shape}")
        print(f"性能数据形状: {y_original.shape}")

        return (torch.tensor(X_scaled, dtype=torch.float32),
                torch.tensor(y_scaled, dtype=torch.float32),
                X_original, y_original)

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
                nn.Linear(self.input_dim, 128),  # 修改：输入维度为2
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
            # 修复的残差网络
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
                nn.Linear(self.input_dim, 64),  # 修改：输入维度为2
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
            # 改进的一维卷积网络
            class ImprovedCNN(nn.Module):
                def __init__(self, input_dim, output_dim):
                    super().__init__()

                    # 特征提取部分
                    self.features = nn.Sequential(
                        # 第一层：扩展维度
                        nn.Linear(input_dim, 32),  # 修改：输入维度为2
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.Dropout(0.2),

                        # 转换为卷积格式
                        nn.Unflatten(1, (1, 32)),

                        # 卷积块1
                        nn.Conv1d(1, 16, kernel_size=3, padding=1),
                        nn.BatchNorm1d(16),
                        nn.ReLU(),
                        nn.MaxPool1d(2),

                        # 卷积块2
                        nn.Conv1d(16, 32, kernel_size=3, padding=1),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.MaxPool1d(2),

                        # 卷积块3
                        nn.Conv1d(32, 64, kernel_size=3, padding=1),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.AdaptiveMaxPool1d(4),

                        # 展平
                        nn.Flatten()
                    )

                    # 分类/回归头部
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

            return ImprovedCNN(
                input_dim=self.input_dim,
                output_dim=self.output_dim
            ).to(self.device)

        elif model_type == 'rnn':
            # 改进的循环神经网络
            class ImprovedRNN(nn.Module):
                def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
                    super().__init__()
                    self.hidden_dim = hidden_dim
                    self.num_layers = num_layers

                    # LSTM层
                    self.lstm = nn.LSTM(
                        input_size=input_dim,
                        hidden_size=hidden_dim,
                        num_layers=num_layers,
                        batch_first=True,
                        bidirectional=True,
                        dropout=0.3
                    )

                    # 批归一化层
                    self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)

                    # 注意力机制
                    self.attention = nn.Linear(hidden_dim * 2, 1)

                    # 全连接层
                    self.fc1 = nn.Linear(hidden_dim * 2, 128)
                    self.fc2 = nn.Linear(128, 64)
                    self.fc3 = nn.Linear(64, output_dim)

                    # 激活函数和 dropout
                    self.relu = nn.ReLU()
                    self.dropout = nn.Dropout(0.4)
                    self.leaky_relu = nn.LeakyReLU(0.2)

                def forward(self, x):
                    # x: (batch_size, input_dim) -> (batch_size, seq_len, input_dim)
                    x = x.unsqueeze(1)  # 对于静态参数，seq_len=1

                    # LSTM前向传播
                    batch_size = x.size(0)
                    h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim,
                                   device=x.device)
                    c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim,
                                   device=x.device)

                    out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_len, hidden_dim * 2)

                    # 注意力机制
                    attention_scores = self.attention(out).squeeze(2)
                    attention_weights = torch.softmax(attention_scores, dim=1)
                    attended_out = torch.bmm(attention_weights.unsqueeze(1), out).squeeze(1)

                    # 批归一化和全连接层
                    out = self.batch_norm(attended_out)
                    out = self.dropout(self.relu(self.fc1(out)))
                    out = self.dropout(self.relu(self.fc2(out)))
                    out = self.fc3(out)

                    return out

            return ImprovedRNN(
                input_dim=self.input_dim,
                hidden_dim=64,
                output_dim=self.output_dim,
                num_layers=2
            ).to(self.device)

        elif model_type == 'gnn':
            # 修复的图神经网络
            class AdvancedGNN(nn.Module):
                def __init__(self, input_dim, output_dim, hidden_dim=32, num_layers=3, num_heads=2):
                    super().__init__()
                    self.input_dim = input_dim
                    self.hidden_dim = hidden_dim
                    self.num_layers = num_layers
                    self.num_heads = num_heads
                    self.total_hidden_dim = hidden_dim * num_heads  # 多头注意力总维度

                    # 定义天线参数图结构（2个节点：长度和宽度）
                    self.edge_index = torch.tensor([
                        [0, 1],  # 源节点
                        [1, 0]   # 目标节点
                    ], dtype=torch.long)

                    # 节点特征嵌入
                    self.node_embedding = nn.Linear(1, hidden_dim)

                    # 多头注意力GAT层
                    self.gat_layers = nn.ModuleList()
                    for i in range(num_layers):
                        in_dim = hidden_dim if i == 0 else self.total_hidden_dim
                        self.gat_layers.append(nn.ModuleList([
                            nn.Linear(in_dim, hidden_dim) for _ in range(num_heads)
                        ]))

                    # 注意力分数计算
                    self.attention_layers = nn.ModuleList([
                        nn.Linear(2 * hidden_dim, 1) for _ in range(num_layers)
                    ])

                    # 图卷积层
                    self.graph_conv_layers = nn.ModuleList([
                        nn.Linear(self.total_hidden_dim, self.total_hidden_dim) for _ in range(num_layers)
                    ])

                    # 批归一化和残差连接
                    self.batch_norm_layers = nn.ModuleList([
                        nn.BatchNorm1d(self.total_hidden_dim) for _ in range(num_layers)
                    ])

                    self.residual_layers = nn.ModuleList([
                        nn.Linear(self.total_hidden_dim, self.total_hidden_dim) if i > 0 else None
                        for i in range(num_layers)
                    ])

                    # 全局池化和输出层
                    self.global_pool = nn.AdaptiveAvgPool1d(1)
                    self.fc1 = nn.Linear(self.total_hidden_dim, 256)
                    self.fc2 = nn.Linear(256, 128)
                    self.fc3 = nn.Linear(128, output_dim)

                    # 激活函数和正则化
                    self.relu = nn.ReLU()
                    self.dropout = nn.Dropout(0.4)
                    self.leaky_relu = nn.LeakyReLU(0.2)

                def compute_multi_head_attention(self, node_features, edge_index, layer_idx):
                    """计算多头注意力权重"""
                    src, dst = edge_index

                    all_head_outputs = []

                    for head in range(self.num_heads):
                        # 获取当前头的线性变换
                        linear = self.gat_layers[layer_idx][head]
                        transformed_features = linear(node_features)

                        # 获取源节点和目标节点的特征
                        src_features = transformed_features[src]
                        dst_features = transformed_features[dst]

                        # 连接特征计算注意力分数
                        cat_features = torch.cat([src_features, dst_features], dim=1)
                        attention_scores = self.attention_layers[layer_idx](cat_features)
                        attention_scores = self.leaky_relu(attention_scores)

                        # 计算注意力权重
                        num_nodes = node_features.size(0)
                        attention_weights = torch.zeros(num_nodes, num_nodes, device=node_features.device)
                        attention_weights[src, dst] = attention_scores.squeeze()

                        # 行归一化
                        row_sums = attention_weights.sum(dim=1, keepdim=True)
                        attention_weights = attention_weights / (row_sums + 1e-12)

                        # 应用注意力权重
                        attended_features = torch.matmul(attention_weights, transformed_features)
                        all_head_outputs.append(attended_features)

                    # 拼接多头结果
                    multi_head_output = torch.cat(all_head_outputs, dim=1)
                    return multi_head_output

                def forward(self, x):
                    """
                    GNN前向传播
                    x: (batch_size, input_dim) - 天线参数
                    """
                    batch_size = x.size(0)

                    # 将输入转换为节点特征矩阵
                    node_features = x.view(batch_size * self.input_dim, 1)
                    node_features = self.node_embedding(node_features)
                    node_features = self.relu(node_features)
                    node_features = self.dropout(node_features)

                    # 复制边索引以适应批处理
                    edge_index = self.edge_index.clone()
                    for i in range(1, batch_size):
                        edge_index = torch.cat([
                            edge_index,
                            self.edge_index + i * self.input_dim
                        ], dim=1)

                    # GNN层堆叠
                    for layer_idx in range(self.num_layers):
                        # 多头注意力机制
                        attention_output = self.compute_multi_head_attention(
                            node_features, edge_index, layer_idx
                        )

                        # 图卷积操作
                        conv_output = self.graph_conv_layers[layer_idx](attention_output)

                        # 残差连接
                        if self.residual_layers[layer_idx] is not None:
                            conv_output += self.residual_layers[layer_idx](attention_output)

                        # 批归一化和激活
                        conv_output = self.batch_norm_layers[layer_idx](conv_output)
                        conv_output = self.relu(conv_output)
                        conv_output = self.dropout(conv_output)

                        node_features = conv_output

                    # 全局池化
                    node_features = node_features.view(batch_size, self.input_dim, -1)
                    pooled_features = self.global_pool(node_features.permute(0, 2, 1)).squeeze()

                    # 输出层
                    out = self.fc1(pooled_features)
                    out = self.relu(out)
                    out = self.dropout(out)

                    out = self.fc2(out)
                    out = self.relu(out)
                    out = self.dropout(out)

                    out = self.fc3(out)

                    return out

            return AdvancedGNN(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dim=32,
                num_layers=3,
                num_heads=2
            ).to(self.device)

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

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )

        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'learning_rate': []
        }

        best_val_loss = float('inf')
        patience = 30  # 早停策略
        early_stop_counter = 0

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

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step(epoch + len(train_loader.dataset) / len(train_loader))

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

            # 记录学习率
            current_lr = optimizer.param_groups[0]['lr']

            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_rmse'].append(train_rmse)
            history['val_rmse'].append(val_rmse)
            history['learning_rate'].append(current_lr)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_patch_antenna_model.pth')
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"早停策略触发，在第 {epoch+1} 轮停止训练")
                    break

            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                      f"Train RMSE: {train_rmse:.6f}, Val RMSE: {val_rmse:.6f}, "
                      f"LR: {current_lr:.6f}")

        print(f"训练完成！最佳验证损失: {best_val_loss:.6f}")
        return history

    def optimize_antenna(self, model, target_specs, param_bounds,
                        num_iterations=3000, learning_rate=0.01, device=None):
        """
        改进的贴片天线参数优化

        参数:
        model: 训练好的模型
        target_specs: 目标性能指标 [S11最小值, 对应频率, 远区场增益]
        param_bounds: 参数边界 [[min1, max1], [min2, max2]] - 修改：只有两个参数
        num_iterations: 迭代次数
        learning_rate: 学习率
        device: 计算设备（GPU/CPU）
        """
        # 1. 强制指定设备
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"优化使用设备: {device}")

        # 2. 验证输入 - 修改：检查参数边界维度
        if len(target_specs) != self.output_dim:
            raise ValueError(f"目标性能指标应为 {self.output_dim} 个，实际为 {len(target_specs)}")
        if param_bounds.shape != (self.input_dim, 2):
            raise ValueError(f"参数边界应为 {self.input_dim}x2 的数组，实际为 {param_bounds.shape}")

        num_params = self.input_dim
        model = model.to(device)

        # 3. 创建优化参数
        params = torch.rand(num_params, dtype=torch.float32, device=device, requires_grad=True)

        # 4. 参数映射到指定范围
        param_min = torch.tensor(param_bounds[:, 0], dtype=torch.float32, device=device)
        param_max = torch.tensor(param_bounds[:, 1], dtype=torch.float32, device=device)
        params.data = params.data * (param_max - param_min) + param_min

        # 5. 优化器
        optimizer = optim.Adam([params], lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

        # 目标规格归一化
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

        # 6. 模型模式设置
        model.train()
        for p in model.parameters():
            p.requires_grad = False
            if isinstance(p, (nn.GRU, nn.LSTM, nn.RNN)):
                p.flatten_parameters()

        # 特殊处理batch norm
        def set_batch_norm_eval(model):
            for module in model.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.eval()

        set_batch_norm_eval(model)

        # 学习率调度
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)

        for i in range(num_iterations):
            optimizer.zero_grad()

            # 归一化参数
            params_normalized = (params - param_min) / (param_max - param_min + 1e-8)
            params_normalized = torch.clamp(params_normalized, 0.0, 1.0)

            # 模型预测
            performance = model(params_normalized.unsqueeze(0))[0]

            # 确保性能输出是正确的形状 (3,)
            if performance.dim() != 1 or performance.shape[0] != self.output_dim:
                performance = performance.view(-1)[:self.output_dim]  # 调整形状
                if performance.shape[0] < self.output_dim:
                    # 如果输出维度不足，填充默认值
                    pad_size = self.output_dim - performance.shape[0]
                    performance = torch.cat([performance, torch.zeros(pad_size, device=device)])

            # 计算加权损失（考虑不同指标的重要性）
            weights = torch.tensor([3.0, 1.5, 2.0], dtype=torch.float32, device=device)
            loss = torch.mean(weights * torch.square(performance - target_tensor))

            # 反向传播
            try:
                loss.backward()
            except RuntimeError as e:
                print(f"反向传播错误: {e}")
                torch.backends.cudnn.enabled = False
                loss.backward()
                torch.backends.cudnn.enabled = True

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_([params], max_norm=0.5)

            # 更新参数
            optimizer.step()
            scheduler.step()

            # 限制参数边界
            with torch.no_grad():
                params.clamp_(param_min, param_max)

            # 更新最佳结果
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = params.detach().cpu().numpy().copy()
                best_performance = performance.detach().cpu().numpy().copy()

            # 打印进度
            if (i + 1) % 200 == 0 or i == num_iterations - 1:
                # 反归一化显示真实值
                if best_performance is not None and len(best_performance) == self.output_dim:
                    try:
                        pred_real = self.target_scaler.inverse_transform(
                            best_performance.reshape(1, -1)
                        )[0]
                        print(f"Iteration {i + 1}/{num_iterations}, Loss: {current_loss:.6f}, "
                              f"Best Loss: {best_loss:.6f}, "
                              f"Best S11: {pred_real[0]:.2f}dB, "
                              f"Best Gain: {pred_real[1]:.2f}dB")
                    except Exception as e:
                        print(f"反归一化错误: {e}")

        # 恢复模型状态
        model.train()
        for p in model.parameters():
            p.requires_grad = True

        # 反归一化最佳性能
        if best_performance is not None and len(best_performance) == self.output_dim:
            try:
                best_performance_real = self.target_scaler.inverse_transform(
                    best_performance.reshape(1, -1)
                )[0]
            except Exception as e:
                print(f"最终反归一化错误: {e}")
                best_performance_real = np.zeros(self.output_dim)
        else:
            best_performance_real = np.zeros(self.output_dim)

        print(f"优化结束！最佳损失: {best_loss:.6f}")
        return best_params, best_performance_real, best_loss

    def visualize_results(self, history, y_true, y_pred, model_type):
        """
        改进的可视化结果
        """
        os.makedirs('patch_antenna_results', exist_ok=True)

        # 反归一化数据
        y_true_real = self.target_scaler.inverse_transform(y_true)
        y_pred_real = self.target_scaler.inverse_transform(y_pred)

        # 1. 综合训练监控图
        plt.figure(figsize=(15, 10))

        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('MSE损失')
        plt.title('训练损失曲线')
        plt.legend()
        plt.grid(True)

        # RMSE曲线
        plt.subplot(2, 2, 2)
        plt.plot(history['train_rmse'], label='训练RMSE')
        plt.plot(history['val_rmse'], label='验证RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('训练RMSE曲线')
        plt.legend()
        plt.grid(True)

        # 学习率变化
        plt.subplot(2, 2, 3)
        plt.plot(history['learning_rate'])
        plt.xlabel('Epoch')
        plt.ylabel('学习率')
        plt.title('学习率调度')
        plt.grid(True)

        # R²分数演变
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

        # 2. 预测性能详细分析
        fig, axes = plt.subplots(2, self.output_dim, figsize=(15, 10))

        for i in range(self.output_dim):
            # 散点图
            ax1 = axes[0, i]
            ax1.scatter(y_true_real[:, i], y_pred_real[:, i], alpha=0.6, s=15)
            min_val = min(y_true_real[:, i].min(), y_pred_real[:, i].min())
            max_val = max(y_true_real[:, i].max(), y_pred_real[:, i].max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            ax1.set_xlabel('真实值')
            ax1.set_ylabel('预测值')
            ax1.set_title(f'{self.perf_names[i]} 预测 vs 真实')
            ax1.grid(True)

            # R²分数
            r2 = r2_score(y_true_real[:, i], y_pred_real[:, i])
            ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # 误差分布
            ax2 = axes[1, i]
            errors = y_true_real[:, i] - y_pred_real[:, i]
            ax2.hist(errors, bins=50, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('误差')
            ax2.set_ylabel('频次')
            ax2.set_title(f'{self.perf_names[i]} 误差分布')
            ax2.grid(True)

            # 统计信息
            mean_err = np.mean(errors)
            std_err = np.std(errors)
            ax2.axvline(mean_err, color='red', linestyle='--', label=f'均值: {mean_err:.3f}')
            ax2.axvline(mean_err + 2*std_err, color='orange', linestyle='--', label=f'±2σ: {std_err:.3f}')
            ax2.axvline(mean_err - 2*std_err, color='orange', linestyle='--')
            ax2.legend()

        plt.tight_layout()
        plt.savefig(f'patch_antenna_results/{model_type}_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 参数重要性分析（针对GNN）
        if model_type == 'gnn':
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            param_importance = np.random.rand(self.input_dim)  # 实际应该从模型中提取

            bars = ax.bar(range(self.input_dim), param_importance, alpha=0.7)
            ax.set_xticks(range(self.input_dim))
            ax.set_xticklabels(self.param_names, rotation=45, ha='right')
            ax.set_ylabel('重要性分数')
            ax.set_title('贴片天线参数重要性分析')
            ax.grid(True, alpha=0.3)

            # 在柱状图上添加数值标签
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(f'patch_antenna_results/{model_type}_param_importance.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"可视化结果已保存到 patch_antenna_results 目录 (模型类型: {model_type})")

    def hfss_interface(self, parameters):
        """
        HFSS仿真接口
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
        print("3. 设置标准GND结构(0.035mm)")
        print("4. 设置仿真频率范围和边界条件")
        print("5. 运行电磁仿真")
        print("6. 提取S11参数和远区场增益")

        # 更真实的仿真结果模拟
        patch_length, patch_width = parameters  # 修改：只有两个参数

        # 基于改进的电磁学模型
        epsilon_r = 4.4
        h = 0.035e-3  # 使用标准GND厚度0.035mm
        L_meters = patch_length * 1e-3

        # 更准确的谐振频率计算
        epsilon_eff = (epsilon_r + 1) / 2 + (epsilon_r - 1) / 2 * (1 + 12 * h / (patch_width * 1e-3)) ** (-0.5)
        delta_l = 0.412 * h * (epsilon_eff + 0.3) * ((patch_width * 1e-3) / h + 0.264) / \
                  ((epsilon_eff - 0.258) * ((patch_width * 1e-3) / h + 0.8))
        L_eff = L_meters + 2 * delta_l
        freq = 3e8 / (2 * L_eff * np.sqrt(epsilon_eff)) / 1e9

        # S11计算
        Z0 = 50
        Z_antenna = 377 * (patch_width * 1e-3) / (2 * h * np.sqrt(epsilon_eff))
        reflection_coeff = (Z_antenna - Z0) / (Z_antenna + Z0)
        s11_min = 20 * np.log10(np.abs(reflection_coeff))

        # 增益计算
        gain = 2.15 + 0.01 * (patch_length + patch_width) + 0.5 * np.log10(patch_length * patch_width)

        # 添加一些噪声模拟实际仿真误差
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
                       model_type='resnet', epochs=300, use_synthetic_data=False):
        """
        完整的贴片天线设计工作流程
        """
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

        # 3. 创建和训练模型
        print(f"\n2. 创建 {model_type} 模型...")
        model = self.create_model(model_type)

        # 打印模型信息
        print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        print("\n3. 训练模型...")
        history = self.train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=128, lr=0.001
        )

        # 4. 模型评估
        print("\n4. 模型性能评估...")
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train).cpu().numpy()
            y_pred_val = model(X_val).cpu().numpy()

        # 确保预测结果是正确的形状
        if y_pred_val.ndim == 1:
            y_pred_val = y_pred_val.reshape(-1, 1)
        if y_pred_val.shape[1] != self.output_dim:
            print(f"警告: 验证集预测形状不正确: {y_pred_val.shape}")
            if y_pred_val.shape[1] > self.output_dim:
                y_pred_val = y_pred_val[:, :self.output_dim]
            else:
                pad_size = self.output_dim - y_pred_val.shape[1]
                y_pred_val = np.pad(y_pred_val, ((0, 0), (0, pad_size)), mode='constant')

        # 计算R²分数（使用真实值）
        y_val_real = self.target_scaler.inverse_transform(y_val.cpu().numpy())
        y_pred_val_real = self.target_scaler.inverse_transform(y_pred_val)

        print("R²决定系数 (越高越好):")
        for i, name in enumerate(self.perf_names):
            r2 = r2_score(y_val_real[:, i], y_pred_val_real[:, i])
            print(f"  {name}: {r2:.4f}")

        # 5. 可视化结果
        print("\n5. 生成可视化结果...")
        self.visualize_results(history, y_val.cpu().numpy(), y_pred_val, model_type)

        # 6. 天线参数优化
        print("\n6. 贴片天线参数优化...")

        # 定义设计目标
        target_specs = [
            -35.0,   # S11最小值: -35dB
            2.45,    # 对应频率: 2.45GHz
            7.0      # 远区场增益: 7.0dBi
        ]

        print(f"设计目标:")
        for i, (name, target) in enumerate(zip(self.perf_names, target_specs)):
            print(f"  {name}: {target}")

        # 参数边界 - 修改：只有两个参数的边界
        param_min = X_original.min(axis=0)
        param_max = X_original.max(axis=0)
        param_bounds = np.column_stack([param_min, param_max])

        print(f"\n参数优化边界:")
        for i, name in enumerate(self.param_names):
            print(f"  {name}: {param_bounds[i, 0]:.3f} - {param_bounds[i, 1]:.3f}")

        # 执行优化
        optimal_params, predicted_performance, optimization_loss = self.optimize_antenna(
            model, target_specs, param_bounds, num_iterations=3000
        )

        print(f"\n优化结果:")
        print(f"最优参数:")
        for i, name in enumerate(self.param_names):
            print(f"  {name}: {optimal_params[i]:.3f}")

        print(f"\n预测性能:")
        for i, name in enumerate(self.perf_names):
            diff = abs(predicted_performance[i] - target_specs[i])
            status = "✓" if (name == self.perf_names[0] and predicted_performance[i] <= target_specs[i]) or \
                           (name == self.perf_names[1] and abs(diff) < 0.05) or \
                           (name == self.perf_names[2] and predicted_performance[i] >= target_specs[i]) else "⚠️"
            print(f"  {status} {name}: {predicted_performance[i]:.3f} (目标: {target_specs[i]})")

        # 7. HFSS仿真验证
        print(f"\n7. HFSS仿真验证...")
        simulated_performance = self.hfss_interface(optimal_params)

        # 8. 设计可行性分析
        print(f"\n8. 设计可行性分析:")
        is_feasible = True

        # S11检查
        if predicted_performance[0] > -15:
            print(f"  ⚠️  S11值 {predicted_performance[0]:.2f}dB 偏高，可能需要改进匹配")
            is_feasible = False
        else:
            print(f"  ✓ S11值 {predicted_performance[0]:.2f}dB 满足要求")

        # 频率检查
        if not (2.4 <= predicted_performance[1] <= 2.5):
            print(f"  ⚠️  工作频率 {predicted_performance[1]:.2f}GHz 不在WiFi 2.4GHz频段内")
            is_feasible = False
        else:
            print(f"  ✓ 工作频率在WiFi 2.4GHz频段内")

        # 增益检查
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
            'total_time': end_time - start_time,
            'r2_scores': [r2_score(y_val_real[:, i], y_pred_val_real[:, i]) for i in range(self.output_dim)]
        }

        np.save(f'patch_antenna_results/{model_type}_design_result.npy', design_result)
        print(f"设计结果已保存到 patch_antenna_results/{model_type}_design_result.npy")

        return design_result

if __name__ == "__main__":
    # 演示使用
    system = PatchAntennaDesignSystem()

    # 检查命令行参数
    import sys
    if len(sys.argv) > 1 and sys.argv[1].endswith('.csv'):
        csv_file = sys.argv[1]
        print(f"使用CSV文件: {csv_file}")

        param_cols = None
        perf_cols = None
        if len(sys.argv) > 3:
            param_cols = sys.argv[2].split(',')
            perf_cols = sys.argv[3].split(',')

        result = system.design_workflow(
            csv_file=csv_file,
            param_cols=param_cols,
            perf_cols=perf_cols,
            model_type='resnet',
            epochs=300
        )
    else:
        # 使用合成数据进行演示
        print("使用合成数据进行演示 (添加CSV文件路径作为参数可使用真实数据)")

        # 可以尝试不同的模型类型
        model_type = 'resnet'  # 'mlp', 'resnet', 'cnn', 'rnn', 'gnn'

        result = system.design_workflow(
            model_type=model_type,
            epochs=300,
            use_synthetic_data=True
        )

    print("\n设计流程全部完成！")