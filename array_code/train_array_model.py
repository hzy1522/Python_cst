# dual_cgan_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 正向生成器（阵列 -> 方向图）
class ForwardGenerator(nn.Module):
    def __init__(self, noise_dim=100, condition_dim=225, output_dim=360):
        super(ForwardGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        
        self.input_layer = nn.Linear(noise_dim + condition_dim, 512)
        self.hidden1 = nn.Linear(512, 1024)
        self.hidden2 = nn.Linear(1024, 2048)
        self.hidden3 = nn.Linear(2048, 1024)
        self.output_layer = nn.Linear(1024, output_dim)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=1)
        x = self.relu(self.input_layer(x))
        x = self.dropout(x)
        x = self.relu(self.hidden1(x))
        x = self.dropout(x)
        x = self.relu(self.hidden2(x))
        x = self.dropout(x)
        x = self.relu(self.hidden3(x))
        x = self.tanh(self.output_layer(x))
        return x

# 正向判别器
class ForwardDiscriminator(nn.Module):
    def __init__(self, input_dim=360, condition_dim=225):
        super(ForwardDiscriminator, self).__init__()
        self.input_layer = nn.Linear(input_dim + condition_dim, 1024)
        self.hidden1 = nn.Linear(1024, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.output_layer = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, direction_pattern, condition):
        x = torch.cat([direction_pattern, condition], dim=1)
        x = self.relu(self.input_layer(x))
        x = self.dropout(x)
        x = self.relu(self.hidden1(x))
        x = self.dropout(x)
        x = self.relu(self.hidden2(x))
        x = self.sigmoid(self.output_layer(x))
        return x

# 反向生成器（方向图 -> 阵列）
class InverseGenerator(nn.Module):
    def __init__(self, noise_dim=100, condition_dim=360, output_dim=225):
        super(InverseGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        
        self.input_layer = nn.Linear(noise_dim + condition_dim, 512)
        self.hidden1 = nn.Linear(512, 1024)
        self.hidden2 = nn.Linear(1024, 2048)
        self.hidden3 = nn.Linear(2048, 1024)
        self.output_layer = nn.Linear(1024, output_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=1)
        x = self.relu(self.input_layer(x))
        x = self.dropout(x)
        x = self.relu(self.hidden1(x))
        x = self.dropout(x)
        x = self.relu(self.hidden2(x))
        x = self.dropout(x)
        x = self.relu(self.hidden3(x))
        x = self.sigmoid(self.output_layer(x))
        return x

# 反向判别器
class InverseDiscriminator(nn.Module):
    def __init__(self, input_dim=225, condition_dim=360):
        super(InverseDiscriminator, self).__init__()
        self.input_layer = nn.Linear(input_dim + condition_dim, 1024)
        self.hidden1 = nn.Linear(1024, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.output_layer = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, array_data, condition):
        x = torch.cat([array_data, condition], dim=1)
        x = self.relu(self.input_layer(x))
        x = self.dropout(x)
        x = self.relu(self.hidden1(x))
        x = self.dropout(x)
        x = self.relu(self.hidden2(x))
        x = self.sigmoid(self.output_layer(x))
        return x

# 数据集类
class AntennaDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        array_data = self.data.iloc[idx, :225].values.astype(np.float32)
        pattern_data = self.data.iloc[idx, 225:].values.astype(np.float32)
        
        return {
            'array': torch.tensor(array_data),
            'pattern': torch.tensor(pattern_data)
        }

# 双向CGAN模型
class DualCGAN:
    def __init__(self, noise_dim=100, array_dim=225, pattern_dim=360, lr=0.0002, beta1=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_dim = noise_dim
        self.array_dim = array_dim
        self.pattern_dim = pattern_dim
        
        # 初始化正向模型
        self.forward_generator = ForwardGenerator(noise_dim, array_dim, pattern_dim).to(self.device)
        self.forward_discriminator = ForwardDiscriminator(pattern_dim, array_dim).to(self.device)
        
        # 初始化反向模型
        self.inverse_generator = InverseGenerator(noise_dim, pattern_dim, array_dim).to(self.device)
        self.inverse_discriminator = InverseDiscriminator(array_dim, pattern_dim).to(self.device)
        
        # 损失函数
        self.criterion = nn.BCELoss()
        
        # 优化器
        self.optimizer_fg = optim.Adam(self.forward_generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_fd = optim.Adam(self.forward_discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_ig = optim.Adam(self.inverse_generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_id = optim.Adam(self.inverse_discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    def train_forward(self, dataloader, epochs=100):
        """训练正向模型"""
        G_losses = []
        D_losses = []
        
        for epoch in range(epochs):
            for i, data in enumerate(dataloader):
                real_patterns = data['pattern'].to(self.device)
                conditions = data['array'].to(self.device)
                batch_size = real_patterns.size(0)
                
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # 训练正向判别器
                self.forward_discriminator.zero_grad()
                
                outputs = self.forward_discriminator(real_patterns, conditions)
                d_loss_real = self.criterion(outputs, real_labels)
                d_loss_real.backward()
                
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                fake_patterns = self.forward_generator(noise, conditions)
                outputs = self.forward_discriminator(fake_patterns.detach(), conditions)
                d_loss_fake = self.criterion(outputs, fake_labels)
                d_loss_fake.backward()
                
                d_loss = d_loss_real + d_loss_fake
                self.optimizer_fd.step()
                
                # 训练正向生成器
                self.forward_generator.zero_grad()
                
                outputs = self.forward_discriminator(fake_patterns, conditions)
                g_loss = self.criterion(outputs, real_labels)
                g_loss.backward()
                self.optimizer_fg.step()
                
                G_losses.append(g_loss.item())
                D_losses.append(d_loss.item())
                
                if i % 100 == 0:
                    print(f'[Forward] Epoch [{epoch}/{epochs}], Step [{i}/{len(dataloader)}], '
                          f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')
            
            if epoch % 10 == 0:
                self.save_forward_model(f'forward_model_epoch_{epoch}.pth')
                
        self.save_forward_model('forward_final_model.pth')
        return G_losses, D_losses
    
    def train_inverse(self, dataloader, epochs=100):
        """训练反向模型"""
        G_losses = []
        D_losses = []
        
        for epoch in range(epochs):
            for i, data in enumerate(dataloader):
                real_arrays = data['array'].to(self.device)
                conditions = data['pattern'].to(self.device)
                batch_size = real_arrays.size(0)
                
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # 训练反向判别器
                self.inverse_discriminator.zero_grad()
                
                outputs = self.inverse_discriminator(real_arrays, conditions)
                d_loss_real = self.criterion(outputs, real_labels)
                d_loss_real.backward()
                
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                fake_arrays = self.inverse_generator(noise, conditions)
                outputs = self.inverse_discriminator(fake_arrays.detach(), conditions)
                d_loss_fake = self.criterion(outputs, fake_labels)
                d_loss_fake.backward()
                
                d_loss = d_loss_real + d_loss_fake
                self.optimizer_id.step()
                
                # 训练反向生成器
                self.inverse_generator.zero_grad()
                
                outputs = self.inverse_discriminator(fake_arrays, conditions)
                g_loss = self.criterion(outputs, real_labels)
                g_loss.backward()
                self.optimizer_ig.step()
                
                G_losses.append(g_loss.item())
                D_losses.append(d_loss.item())
                
                if i % 100 == 0:
                    print(f'[Inverse] Epoch [{epoch}/{epochs}], Step [{i}/{len(dataloader)}], '
                          f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')
            
            if epoch % 10 == 0:
                self.save_inverse_model(f'inverse_model_epoch_{epoch}.pth')
                
        self.save_inverse_model('inverse_final_model.pth')
        return G_losses, D_losses
    
    def train_both(self, dataloader, epochs=100):
        """同时训练两个模型"""
        print("开始训练正向模型...")
        fg_losses, fd_losses = self.train_forward(dataloader, epochs)
        
        print("开始训练反向模型...")
        ig_losses, id_losses = self.train_inverse(dataloader, epochs)
        
        return {
            'forward_g_losses': fg_losses,
            'forward_d_losses': fd_losses,
            'inverse_g_losses': ig_losses,
            'inverse_d_losses': id_losses
        }
    
    def generate_pattern(self, array_condition):
        """给定阵列生成方向图"""
        self.forward_generator.eval()
        with torch.no_grad():
            condition = torch.tensor(array_condition, dtype=torch.float32).unsqueeze(0).to(self.device)
            noise = torch.randn(1, self.noise_dim).to(self.device)
            generated_pattern = self.forward_generator(noise, condition)
        self.forward_generator.train()
        return generated_pattern.cpu().numpy()[0]
    
    def generate_array(self, pattern_condition, threshold=0.5):
        """给定方向图生成阵列"""
        self.inverse_generator.eval()
        with torch.no_grad():
            condition = torch.tensor(pattern_condition, dtype=torch.float32).unsqueeze(0).to(self.device)
            noise = torch.randn(1, self.noise_dim).to(self.device)
            generated_array = self.inverse_generator(noise, condition)
        self.inverse_generator.train()
        array_result = generated_array.cpu().numpy()[0]
        # 二值化处理
        binary_array = (array_result >= threshold).astype(int)
        return array_result, binary_array
    
    def save_forward_model(self, path):
        """保存正向模型"""
        torch.save({
            'generator_state_dict': self.forward_generator.state_dict(),
            'discriminator_state_dict': self.forward_discriminator.state_dict(),
            'optimizer_fg_state_dict': self.optimizer_fg.state_dict(),
            'optimizer_fd_state_dict': self.optimizer_fd.state_dict(),
        }, path)
    
    def save_inverse_model(self, path):
        """保存反向模型"""
        torch.save({
            'generator_state_dict': self.inverse_generator.state_dict(),
            'discriminator_state_dict': self.inverse_discriminator.state_dict(),
            'optimizer_ig_state_dict': self.optimizer_ig.state_dict(),
            'optimizer_id_state_dict': self.optimizer_id.state_dict(),
        }, path)
    
    def load_forward_model(self, path):
        """加载正向模型"""
        checkpoint = torch.load(path)
        self.forward_generator.load_state_dict(checkpoint['generator_state_dict'])
        self.forward_discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_fg.load_state_dict(checkpoint['optimizer_fg_state_dict'])
        self.optimizer_fd.load_state_dict(checkpoint['optimizer_fd_state_dict'])
    
    def load_inverse_model(self, path):
        """加载反向模型"""
        checkpoint = torch.load(path)
        self.inverse_generator.load_state_dict(checkpoint['generator_state_dict'])
        self.inverse_discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_ig.load_state_dict(checkpoint['optimizer_ig_state_dict'])
        self.optimizer_id.load_state_dict(checkpoint['optimizer_id_state_dict'])

# 训练脚本
def train_models():
    # 创建数据集和数据加载器
    dataset = AntennaDataset('antenna_data.csv')  # 替换为你的CSV文件路径
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 初始化双模型
    dual_cgan = DualCGAN(noise_dim=100, array_dim=225, pattern_dim=360)
    
    # 训练两个模型
    losses = dual_cgan.train_both(dataloader, epochs=100)
    
    # 绘制损失曲线
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(losses['forward_g_losses'], label='Forward Generator Loss')
    axes[0, 0].set_title('Forward Generator Loss')
    axes[0, 0].set_xlabel('Iterations')
    axes[0, 0].set_ylabel('Loss')
    
    axes[0, 1].plot(losses['forward_d_losses'], label='Forward Discriminator Loss')
    axes[0, 1].set_title('Forward Discriminator Loss')
    axes[0, 1].set_xlabel('Iterations')
    axes[0, 1].set_ylabel('Loss')
    
    axes[1, 0].plot(losses['inverse_g_losses'], label='Inverse Generator Loss')
    axes[1, 0].set_title('Inverse Generator Loss')
    axes[1, 0].set_xlabel('Iterations')
    axes[1, 0].set_ylabel('Loss')
    
    axes[1, 1].plot(losses['inverse_d_losses'], label='Inverse Discriminator Loss')
    axes[1, 1].set_title('Inverse Discriminator Loss')
    axes[1, 1].set_xlabel('Iterations')
    axes[1, 1].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('dual_model_training_losses.png')
    plt.show()

# 推理脚本
def inference_example():
    # 加载训练好的模型
    dual_cgan = DualCGAN()
    dual_cgan.load_forward_model('forward_final_model.pth')
    dual_cgan.load_inverse_model('inverse_final_model.pth')
    
    # 示例1: 阵列 -> 方向图
    print("=== 正向推理示例 ===")
    test_array = np.random.randint(0, 2, size=(15, 15)).astype(np.float32)
    predicted_pattern = dual_cgan.generate_pattern(test_array.flatten())
    print(f"输入阵列形状: {test_array.shape}")
    print(f"预测方向图形状: {predicted_pattern.shape}")
    print(f"预测方向图前10个值: {predicted_pattern[:10]}")
    
    # 示例2: 方向图 -> 阵列
    print("\n=== 反向推理示例 ===")
    test_pattern = np.random.uniform(-1, 1, size=360).astype(np.float32)
    predicted_array_prob, predicted_array_binary = dual_cgan.generate_array(test_pattern)
    print(f"输入方向图形状: {test_pattern.shape}")
    print(f"预测阵列概率形状: {predicted_array_prob.shape}")
    print(f"预测二值化阵列形状: {predicted_array_binary.shape}")
    print("预测二值化阵列 (前5行):")
    print(predicted_array_binary[:5])

if __name__ == "__main__":
    # 训练模型
    train_models()
    
    # 或者进行推理测试
    # inference_example()
