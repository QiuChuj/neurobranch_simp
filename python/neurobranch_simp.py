import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
from torch.nn import MSELoss

class SimpleSATNet(nn.Module):
    """
    简化的SAT启发式分数预测网络
    输入: [batch_size, 1000, 8] 或 [1000, 8]
    输出: [batch_size, 1000, 1] 或 [1000, 1]
    """
    
    def __init__(self, params):
        super(SimpleSATNet, self).__init__()
        input_dim = params['input_dim']
        hidden_dims = params['hidden_dims']
        output_dim = params['output_dim']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # 轻微dropout防止过拟合
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.train_mode = params["train_mode"]
        self.model_path = params["model_path"]
        # 将模型移到指定设备
        if not self.train_mode:
            # 加载模型并移到指定设备
            self.load()
            self.eval()
        elif not self.new_model:
            self.load()
            self.train()
        else:
            self.train()
        self.to(self.device)
        
    def forward(self, x):
        """
        前向传播
        x: 形状为 [batch, 1000, 8] 或 [1000, 8]
        返回: 形状为 [batch, 1000, 1] 或 [1000, 1]
        """
        original_shape = x.shape
        
        # 如果输入是3D [batch, vars, features]，重塑为2D用于处理
        if len(original_shape) == 3:
            batch_size, num_vars, num_features = original_shape
            x = x.view(-1, num_features).detach().to(self.device)  # [batch*1000, 8]
        else:
            batch_size = 1
            num_vars = original_shape[0]
            x = x.view(-1, original_shape[1]).detach().to(self.device)  # [1000, 8]
        
        # 通过网络处理
        output = self.network(x)  # [batch*1000, 1] 或 [1000, 1]
        
        # 恢复原始形状
        if len(original_shape) == 3:
            output = output.view(batch_size, num_vars, -1)  # [batch, 1000, 1]
        else:
            output = output.view(num_vars, -1)  # [1000, 1]
            
        return output
    
    def save(self):
        torch.save(self.state_dict(), self.model_path)
        
    def load(self):
        self.load_state_dict(torch.load(self.model_path))
        
    def train_epoch(self, train_loader, val_loader, optimizer, epochs):
        criterion = MSELoss()  # 均方误差损失函数
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # 训练阶段
            epoch_train_loss = 0.0
            start_time = time.time()
            i = 0
            
            for args_batch, labels_batch in train_loader:
                i += 1
                print("Training: file ", i)
                
                # 前向传播
                # print("Forward:")
                optimizer.zero_grad()
                output = self.forward(args_batch)
                
                # 计算损失
                # print("Loss computation:")
                loss = criterion(output, labels_batch)
                
                # 反向传播
                # print("Loss backward:")
                loss.backward()
                #! 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                # print("Loss backward complete.")
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            # 计算平均训练损失
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            epoch_val_loss = 0.0
            with torch.no_grad():
                for args_batch, labels_batch in val_loader:
                    print("Evaluating:")
                    # 解包批次数据
                    
                    output = self.forward(args_batch)
                    loss = criterion(output, labels_batch)
                    epoch_val_loss += loss.item()
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # 打印训练信息
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{epochs} | '
                f'Train Loss: {avg_train_loss:.6f} | '
                f'Val Loss: {avg_val_loss:.6f} | '
                f'Time: {epoch_time:.2f}s')
        
        # 保存训练好的模型
        self.save()
        print(f'Model saved to {self.model_path}')
        
        return train_losses, val_losses
    
    def apply(self, data_loader):
        for args_batch in data_loader:
            output = self.forward(args_batch)
        return output

class VariableWiseNet(nn.Module):
    """
    更简洁的版本：对每个变量独立应用相同的MLP
    参数更少，计算更高效
    """
    
    def __init__(self, params):
        super(VariableWiseNet, self).__init__()
        input_dim = params['input_dim']
        hidden_dim = params['hidden_dim']
        output_dim = params['output_dim']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 共享的变量处理网络
        self.var_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.train_mode = params["train_mode"]
        self.model_path = params["model_path"]
        # 将模型移到指定设备
        if not self.train_mode:
            # 加载模型并移到指定设备
            self.load()
            self.eval()
        elif not self.new_model:
            self.load()
            self.train()
        else:
            self.train()
        self.to(self.device)
        
    def forward(self, x):
        """
        前向传播 - 对每个变量独立应用网络
        x: [batch, 1000, 8] 或 [1000, 8]
        返回: [batch, 1000, 1] 或 [1000, 1]
        """
        original_shape = x.shape
        
        if len(original_shape) == 3:
            batch_size, num_vars, num_features = original_shape
            # 对每个变量应用相同的处理
            outputs = []
            for i in range(num_vars):
                var_features = x[:, i, :].detach().to(self.device)  # [batch, 8]
                var_output = self.var_processor(var_features)  # [batch, 1]
                outputs.append(var_output)
            
            # 组合结果 [batch, 1000, 1]
            return torch.stack(outputs, dim=1)
            
        else:
            # 单样本情况 [1000, 8]
            outputs = []
            for i in range(original_shape[0]):
                var_features = x[i:i+1, :].detach().to(self.device)  # [1, 8]
                var_output = self.var_processor(var_features)  # [1, 1]
                outputs.append(var_output)
            
            return torch.cat(outputs, dim=0)  # [1000, 1]
        
    def save(self):
        torch.save(self.state_dict(), self.model_path)
        
    def load(self):
        self.load_state_dict(torch.load(self.model_path))
        
    def train_epoch(self, train_loader, val_loader, optimizer, epochs):
        criterion = MSELoss()  # 均方误差损失函数
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # 训练阶段
            epoch_train_loss = 0.0
            start_time = time.time()
            i = 0
            
            for args_batch, labels_batch in train_loader:
                i += 1
                print("Training: file ", i)
                
                # 前向传播
                # print("Forward:")
                optimizer.zero_grad()
                output = self.forward(args_batch)
                
                # 计算损失
                # print("Loss computation:")
                loss = criterion(output, labels_batch)
                
                # 反向传播
                # print("Loss backward:")
                loss.backward()
                #! 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                # print("Loss backward complete.")
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            # 计算平均训练损失
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            epoch_val_loss = 0.0
            with torch.no_grad():
                for args_batch, labels_batch in val_loader:
                    print("Evaluating:")
                    # 解包批次数据
                    
                    output = self.forward(args_batch)
                    loss = criterion(output, labels_batch)
                    epoch_val_loss += loss.item()
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # 打印训练信息
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{epochs} | '
                f'Train Loss: {avg_train_loss:.6f} | '
                f'Val Loss: {avg_val_loss:.6f} | '
                f'Time: {epoch_time:.2f}s')
        
        # 保存训练好的模型
        self.save()
        print(f'Model saved to {self.model_path}')
        
        return train_losses, val_losses
    
    def apply(self, data_loader):
        for args_batch in data_loader:
            output = self.forward(args_batch)
        return output

def create_simple_model():
    """创建简化模型"""
    # 模型配置参数
    with open("/home/richard/project/neurobranch_simp/configs/params.json", 'r') as f:
        params = json.load(f)

    use_shared_weights = params['use_shared_weights']
    if use_shared_weights:
        return VariableWiseNet(params)
    else:
        return SimpleSATNet(params)
