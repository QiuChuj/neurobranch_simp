import numpy as np
import torch
import sysv_ipc
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd

class ApplyDataset(Dataset):
    """从共享内存加载数据的简化数据集类"""
    
    def __init__(self, shm_name="/tmp/nn_shared", size=(1000, 8)):
        self.shm_name = shm_name
        self.size = size
        self.total_elements = size[0] * size[1]
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        """从共享内存读取数据"""
        try:
            # 连接到共享内存
            shm = sysv_ipc.SharedMemory(self.shm_name)
            
            # 读取原始字节数据
            data_bytes = shm.read()
            
            # 转换为numpy数组（假设数据是float32格式）
            data_array = np.frombuffer(data_bytes, dtype=np.float32)
            
            # 重塑为正确形状
            if len(data_array) >= self.total_elements:
                data_array = data_array[:self.total_elements].reshape(self.size)
            else:
                # 数据长度不匹配，用零填充
                padded_data = np.zeros(self.total_elements, dtype=np.float32)
                padded_data[:len(data_array)] = data_array
                data_array = padded_data.reshape(self.size)
            
            # 转换为PyTorch张量
            data_tensor = torch.FloatTensor(data_array)
            
            return data_tensor
            
        except sysv_ipc.ExistentialError:
            print(f"错误: 共享内存 {self.shm_name} 不存在")
            # 返回零张量作为后备
            return torch.zeros(self.size, dtype=torch.float32)
        except Exception as e:
            print(f"读取共享内存时出错: {e}")
            return torch.zeros(self.size, dtype=torch.float32)

def create_data_loader():
    dataset = ApplyDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    return dataloader

class TrainDataset(Dataset):
    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.file_pairs = self._get_file_pairs()
        
    def _get_file_pairs(self):
        """获取匹配的CNF和CSV文件对"""
        data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.cnf')]
        label_files = [f for f in os.listdir(self.label_dir) if f.endswith('.csv')]
        
        file_pairs = []
        for data_file in data_files:
            base_name = os.path.splitext(data_file)[0]
            label_file = f"{base_name}.csv"
            if label_file in label_files:
                file_pairs.append((os.path.join(self.data_dir, data_file),
                                  os.path.join(self.label_dir, label_file)))
        
        return file_pairs
        
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        data_path, label_path = self.file_pairs[idx]
        data = self._parse_data(data_path)
        label = self._parse_labels(label_path)
        return data, label
    
    def _parse_data(self, data_path):
        data_df = pd.read_csv(data_path, header=None)
        data_tensor = torch.tensor(data_df.values, dtype=torch.float32)
        return data_tensor
    
    def _parse_labels(self, label_path):
        label_df = pd.read_csv(label_path, header=None)
        label_tensor = torch.tensor(label_df.values, dtype=torch.float32)
        return label_tensor
    
def create_data_loaders(data_dir, label_dir, batch_size, val_split):
    full_dataset = TrainDataset(data_dir, label_dir)
    val_size = int(val_split*len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader, val_loader
    