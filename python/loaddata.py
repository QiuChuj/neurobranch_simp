import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd


class TrainDataset(Dataset):
    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.file_pairs = self._get_file_pairs()

    def _get_file_pairs(self):
        """获取匹配的CNF和CSV文件对"""
        data_files = [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]
        label_files = [f for f in os.listdir(self.label_dir) if f.endswith(".csv")]

        file_pairs = []
        for data_file in data_files:
            base_name = os.path.splitext(data_file)[0]
            label_file = f"{base_name}.csv"
            if label_file in label_files:
                file_pairs.append(
                    (
                        os.path.join(self.data_dir, data_file),
                        os.path.join(self.label_dir, label_file),
                    )
                )

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
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader
