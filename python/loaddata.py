import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):
    def __init__(self, data_dir, label_dir,
                 max_vars=10000, num_features=9):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.max_vars = max_vars
        self.num_features = num_features
        self.file_pairs = self._get_file_pairs()

    def _get_file_pairs(self):
        """获取匹配的 feature.bin 和 score.bin 文件对"""
        data_files = [f for f in os.listdir(self.data_dir)
                      if f.endswith(".bin")]
        label_files = [f for f in os.listdir(self.label_dir)
                       if f.endswith(".bin")]

        file_pairs = []
        for data_file in data_files:
            base_name = os.path.splitext(data_file)[0]  # decision123
            label_file = f"{base_name}.bin"
            if label_file in label_files:
                file_pairs.append(
                    (
                        os.path.join(self.data_dir, data_file),
                        os.path.join(self.label_dir, label_file),
                    )
                )

        file_pairs.sort()  # 可选：按名字排序
        return file_pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        data_path, label_path = self.file_pairs[idx]
        data = self._parse_data(data_path)
        label = self._parse_labels(label_path)
        return data, label

    def _parse_data(self, data_path):
        """
        读取 features.bin：
        - 现在 C 端可能写入的是 <= max_vars * num_features 个 double
        - 若不足 expected_size，则在尾部补 0，补成 (max_vars, num_features)
        """
        expected_size = self.max_vars * self.num_features  # 10000 * 9 = 90000

        arr = np.fromfile(data_path, dtype=np.float64)
        actual_size = arr.size

        if actual_size > expected_size:
            # 数据比预期还大，说明写入格式有问题，仍然报错
            raise ValueError(
                f"数据文件大小超出预期: {data_path}, "
                f"读取到 {actual_size} 个 double, 期望至多 {expected_size}"
            )
        elif actual_size < expected_size:
            # 不足时在尾部补 0
            padded = np.zeros(expected_size, dtype=np.float64)
            padded[:actual_size] = arr
            arr = padded

        # 现在 arr 一定是 expected_size 长度
        arr = arr.reshape(self.max_vars, self.num_features)  # (max_vars, num_features)

        # 转为 torch.float32
        data_tensor = torch.from_numpy(arr.astype(np.float32))
        return data_tensor  # shape: (max_vars, num_features)

    def _parse_labels(self, label_path):
        """
        读取 scores.bin：
        - 现在 C 端可能写入的是 <= max_vars 个 double
        - 若不足 expected_size，则在尾部补 0，补成 (max_vars, 1)
        """
        expected_size = self.max_vars  # 10000

        arr = np.fromfile(label_path, dtype=np.float64)
        actual_size = arr.size

        if actual_size > expected_size:
            raise ValueError(
                f"标签文件大小超出预期: {label_path}, "
                f"读取到 {actual_size} 个 double, 期望至多 {expected_size}"
            )
        elif actual_size < expected_size:
            padded = np.zeros(expected_size, dtype=np.float64)
            padded[:actual_size] = arr
            arr = padded

        # (max_vars,) -> (max_vars, 1)
        arr = arr.reshape(self.max_vars, 1)

        label_tensor = torch.from_numpy(arr.astype(np.float32))
        return label_tensor  # shape: (max_vars, 1)


def create_data_loaders(data_dir, label_dir, batch_size, val_split,
                        max_vars=10000, num_features=9):
    dataset = TrainDataset(data_dir, label_dir,
                           max_vars=max_vars,
                           num_features=num_features)

    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4
    )

    return train_loader, val_loader