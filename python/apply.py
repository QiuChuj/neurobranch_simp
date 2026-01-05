import sysv_ipc
import ctypes
import numpy as np
import time
import json
import argparse
import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader

from neurobranch_simp import create_simple_model

with open("/home/richard/project/neurobranch_simp/configs/config.json") as f:
    config = json.load(f)
SIMP_VAR = config['max_vars']

# 定义与C结构体完全匹配的ctypes结构
class SharedData(ctypes.Structure):
    _fields_ = [
        ("features", ctypes.c_double * SIMP_VAR * 9),
        ("result", ctypes.c_double * SIMP_VAR),
        ("ready", ctypes.c_int),
        ("n_vars", ctypes.c_uint),
        ("n_clauses", ctypes.c_uint),
    ]


class SharedFeaturesDataset(Dataset):
    """
    用于包装一次从共享内存读出的特征张量。
    等价于原来 loaddata.ApplyDataset，只是数据来源变成 SharedData.features。
    """

    def __init__(self, features_tensor: torch.Tensor):
        # features_tensor: shape (SIMP_VAR, 9), float32
        self.features_tensor = features_tensor

    def __len__(self):
        # 仅有一个样本
        return 1

    def __getitem__(self, idx):
        # 返回整块 (SIMP_VAR, 9) 特征
        return self.features_tensor


def tensor_to_c_array(tensor, target_size=SIMP_VAR):
    """
    高效地将Tensor转换为c_double数组（用于写回 result[]）
    """
    np_array = tensor.detach().cpu().numpy().astype(np.float64).flatten()

    # 处理大小不匹配
    if len(np_array) < target_size:
        np_array = np.pad(
            np_array,
            (0, target_size - len(np_array)),
            mode="constant",
            constant_values=0,
        )
    elif len(np_array) > target_size:
        np_array = np_array[:target_size]

    c_array = (ctypes.c_double * target_size)()
    ctypes.memmove(
        ctypes.addressof(c_array),
        np_array.ctypes.data,
        target_size * ctypes.sizeof(ctypes.c_double),
    )

    return c_array


def create_dataloader_from_shared_struct(shared_struct: SharedData) -> DataLoader:
    """
    从 SharedData.features 中构造一个 DataLoader，行为等价于原来的 create_data_loader()：
      - Dataset.__len__ == 1
      - __getitem__ 返回 (SIMP_VAR, 9) 的 tensor
      - DataLoader(batch_size=1) -> 单个 batch，shape 为 (1, SIMP_VAR, 9)
    """
    # 注意：features 是 c_double[SIMP_VAR * 9]，这里直接指定形状 (SIMP_VAR, 9)
    features_np = np.ctypeslib.as_array(shared_struct.features, shape=(SIMP_VAR, 9))

    # 转为 float32，并拷贝一份，避免只读缓冲区问题
    features_np = np.array(features_np, dtype=np.float32, copy=True)

    # 转为 tensor, shape (SIMP_VAR, 9)
    features_tensor = torch.from_numpy(features_np).T  # float32

    dataset = SharedFeaturesDataset(features_tensor)

    # 不再使用多进程 worker，开销更小，而且我们只需要一个 batch
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    return dataloader


def python_nn_process(SHM_FILE):
    try:
        net = create_simple_model()

        # 生成相同的key（与C程序一致）
        key = sysv_ipc.ftok(SHM_FILE, 84)

        # 计算结构体大小
        struct_size = ctypes.sizeof(SharedData)

        # 如果之前有旧的共享内存段，按你原来的逻辑先删掉
        try:
            shm_existing = sysv_ipc.SharedMemory(key)
            shm_existing.remove()
        except sysv_ipc.ExistentialError:
            pass

        # 创建/连接共享内存和信号量
        shm = sysv_ipc.SharedMemory(key, sysv_ipc.IPC_CREAT, size=struct_size)
        sem = sysv_ipc.Semaphore(key, sysv_ipc.IPC_CREAT)

        print(f"Python端连接成功，共享内存大小: {struct_size} 字节")

        while True:
            sem.acquire()  # 加锁

            # 读取共享内存数据（整块 struct）
            shared_memory_data = shm.read()

            # 将字节数据转换为结构体副本
            shared_struct = SharedData.from_buffer_copy(shared_memory_data)

            # 检查数据是否就绪
            if shared_struct.ready == 1:
                print("检测到C端数据就绪!")

                # 直接使用 shared_struct.features 构建 dataloader
                data_loader = create_dataloader_from_shared_struct(shared_struct)

                # 调用神经网络
                print("Apply Neurobranch ··· ···")
                # print("Shape of input features:", next(iter(data_loader)).shape)
                nn_result = net.apply(data_loader)
                print("apply end")

                # 更新结果
                shared_struct.result = tensor_to_c_array(nn_result)
                shared_struct.ready = 2  # 标记Python处理完成

                # 将更新后的结构体写回共享内存
                shm.write(ctypes.string_at(ctypes.byref(shared_struct), struct_size))

            sem.release()  # 解锁

            # 这里的 sleep 是之前就有的，防止空转占用CPU
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nPython程序正常退出")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    print("[apply] 进程启动", flush=True)
    print(f"[apply] argv = {sys.argv}", flush=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID for parallel execusion")
    args = parser.parse_args()
    
    # 根据 ID 生成唯一的文件路径或 Key
    SHM_FILE = f"/tmp/neurobranch_simp_{args.worker_id}"
    print(f"[apply] worker_id = {args.worker_id}", flush=True)
    print(f"[apply] SHM_FILE = {SHM_FILE}", flush=True)
    
    # 确保文件存在
    os.system(f"touch {SHM_FILE}")
    # print("神经网络开始求解")
    python_nn_process(SHM_FILE)
