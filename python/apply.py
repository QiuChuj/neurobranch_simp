import sysv_ipc
import ctypes
import numpy as np
import time
from neurobranch_simp import create_simple_model
from loaddata import create_data_loader


# 定义与C结构体完全匹配的ctypes结构
class SharedData(ctypes.Structure):
    _fields_ = [
        ("features", ctypes.c_double * 1000 * 9),
        ("result", ctypes.c_double * 1000),
        ("ready", ctypes.c_int),
        ("n_vars", ctypes.c_uint),
        ("n_clauses", ctypes.c_uint),
    ]


def tensor_to_c_array(tensor, target_size=1000):
    """
    高效地将Tensor转换为c_double数组
    """
    # 转换为numpy并确保数据类型
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

    # 直接通过内存复制（高效方法）
    c_array = (ctypes.c_double * target_size)()
    ctypes.memmove(
        ctypes.addressof(c_array),
        np_array.ctypes.data,
        target_size * ctypes.sizeof(ctypes.c_double),
    )

    return c_array


def python_nn_process():
    try:
        net = create_simple_model()

        # 生成相同的key（与C程序一致）
        key = sysv_ipc.ftok("/tmp/neurobranch_simp", 84)

        # 计算结构体大小
        struct_size = ctypes.sizeof(SharedData)

        try:
            shm_existing = sysv_ipc.SharedMemory(key)
            shm_existing.remove()
        except sysv_ipc.ExistentialError:
            pass

        # 创建/连接共享内存
        shm = sysv_ipc.SharedMemory(key, sysv_ipc.IPC_CREAT, size=struct_size)
        sem = sysv_ipc.Semaphore(key, sysv_ipc.IPC_CREAT)

        # print(f"Python端连接成功，共享内存大小: {struct_size} 字节")

        while True:
            sem.acquire()  # 加锁

            # 读取共享内存数据
            shared_memory_data = shm.read()

            # 将字节数据转换为结构体
            shared_struct = SharedData.from_buffer_copy(shared_memory_data)

            # 检查数据是否就绪
            if shared_struct.ready == 1:
                # print("检测到C端数据就绪!")
                data_loader = create_data_loader()
                # print("dataloader创建完毕")

                # 模拟神经网络处理
                # 这里可以调用你的神经网络模型
                # print("Apply Neurobranch ··· ···")
                nn_result = net.apply(data_loader)
                # print("apply end")

                # 更新结果
                shared_struct.result = tensor_to_c_array(nn_result)
                shared_struct.ready = 2  # 标记Python处理完成

                # print(f"神经网络计算结果: {nn_result}")

                # 将更新后的结构体写回共享内存
                shm.write(ctypes.string_at(ctypes.byref(shared_struct), struct_size))

            sem.release()  # 解锁
            #! 这里会不会导致decision时间变长？
            time.sleep(0.1)  # 避免过度占用CPU

    except KeyboardInterrupt:
        print("\nPython程序正常退出")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    python_nn_process()
