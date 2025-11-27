import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # 可用 GPU 数量
    print("device_count: {}".format(torch.cuda.device_count()))

    # 当前是几号 GPU
    print("current_device: ", torch.cuda.current_device())

    # 查看设备算力，device 不指定查看所有
    print(torch.cuda.get_device_capability(device=None))

    # 设备名称
    print(torch.cuda.get_device_name())

    # 当前 cuda 是否可用
    print(torch.cuda.is_available())

    # cuda 架构信息
    print(torch.cuda.get_arch_list())

    # 0号 GPU 属性
    print(torch.cuda.get_device_properties(0))

    # 显存信息
    print(torch.cuda.mem_get_info(device=None))

    # GPU 详细信息
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # 清GPU 缓存
    print(torch.cuda.empty_cache())