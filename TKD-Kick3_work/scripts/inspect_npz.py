import numpy as np
import os

# 设置要扫描的根目录 (这是根据你的报错日志设置的路径)
TARGET_DIR = r"<PROJECT_ROOT>\data\sequences\val\axe"


def inspect_file(filepath):
    """检查单个 npz 文件的内容"""
    filename = os.path.basename(filepath)

    try:
        # allow_pickle=True 以防数据包含对象数组
        with np.load(filepath, allow_pickle=True) as data:
            keys = list(data.keys())
            print(f"[{filename}] 包含 Keys: {keys}")

            # 遍历检查每个 Key 对应的数据
            for key in data.keys():
                obj = data[key]

                # 如果是 numpy 数组，打印形状和类型
                if isinstance(obj, np.ndarray):
                    # 检查是否包含 NaN (空值)
                    nan_count = 0
                    if np.issubdtype(obj.dtype, np.number):
                        nan_count = np.isnan(obj).sum()

                    info = f"  - Key: '{key}' | Shape: {obj.shape} | Dtype: {obj.dtype}"
                    if nan_count > 0:
                        info += f" | ⚠️ NaN数量: {nan_count}"
                    print(info)
                else:
                    print(f"  - Key: '{key}' | Type: {type(obj)} | Content: {obj}")

            print("-" * 40)

    except Exception as e:
        print(f"❌ 读取失败 [{filename}]: {e}")
        print("-" * 40)


def scan_directory(directory):
    """遍历目录并检查所有 npz 文件"""
    print(f"正在扫描目录: {directory}\n")
    if not os.path.exists(directory):
        print(f"❌ 错误: 目录不存在 - {directory}")
        return

    count = 0
    target_files = ["axe_0172.npz", "axe_0174.npz"]  # 重点关注这两个报错的文件

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".npz"):
                # 只检查报错的文件，或者前几个文件作为对比
                is_target = any(t in file for t in target_files)

                # 如果你想扫描所有文件，去掉下面这个 if 判断即可
                if not is_target and count >= 3:
                    continue

                filepath = os.path.join(root, file)
                inspect_file(filepath)
                count += 1

    if count == 0:
        print("未找到 .npz 文件，请检查路径。")


if __name__ == "__main__":
    scan_directory(TARGET_DIR)