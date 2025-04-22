import numpy as np
import torch

# 原始Python列表（矩阵）
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

# 打印原始列表
print("原始Python列表:")
print(encoded_sequences)
print("类型:", type(encoded_sequences))
print("形状: 无法直接获取Python列表的形状")
print("-" * 50)

# 转换为NumPy数组
np_array = np.array(encoded_sequences)
print("NumPy数组:")
print(np_array)
print("类型:", type(np_array))
print("形状:", np_array.shape)
print("数据类型:", np_array.dtype)
print("-" * 50)

# 转换为PyTorch张量
model_inputs = torch.tensor(encoded_sequences)
print("PyTorch张量:")
print(model_inputs)
print("类型:", type(model_inputs))
print("形状:", model_inputs.shape)
print("数据类型:", model_inputs.dtype)
print("设备:", model_inputs.device)
print("-" * 50)

# 演示张量操作
print("张量操作示例:")
# 1. 索引操作
print("第一行:", model_inputs[0])
# 2. 数学运算
print("所有元素加1:", model_inputs + 1)
# 3. 维度变换
print("转置:", model_inputs.transpose(0, 1))
# 4. 统计操作
print("平均值:", model_inputs.float().mean())
print("最大值:", model_inputs.max())