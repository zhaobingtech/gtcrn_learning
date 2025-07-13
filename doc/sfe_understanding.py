import torch
from gtcrn import SFE

# 创建一个随机输入张量
# x = torch.randn(1, 3, 10, 12)
# 输入张量形状为 (batch_size, channels, height, width)
batch_size =1
channels = 2
T= 3
F = 4
num = batch_size * channels * T * F
x = torch.arange(0, num).float().view(batch_size, channels, T, F)  # 添加一个维度
# 初始化 SFE 模块
sfe = SFE(kernel_size=3, stride=1)

# 前向传播
xs = sfe(x)

print("Input shape:", x.shape)
print("Output shape:", xs.shape)

print("x=:",x)
print("xs:", xs)


print("end")
# Input shape: torch.Size([1, 3, 100, 129])
# Output shape: torch.Size([1, 9, 100, 129])
