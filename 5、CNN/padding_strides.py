import torch
from torch import nn

# 定义一个函数计算卷积层


def comp_conv2d(conv2d, X):
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])


conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

X = torch.rand(8, 8)
#print(comp_conv2d(conv2d, X).shape)
'''
torch.Size([8, 8])
'''

# 可以设置高和宽上不同填充数，使得输出和输出具有相同的高度
conv2d = nn.Conv2d(in_channels=1, out_channels=1,
                   kernel_size=(5, 3), padding=(2, 1))
#print(comp_conv2d(conv2d, X).shape)
'''
torch.Size([8, 8])
'''

# 步幅
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
#print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
#print(comp_conv2d(conv2d, X).shape)
