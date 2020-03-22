import torch
from torch import nn
import sys


# 多输入通道
def corr2d(X, K):
    h, w = K.shape
    # 定义输出的尺寸
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 卷积计算
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

# 每个通道计算再相加


def corr2d_multi_in(X, K):
    res = corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += corr2d(X[i, :, :], K[i, :, :])
    return res


X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

#print(corr2d_multi_in(X, K))
'''
tensor([[ 56.,  72.],
        [104., 120.]])
'''

# 多输出通道
#输入通道数和输出通道数为c_i, c_o
#高和宽为k_h, k_w
# 为每个输出通道分别创建c_i×k_h×k_w的核数组，连结为c_o×c_i×k_h×k_w


def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K])


K = torch.stack([K, K + 1, K + 2])
#print(corr2d_multi_in_out(X, K))
'''
tensor([[[ 56.,  72.],
         [104., 120.]],

        [[ 76., 100.],
         [148., 172.]],

        [[ 96., 128.],
         [192., 224.]]])
'''

# 1×1卷积


def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h * w)
    K = K.view(c_o, c_i)
    Y = torch.mm(K, X)
    return Y.view(c_o, h, w)


X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

#print((Y1 - Y2).norm().item() < 1e-6)
