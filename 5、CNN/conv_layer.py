#二维卷积层
import torch
from torch import nn

#卷积运算的实现
#输入的参数为两个二维Tensor
def corr2d(X, K):
    h, w = K.shape
    #定义输出的尺寸
    Y = torch.zeros((X.shape[0] -h + 1, X.shape[1] -w +1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            #卷积计算
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

'''
#卷积测试
X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
print(corr2d(X, K))
'''

#卷积层定义
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        #根据卷积核的尺寸初始化卷积核权重
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

#边缘检测
#构造6×8的图像，中间4列为黑，其余为白
X = torch.ones(6, 8)
X[:, 2:6] = 0
#print(X)

'''
tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.]])

'''
#构建1×2的卷积核，横向相邻元素相同，输出0,不同输出1
K = torch.tensor([[1, -1]])
#print(corr2d(X, K))
Y = corr2d(X, K)

'''
tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])
'''

#学习卷积核参数
conv2d = Conv2D(kernel_size=(1, 2))

step = 20
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()
    l.backward()

    #梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    #梯度清零
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i + 1) % 5 == 0:
        print('Step %d, loss %.3f' % (i + 1, l.item()))
'''
Step 5, loss 1.388
Step 10, loss 0.200
Step 15, loss 0.035
Step 20, loss 0.007
'''

print('weight:', conv2d.weight.data)
print('bias:', conv2d.bias.data)
'''
weight: tensor([[ 1.0425, -1.0428]])
bias: tensor([0.0002])
'''
