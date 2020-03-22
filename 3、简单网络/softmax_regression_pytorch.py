# 导入包
from collections import OrderedDict
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.nn import init
import numpy as np
import sys

# 设置batch_size和载入数据
batch_size = 256

# 下载数据集
mnist_train = torchvision.datasets.FashionMNIST(
    root='../Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(
    root='../Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

# 多进程读取数据
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4

train_iter = torch.utils.data.DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(
    mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 使用线性模块定义softmax回归
num_inputs = 784
num_outputs = 10


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        y = self.linear(x.view(x.shapa[0], -1))  # 将输入的28×28的像素图转换成1×784
        return y


net = LinearNet(num_inputs, num_outputs)

# 定义一个形状转换函数


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


# 定义模型

net = nn.Sequential(
    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)

# 初始化权值
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# 定义loss
loss = nn.CrossEntropyLoss()

# 优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练
num_epochs = 5

# 准确度函数


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

# 训练函数


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' %
              (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, loss, num_epochs,
          batch_size, None, None, optimizer)
