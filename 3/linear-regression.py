#%matplotlib inline
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

#构造数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

features = torch.randn(num_examples, num_inputs, dtype=torch.float32) #随机生成一个1000×2的tensor

labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b #根据参数生成labels

labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32) #加上均值为0,标准茶为0.01的正态分布

def use_svg_display():
    display.set_matplotlib_formats('svg') #定义为用矢量图显示

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
#plt.show()

#每次返回batch_size个随机样本的特征和标签
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) #得到一个乱序的list
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) #考虑最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)


batch_size = 10

#初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

#定义模型，mm函数计算矩阵乘法
def linreg(X, w, b):
    return torch.mm(X, w) + b

#定义损失函数MSE
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

#小批量随机梯度下降算法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs): #迭代周期
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward() #小批量的loss对模型参数求梯度
        sgd([w, b], lr, batch_size) #使用小批量随机梯度下降爱嗯迭代模型参数

        w.grad.data.zero_()
        b.grad.data.zero_()

    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w, w)
print(true_b, b) #学习到的参数与实际参数之间相差无几