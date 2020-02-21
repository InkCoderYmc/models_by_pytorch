#pytorch实现线性回归
import torch
import numpy as np

#生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] +true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

#使用data包读取数据
import torch.utils.data as Data

batch_size = 10
dataset = Data.TensorDataset(features, labels) #将特征和标签组合
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True) #随机读取batch_size大小的小批量数据

#定义模型
class LinearNet(torch.nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = torch.nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net)

#使用nn.Sequential搭建网络
#写法1
net = torch.nn.Sequential(torch.nn.Linear(num_inputs, 1))
print(net)
print(net[0])

#写法2
net = torch.nn.Sequential()
net.add_module('linear', torch.nn.Linear(num_inputs, 1))
print(net)
print(net[0])

#写法3
from _collections import OrderedDict
net = torch.nn.Sequential(OrderedDict([('linear', torch.nn.Linear(num_inputs, 1))]))
print(net)
print(net[0])

#查看模型参数
for param in net.parameters():
    print(param)

#初始化
from torch.nn import init

init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)

#损失函数
loss = torch.nn.MSELoss()

#优化算法
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)


#调整学习率
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1
print(optimizer)

#训练模型
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() #梯度清零
        l.backward()
        optimizer.step()
    print('eopch %d, loss: %f' % (epoch, l.item()))

dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
