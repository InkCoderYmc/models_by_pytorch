import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))

print(net)
X = torch.rand(2, 4)
Y = net(X).sum()

#可以通过Module类的parameters()或者names_parameters方法来访问所有参数
print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param)

#访问单层的参数
for name, param in net[0].named_parameters():
    print(name, param, type(param))
#param的类型是torch.nn.parameter.Parameter，与Tensor的区别是如果一个Tensor是Parameter，则会被自动添加法哦模型的参数列表中

#Parameter与Tensor其余都是相同的
weight_0 = list(net[0].parameters())[0]
print(weight_0)
print(weight_0.data)
print( weight_0.grad)
Y.backward()
print(weight_0.grad)


#参数初始化
#init模块中提供了多种预设的初始化方法

#将权重参数初始化为均值为0、标准差为0.01的正态分布随机数，将偏差参数清零
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)


#使用常数初始化权重参数
for name, param in net.named_parameters():
    if 'bias' in name:
        init.constant_(param, val=0)
        print(name, param.data)


#pytorch中的noraml_的实现
def normal_(tensor, mean=0, std=1):
    with torch.no_grad():
        return tensor.normal_(mean, std)

#自定义一半概率初始化为0,另一半概率初始化为[-10. -5]和[5, 10]两个区间里均匀分布的随机数
def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()

for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)

#通过参数的data来改写模型参数数值同时不影响梯度
for name, param in net.named_parameters():
    if 'bias' in name:
        param.data += 1
        print(name, param.data)


#共享参数模型
