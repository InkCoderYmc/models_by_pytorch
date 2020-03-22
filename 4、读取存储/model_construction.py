# 基于Module类的模型构造
import torch
from torch import nn

# 继承Module类定义需要定义的网络
# 重载__init__函数「创建模型参数」和forward函数「定义正向传播」


class MLP(nn.Module):
    # 声明带有模型参数的层，两个全连接层
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)  # 隐藏层
        self.act = nn.ReLU()  # 激活函数
        self.output = nn.Linear(256, 10)  # 输出层

    # 定义模型的正向传播
    def forward(self, x):
        a = self.act(self.hidden(x))
        # print(a.shape)
        return self.output(a)


'''
#测试
X = torch.rand(2, 784)
net = MLP()
print(net)
print(net(X))
'''

# pytorch实现了继承自Module的可以方便构建模型的类：Sequential、ModuleList、ModuleDict

# Sequential类的模仿实现


class MySequential(nn.Module):
    from collections import OrderedDict

    def __init__(self, *args):
        super(MySequential, self).__init__()
        # r如果传入的是一个OrderedDict
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].item():
                # add_module函数将module添加进self._modules(一个OrderedDict)
                self.add_module(key, module)
        else:  # 如果传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        # self._modules返回一个OrderedDict，保证会按照成员添加时的顺序遍历成员
        for module in self._modules.values():
            input = module(input)
        return input


'''
X = torch.rand(2, 784)
net = MySequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
)
print(net)
print(net(X))
'''

# 用ModuleList实现


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x


class Module_ModuleList(nn.Module):
    def __init__(self):
        super(Module_ModuleList, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10)])


class Module_List(nn.Module):
    def __init__(self):
        super(Module_List, self).__init__()
        self.linears = [nn.Linear(10, 10)]


net1 = Module_ModuleList()
net2 = Module_List()

print('net1:')
for p in net1.parameters():
    print(p.size())

print('net2:')
for p in net2.parameters():
    print(p)

# ModuleDict实现
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10)
print(net['linear'])
print(net.output)
print(net)

# 构建模型


class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        # 使用创建的常数参数，以及relu和mm函数
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)

        # 复用全连接层
        x = self.linear(x)
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()


'''
X = torch.rand(2, 20)
net = FancyMLP()
print(net)
print(net(X))
'''

# 嵌套调用，因为FancyMLP和Sequential类都是Module类的子类


class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())

    def forward(self, x):
        return self.net(x)


net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())

X = torch.rand(2, 40)
print(net)
print(net(X))
