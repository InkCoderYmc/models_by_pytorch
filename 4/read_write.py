#有时需要把训练好摸模型部署到很多不同的设备，这种情况下，可以把训练好的模型参数存储在硬盘上供后续读取使用
import torch
from torch import nn


#读写Tensor
x = torch.ones(3)
torch.save(x, 'x.pt')

x2 = torch.load('x.pt')
print(x2)

#存储一个Tensor列表并读回内存
y = torch.zeros(4)
torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
print(xy_list)

#存储并读取一个从字符串映射到Tensor的字典
torch.save({'x': x, 'y': y}, 'xy_dict.pt')
xy = torch.load('xy_dict.pt')
print(xy)


#读写模型
#pytorch中参数parameters()访问，state_dict是一个从参数名称映射到参数Tensor的字典对象
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

net = MLP()
print(net.state_dict())

#优化器的state_dict()，是优化器的超参数
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
print(optimizer.state_dict())

#保存和加载模型
#保存
'''
torch.save(model, PATH)
'''
#加载
'''
model = torch.load(PATH)
'''

X = torch.rand(2, 3)
Y = net(X)

PATH = './net.pt'
torch.save(net.state_dict(), PATH)

net2 = MLP()
net2.load_state_dict(torch.load(PATH))
Y2 = net2(X)
print(Y)
print(Y2)
print(Y2 == Y)