import time
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


def load_data_fashion_mnist_toImageNet(batch_size, resize=None):
    trans = []

    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))

    trans.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(
        root='../Datasets/FashionMNIST', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../Datasets/FashionMNIST', train=False, download=True, transform=transform)

    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) ==
                            y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:  # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1)
                                == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

# 稠密连接网络与残差网络的区别在于跨层连接上用连结

# 结构上进行优化


def conv_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    )
    return blk

# 定义稠密块


class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels  # 计算输出通道数

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X


blk_1 = DenseBlock(2, 3, 10)
X = torch.rand(4, 3, 8, 8)
Y = blk_1(X)
# 通道数为3+2×10=23
# print(Y.shape)
'''
torch.Size([4, 23, 8, 8])
'''

# 每个稠密块带来通道数的增加，使用过渡层来控制模型复杂度，使用1×1卷积层来减少通道数，使用步幅为2的平均池化层减半高和宽


def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )
    return blk


blk_2 = transition_block(23, 10)
Z = blk_2(Y)
# print(Z.shape)
'''
torch.Size([4, 10, 4, 4])
'''

# 构造DenseNet模型
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    DB = DenseBlock(num_convs, num_channels, growth_rate)
    net.add_module('DenseBlock_%d' % i, DB)
    # 上一个稠密块的输出通道数
    num_channels = DB.out_channels
    # 稠密块之间加入通道数减半的过渡层
    if i != len(num_convs_in_dense_blocks) - 1:
        net.add_module('transition_block_%d' %
                       i, transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

net.add_module('BN', nn.BatchNorm2d(num_channels))
net.add_module('relu', nn.ReLU())
net.add_module('global_avg_pool', GlobalAvgPool2d())
net.add_module('fc', nn.Sequential(
    FlattenLayer(), nn.Linear(num_channels, 10)))

# 检查网络
X = torch.rand((1, 1, 96, 96))
for name, layer in net.named_children():
    X = layer(X)
    #print(name, 'output shape:\t', X.shape)
'''
0 output shape:  torch.Size([1, 64, 48, 48])
1 output shape:  torch.Size([1, 64, 48, 48])
2 output shape:  torch.Size([1, 64, 48, 48])
3 output shape:  torch.Size([1, 64, 24, 24])
DenseBlock_0 output shape:       torch.Size([1, 192, 24, 24])
transition_block_0 output shape:         torch.Size([1, 96, 12, 12])
DenseBlock_1 output shape:       torch.Size([1, 224, 12, 12])
transition_block_1 output shape:         torch.Size([1, 112, 6, 6])
DenseBlock_2 output shape:       torch.Size([1, 240, 6, 6])
transition_block_2 output shape:         torch.Size([1, 120, 3, 3])
DenseBlock_3 output shape:       torch.Size([1, 248, 3, 3])
BN output shape:         torch.Size([1, 248, 3, 3])
relu output shape:       torch.Size([1, 248, 3, 3])
global_avg_pool output shape:    torch.Size([1, 248, 1, 1])
fc output shape:         torch.Size([1, 10])
'''

# 获取数据并训练
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist_toImageNet(
    batch_size, resize=96)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_ch5(net, train_iter, test_iter, batch_size,
          optimizer, device, num_epochs)
