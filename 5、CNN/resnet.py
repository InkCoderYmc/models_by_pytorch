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

# 定义残差块


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


# 输入输出情况
blk = Residual(3, 3)
X = torch.rand((4, 3, 6, 6))
# print(blk(X).shape)
'''
torch.Size([4, 3, 6, 6])
'''

# 增加通道数，减半输出的高和宽
blk = Residual(3, 6, use_1x1conv=True, stride=2)
# print(blk(X).shape)
'''
torch.Size([4, 6, 3, 3])
'''

# ResNet
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# 定义resnet块


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels,
                                use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


# 为ResNet加入残差块
net.add_module('resnet_block1', resnet_block(64, 64, 2, first_block=True))
net.add_module('resnet_block2', resnet_block(64, 128, 2))
net.add_module('resnet_block3', resnet_block(128, 256, 2))
net.add_module('resnet_block4', resnet_block(256, 512, 2))

# 加入全局平均池化层后接上全连接层输出
net.add_module('global_avg_pool', GlobalAvgPool2d())  # 输出为(Batch, 512, 1, 1)
net.add_module('fc', nn.Sequential(FlattenLayer(), nn.Linear(512, 10)))

X = torch.rand(1, 1, 224, 224)
for name, layer in net.named_children():
    X = layer(X)
    #print(name, 'output shape:\t', X.shape)
'''
0 output shape:  torch.Size([1, 64, 112, 112])
1 output shape:  torch.Size([1, 64, 112, 112])
2 output shape:  torch.Size([1, 64, 112, 112])
3 output shape:  torch.Size([1, 64, 56, 56])
resnet_block1 output shape:      torch.Size([1, 64, 56, 56])
resnet_block2 output shape:      torch.Size([1, 128, 28, 28])
resnet_block3 output shape:      torch.Size([1, 256, 14, 14])
resnet_block4 output shape:      torch.Size([1, 512, 7, 7])
global_avg_pool output shape:    torch.Size([1, 512, 1, 1])
fc output shape:         torch.Size([1, 10])
'''

# 获取数据和训练
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist_toImageNet(
    batch_size, resize=96)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_ch5(net, train_iter, test_iter, batch_size,
          optimizer, device, num_epochs)
