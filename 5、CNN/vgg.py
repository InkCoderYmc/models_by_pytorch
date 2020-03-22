import time
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

# 组成规则：连续使用数个相同的填充为1、窗口形状为3×3的卷积
# 后接上一个步幅为2、窗口形状为2×2的最大池化层

# 使用vgg_block实现基础的vgg块


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels,
                                 kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels,
                                 kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 会使宽高减半
    return nn.Sequential(*blk)


conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256),
             (2, 256, 512), (2, 512, 512))

# 经过5个vgg_block块，宽高会变成 224/32=7
fc_featrues = 512 * 7 * 7
fc_hidden_units = 4096


def vgg_11(conv_arch, fc_featrues, fc_hidden_units):
    net = nn.Sequential()

    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module('vgg_block_' + str(i+1),
                       vgg_block(num_convs, in_channels, out_channels))
    net.add_module('fc', nn.Sequential(
        FlattenLayer(),
        nn.Linear(fc_featrues, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, 10)
    ))
    return net


net = vgg_11(conv_arch, fc_featrues, fc_hidden_units)
# 观察每一层的输出形状
X = torch.rand(1, 1, 224, 224)

for name, blk in net.named_children():
    X = blk(X)
    #print(name, 'outputs shape:', X.shape)
'''
vgg_block_1 outputs shape: torch.Size([1, 64, 112, 112])
vgg_block_2 outputs shape: torch.Size([1, 128, 56, 56])
vgg_block_3 outputs shape: torch.Size([1, 256, 28, 28])
vgg_block_4 outputs shape: torch.Size([1, 512, 14, 14])
vgg_block_5 outputs shape: torch.Size([1, 512, 7, 7])
fc outputs shape: torch.Size([1, 10])
'''

# 获取数据
ratio = 8  # 测试使用，构造通道数更小的网络
small_conv_arch = [(1, 1, 64//ratio), (1, 64//ratio, 128//ratio), (2, 128 //
                                                                   ratio, 256//ratio), (2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio)]
net = vgg_11(small_conv_arch, fc_featrues//ratio, fc_hidden_units//ratio)
# print(net)
'''
Sequential(
  (vgg_block_1): Sequential(
    (0): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (vgg_block_2): Sequential(
    (0): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (vgg_block_3): Sequential(
    (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (vgg_block_4): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (vgg_block_5): Sequential(
    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc): Sequential(
    (0): FlattenLayer()
    (1): Linear(in_features=3136, out_features=512, bias=True)
    (2): ReLU()
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=512, out_features=512, bias=True)
    (5): ReLU()
    (6): Dropout(p=0.5, inplace=False)
    (7): Linear(in_features=512, out_features=10, bias=True)
  )
)
'''


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


# 训练
batch_size = 64
train_iter, test_iter = load_data_fashion_mnist_toImageNet(
    batch_size, resize=224)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_ch5(net, train_iter, test_iter, batch_size,
          optimizer, device, num_epochs)
