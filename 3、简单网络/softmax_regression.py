import torch
import torchvision
import torchvision.transforms as transforms
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import sys


# 小批量随机梯度下降算法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


# 使用fashion-MNIST数据集
batch_size = 256
#train_iter, test_iter = load_data_fashion_mnist(batch_size)


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


# 初始化模型参数
# 输入向量为28×28的图像，共784个像素点
num_inputs = 784
# 共有10个类别
num_outputs = 10

# 根据计算得出weight是784×10的矩阵(weight不设置为0,且不相同，这里初始化为均值为0,均方差为0.01的正态分布)
W = torch.tensor(np.random.normal(
    0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
# bias是1×10的矩阵
b = torch.zeros(num_outputs, dtype=torch.float)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


def softmax(X):  # 将所有输入压缩到[0,1]范围内
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdims=True)  # 按行求和
    return X_exp / partition

# 定义模型


def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

# 定义损失函数，这里使用交叉熵


def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

# 准确度函数


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

#print(evaluate_accuracy(test_iter, net))


# 训练模型
num_epochs, lr = 5, 0.1


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


train_ch3(net, train_iter, test_iter, cross_entropy,
          num_epochs, batch_size, [W, b], lr)


# 将数值标签转换成相应的文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shiet', 'trouser', 'pullover', 'dress',
                   'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def use_svg_display():
    display.set_matplotlib_formats('svg')  # 定义为用矢量图显示


def show_fashion_mnist(images, labels):
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))

    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# 预测
X, y = iter(test_iter).next()

true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

show_fashion_mnist(X[0:9], titles[0:9])
