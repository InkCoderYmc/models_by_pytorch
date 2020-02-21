import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import time
import sys

#下载数据集
mnist_train = torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())
#transforms.ToTensor()将尺寸为H×W×C的数据在[0, 255]的PIL图片或者数据类型为np.unit8的Numpy数组转换为尺寸为C×H×W且数据类型为torch.float32范围为[0.0, 1.0]的Tensor

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))
#每个类别的图像分别有6000和1000,分为训练集和数据集
#共有10类，总数为60000和10000

#通过下标访问样本
feature, label = mnist_train[0]
print(feature.shape, label)

#将数值标签转换成相应的文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shiet', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def use_svg_display():
    display.set_matplotlib_formats('svg') #定义为用矢量图显示

def show_fashion_mnist(images, labels):
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))

    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))

batch_size = 256
#多进程读取数据
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

start = time.time()

for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))