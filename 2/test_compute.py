import torch

#触发广播机制
#当两个形状不同的Tensor按元素运算时，可能会触发广播机制
x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)

#索引操作不会开辟新内存
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = x + y
print(id_before)
print(id(y))
print(id(y) == id_before)

#通过[:]操作也就是索引操作将数据写回y中
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = x + y
print(id(y) == id_before)

#使用运算函数的out参数来达到上述效果
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y)
print(id(y) == id_before)

#将Tensor和Numpy互相转换

#使用numpy()函数转换
a = torch.ones(5)
b = a.numpy()
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)

#使用from_numpy()转换
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)

#使用torch.tensor()转换，这会进行数据拷贝，返回的tensor与原数据不再共享内存
c = torch.tensor(a)
a += 1
print(a, c)