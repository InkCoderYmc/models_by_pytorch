import torch

#索引
x = torch.zeros(5, 3)
print(x)

y = x[0, :]
print(y)

y +=1
print(y)
print(x[0, :])
print(x)

#改变形状
y = x.view(15)
z = x.view(-1, 5)
print(x.size(), y.size(), z.size())

x += 1
print(x)
print(y)

x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)

x = torch.randn(1)
print(x)
print(x.item())

