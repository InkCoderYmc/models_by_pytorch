#使用autograd包进行自动求导
import torch

x = torch.ones(2, 2, requires_grad=True) #将.requires_grad设置为True，将追踪其上的所有操作，就可以利用链式法则进行梯度传播
print(x)
print(x.grad_fn)

y = x + 2
print(y)
print(y.grad_fn)
#x是直接创建的，所以没有grad_fn，y是通过一个加法操作创建的，所以有一个grad_fn

print(x.is_leaf, y.is_leaf)

z = y * y * 3
out = z.mean()
print(z, out)

a = torch.randn(2, 2) #缺失情况下默认requires_grad = False
a = ((a * 3) / (a - 1))
print(a)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b)
print(b.grad_fn)

out.backward()
print(x.grad)

#不允许张量对张量求导，只允许标量对张量求导，求导结果是和自变量同形的张量
out2 = x.sum()
out2.backward()
print(x.grad)

out3 = x.sum()
x.grad.data.zero_() #grad在反向传播过程中是累加的，所以一般在反向传播之前需把梯度清零
out3.backward()
print(x.grad)


x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print(z) #z现在是一个张量

v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)
print(x.grad) #x.grad是和x同形的张量

x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2

print(x.requires_grad)
print(y1, y1.requires_grad)
print(y2, y2.requires_grad) #y2没有grad_fn
print(y3, y3.requires_grad)

y3.backward()
print(x.grad) #这里是因为y3得不到y2的grad，只计算了y1

#若想修改tensor的值，且不影响反向传播，则可以对tensor.data进行操作
x = torch.ones(1, requires_grad=True)

print(x.data)
print(x.data.requires_grad)

y = 2 * x
x.data *= 100

y.backward()
print(x)
print(x.grad)