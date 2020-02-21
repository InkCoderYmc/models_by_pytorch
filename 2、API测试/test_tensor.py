import torch

x=torch.empty(5,3)
print(x)

y=torch.rand(5,3)
print(y)

z = torch.zeros(5, 3, dtype=torch.long)
print(z)

x_1 = torch.tensor([5.5, 3])
print(x_1)

x = x.new_ones(5, 3, dtype=torch.float64)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

print(x.size())
print(x.shape)