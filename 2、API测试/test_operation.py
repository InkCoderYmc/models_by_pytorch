import torch

x = torch.ones(5, 3)
y = torch.rand(5, 3)
print(x)
print(y)
print(x + y)

print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)
print(y)
