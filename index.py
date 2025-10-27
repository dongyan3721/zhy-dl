import torch
from torch import nn
from torch.nn import functional as F

X = torch.ones(3, 3)
y = torch.tensor([
    1, 2, 3
], dtype=torch.float)
X = X.to('mps')
y = y.to('mps')

z = torch.matmul(X, y)

print(z)

