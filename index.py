import torch

preds = torch.randn(10, 5).softmax(dim=-1)
target = torch.randint(5, (10,))


print(preds)
print(target)
