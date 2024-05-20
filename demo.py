from cuda_add.api import cuda_add
import torch

n = 100000
a = torch.rand([n]).cuda()
b = torch.rand([n]).cuda()
c = cuda_add(a, b)
d = a + b
loss = torch.sum(torch.abs(c - d)).item()
print("loss:", loss)
assert loss == 0