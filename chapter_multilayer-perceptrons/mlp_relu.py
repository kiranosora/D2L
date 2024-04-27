import torch
from d2l import torch as d2l
import time
beg = time.time()
device = "cuda:0"
#device = "cpu"
x = torch.arange(-8., 8., 1e-2, requires_grad=True, device=device)
w = torch.normal(mean=0, std=0.1, size=(x.shape[0], x.shape[0]), requires_grad=False, device=device)
y = torch.relu(torch.matmul(x, w))
print(f"device: {device}, time used: {time.time() - beg} seconds")
d2l.plot(x.detach().cpu(), y.detach().cpu(), 'x', 'relu(x)', figsize=(20, 10))
d2l.plt.show()
input("press any key")


