import torch
from torch import nn
from d2l import torch as d2l
import numpy as np
import time
beg = time.time()
device = "cuda:0"
#device = "cpu"

def corr2d(X, K, stride=1):
    h, w = K.shape
    Y = torch.zeros(size=(X.shape[0] -h +1, X.shape[1] -w +1), device=device)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

X = torch.tensor(np.arange(9), device=device).reshape(3,-1)
K = torch.tensor([[0.,1.], [2., 3.]], device=device)
print(corr2d(X, K))
print(f"device: {device}, time used:{time.time() - beg} seconds")

class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size, device=device, requires_grad=True))
        self.bias = nn.Parameter(torch.zeros(1, device=device, requires_grad=True))

    def forward(self, X):
        return corr2d(X,self.weight) + self.bias

X = torch.ones((6, 8), device=device)
X[:, 2:6] = 0
print(X)
K = torch.tensor([[1.0, -1.0]], device=device)
Y = corr2d(X, K)
print(Y)

net = Conv2D((1,2)).to(device=device)
lr = 3e-2
for i in range(20):
    Y_hat = net(X)
    loss = (Y - Y_hat)**2
    net.zero_grad()
    loss.sum().backward()
    net.weight.data[:] -= lr * net.weight.grad
    #net.bias.data[:] -= lr * net.bias.grad
    print(f"epoch {i+1}, loss: {loss.sum():.3f}, net.weight: {net.weight.data}, net.bias: {net.bias.data}")