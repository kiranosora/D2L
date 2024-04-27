import torch
from torch import nn
from d2l import torch as d2l

device = "cuda:0"


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y


X = torch.arange(9, device=device, dtype=float).reshape(3, -1)
print(pool2d(X, (2, 2), 'avg'))
print(pool2d(X, (2, 2), 'max'))

X = torch.arange(16, dtype=torch.float32, device=device).reshape((1, 1, 4, 4))


