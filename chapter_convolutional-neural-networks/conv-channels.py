import torch
from d2l import torch as d2l
import time
import numpy as np
beg = time.time()
device = "cuda:0"


#device = "cpu"

def corr2d(X, K, stride=1):
    h, w = K.shape
    Y = torch.zeros(size=(X.shape[0] - h + 1, X.shape[1] - w + 1), device=device)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


def corr2d_multi_in(X, K):
    # iterate over 0th dimension(input channel dimension) and return sum
    return sum(corr2d(x, k) for x, k in zip(X, K))
a = np.arange(9, dtype=float).reshape((3,3))
X = torch.tensor(np.array([a, a+1]), device=device)
print(X)
b = np.arange(4, dtype=float).reshape(2,-1)
K = torch.tensor(np.array([b, b+1]), device=device)
print(corr2d_multi_in(X, K))

def corr2d_multi_in_out(X,K):
    # iterate over 0th dimension( output channel)
    return torch.stack([corr2d_multi_in(X,k) for k in K], 0)

K = torch.stack((K, K+1, K+2), 0)
print(corr2d_multi_in_out(X, K))

def corr2d_multi_in_out_1x1(X,K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h*w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

X = torch.normal(0, 1, (3, 3, 3), device=device)
K = torch.normal(0, 1, (2, 3, 1, 1), device=device)
Y1 = corr2d_multi_in_out(X, K)
Y2 = corr2d_multi_in_out_1x1(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6