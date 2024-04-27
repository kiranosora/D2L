import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
y = layer(torch.FloatTensor(np.arange(6)).reshape(-1,2))
print(y)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.output = nn.Linear(256,10)

    def forward(self, X):
        h = F.relu(self.hidden(X))
        return self.output(h)
net = MLP()
X = torch.randn(size=(2,20))
Y = net(X)
filename = 'mlp.params'
torch.save(net.state_dict(),filename)

clone = MLP()
clone.load_state_dict(torch.load(filename))
clone.eval()
Y_clone = clone(X)
print(f"Y_clone == Y:{Y_clone ==Y}")