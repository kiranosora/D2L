import torch
from torch import nn
from d2l import torch as d2l
import time

beg = time.time()
batch_size = 256
device = "cuda:0"


# device = "cpu"


def preprocess(x, y):
    x = x.to(device)
    y = y.to(device)
    return x, y


class WrappedDataset():
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield self.func(*b)


train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
train_iter = WrappedDataset(train_iter, preprocess)
test_iter = WrappedDataset(test_iter, preprocess)


def dropout_layer(X, dropout: float):
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape).to(device) > dropout).float()
    #print(f"mask: {mask}\n X: {X}, dropout: {dropout}")
    return mask * X / (1.0 - dropout)


X = torch.arange(16, dtype=torch.float32, device=device).reshape((2, 8))
print(X)
print(dropout_layer(X, 0))
print(dropout_layer(X, 1))
print(dropout_layer(X, 0.5))

dropout1, dropout2 = 0.2, 0.5


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape(-1, self.num_inputs)))
        if self.training:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
net = Net(num_inputs=num_inputs, num_outputs=num_outputs, num_hiddens1=num_hiddens1,
          num_hiddens2=num_hiddens2).to(device)
# equal to
# net = nn.Sequential(nn.Flatten(),
#                     nn.Linear(num_inputs,num_hiddens1),
#                     nn.ReLU(),
#                     nn.Dropout(dropout1),
#                     nn.Linear(num_hiddens1, num_hiddens2),
#                     nn.ReLU(),
#                     nn.Dropout(dropout2),
#                     nn.Linear(num_hiddens2, num_outputs))
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)


class Accumulator:
    def __init__(self, n: int):
        self.data = [0.] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # find the predicted result with the largest prob for every sample in batch
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()  # eval model, skip backward (cal grad)
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            # X = X.to(device=device)
            # y = y.to(device=device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss, updater):
    #iterate each batch
    #update each batch
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(batch_size)
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(
            f"({epoch + 1}/{num_epochs}) train_loss: {train_metrics[0]}, train_acc: {train_metrics[1]}, test_acc: {test_acc}")


train_ch3(net=net, train_iter=train_iter, test_iter=test_iter, loss=loss, num_epochs=num_epochs, updater=trainer)

print(f"device: {device}, tot time usd: {time.time() - beg} seconds")
