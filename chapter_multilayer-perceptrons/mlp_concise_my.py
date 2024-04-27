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

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 28 * 28, 10, 256, 12

net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs,num_hiddens1),
                    nn.ReLU(),
                    nn.Linear(num_hiddens1, num_outputs)).to(device=device)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

lr, num_epochs = 0.1, 40
loss = nn.CrossEntropyLoss(reduction='none')
weight_decay = 0.01
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
trainer = torch.optim.SGD([{"params": net[1].weight, 'weight_decay':weight_decay},
                           {"params":net[1].bias},
                           {"params":net[3].weight},
                           {"params":net[3].bias}], lr=lr)


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


train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

print(f"device: {device}, tot time usd: {time.time() - beg} seconds")

def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + "\n" + pred for true, pred in zip(trues, preds)]
    X_ = X[0:n].cpu()
    d2l.show_images(X_.reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter,10)
d2l.plt.show()
input("press any key")