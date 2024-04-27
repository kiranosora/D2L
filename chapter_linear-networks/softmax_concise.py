import torch
from torch import nn
from IPython import display
from d2l import torch as d2l

import time

beg = time.time()
batch_size = 512
device = "cuda:0"


#device="cpu"
def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(device), y.to(device)


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(*b))


train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
train_iter = WrappedDataLoader(train_iter, preprocess)
test_iter = WrappedDataLoader(test_iter, preprocess)

print(f"train_iter: type{type(train_iter)}")
num_inputs = 28 * 28
num_outputs = 10

net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_outputs*4), nn.ReLU(), nn.Linear(num_outputs*4, num_outputs)).to(device=device)


def init_weights(m: nn.Module):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10
d2l.train_ch6


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        #find the predicted result with largest prob for every sample in batch
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


class Accumulator:
    def __init__(self, n: int):
        self.data = [0.] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, idx):
        return self.data[idx]


print(evaluate_accuracy(net, test_iter))


def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        # X = X.to(device)
        # y = y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g--', 'r:'), nrows=1, ncols=1, figsize=(3.5 * 4, 2.5 * 4)):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
        #d2l.plt.show()


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.1, 1.0],
                        legend=['train loss', 'train_acc', 'test_acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    print(f"test_acc: {test_acc}")
    #assert train_acc<0.5, train_loss
    #assert train_acc<=1 and train_acc>0.7, train_acc
    #assert test_acc<=1 and test_acc>0.7, test_acc


lr = 0.1

num_epochs = 200
train_ch3(net=net, train_iter=train_iter, test_iter=test_iter, loss=loss, num_epochs=num_epochs, updater=trainer)
print(f"time used: {time.time() - beg} seconds")


def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + "\n" + pred for true, pred in zip(trues, preds)]
    X_ = X[0:n].cpu()
    d2l.show_images(X_.reshape((n, 28, 28)), 1, n, titles=titles[0:n])


predict_ch3(net, test_iter, 10)
d2l.plt.show()
input("press any key")
