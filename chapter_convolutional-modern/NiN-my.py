import torch
from torch import nn
from d2l import torch as d2l
import time
beg = time.time()
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

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)
train_iter = WrappedDataset(train_iter, preprocess)
test_iter = WrappedDataset(test_iter, preprocess)

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=strides, padding=padding),
                         nn.ReLU(),
                         nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1), nn.ReLU(),
                         nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1), nn.ReLU()
                         )

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    #nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten()
).to(device)

X = torch.rand(size=(1, 1, 224, 224),device=device)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)


def my_sgd(parameters, lr, batch_size):
    with torch.no_grad():
        for param in parameters:
            param.data -= param.grad * lr
            param.grad.zero_()


def train_epoch(net, train_iter, loss, lr, updater):
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(net.parameters(), lr, batch_size)
        preds = y_hat.reshape(-1, y_hat.shape[1]).argmax(axis=1)
        cmp = preds.type(y.dtype) == y
        metric.add(l.sum(), cmp.sum(), y.numel())
    return metric[0]/metric[2], metric[1]/metric[2]

def train(net, train_iter, test_iter, loss, lr, updater, num_epochs):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    for epoch in range(num_epochs):
        loss_sum, train_acc = train_epoch(net, train_iter, loss, lr, updater)
        metric = d2l.Accumulator(2)
        for X, y in test_iter:
            y_hat = net(X)
            preds = y_hat.reshape(-1, y_hat.shape[1]).argmax(axis=1)
            cmp = preds.type(y.dtype) == y
            metric.add(cmp.sum(), y.numel())
        test_acc = metric[0]/metric[1]
        print(f"epoch: {epoch}, loss: {loss_sum}, train_acc: {train_acc}, test_acc: {test_acc}")

loss = nn.CrossEntropyLoss()
updater = torch.optim.SGD(net.parameters(), lr=lr)
updater = my_sgd
train(net, train_iter,test_iter, loss, lr, updater, num_epochs)
#d2l.train_ch6(net=net, train_iter=train_iter, test_iter=test_iter, num_epochs=num_epochs, lr=lr, device=d2l.try_gpu())