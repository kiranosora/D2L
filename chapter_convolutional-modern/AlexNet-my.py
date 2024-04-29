import sys

import torch
from ptflops import get_model_complexity_info
from torch import nn
from d2l import torch as d2l

device = "cuda:0"


#device = "cpu"
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


batch_size, num_epochs, lr = 128, 10, 0.01
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)
train_iter = WrappedDataset(train_iter, preprocess)
test_iter = WrappedDataset(test_iter, preprocess)

net = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Flatten(),
                    nn.Linear(6400, 4096), nn.ReLU(),
                    nn.Linear(4096, 4096), nn.ReLU(),
                    nn.Linear(4096, 1000), nn.ReLU(),
                    nn.Linear(1000, 10)
                    ).to(device=device)

# X = torch.randn(1, 1, 224, 224, device=device)
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape:\t', X.shape)


def my_sgd(params, batch_size, lr):
    for param in params:
        param.data -= param.grad * lr
        param.grad.zero_()


loss = nn.CrossEntropyLoss()

def train_epoch(net, train_iter, lr, updater):
    metric = d2l.Accumulator(2)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        #print(f"l.sum: {l.sum()}")
        net.zero_grad()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(net.parameters(), batch_size=batch_size, lr=lr)
        preds = y_hat.reshape((-1, y_hat.shape[-1])).argmax(axis=1)
        cmp = preds.type(y.dtype) == y
        metric.add(cmp.sum(), y.numel())
    return metric[0] / metric[1]


for param in net.parameters():
    print(f"params: {param}")



def train(net, train_iter, test_iter, lr, updater):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    for epoch in range(num_epochs):
        train_acc = train_epoch(net, train_iter, lr, updater)
        metric = d2l.Accumulator(2)
        for X, y in test_iter:
            y_hat = net(X)
            preds = y_hat.reshape((-1, y_hat.shape[-1])).argmax(axis=1)
            cmp = preds.type(y.dtype) == y
            metric.add(cmp.sum(), y.numel())
        test_acc = metric[0] / metric[1]
        print(f"epoch= {epoch}, train_acc: {train_acc}, test_acc: {test_acc}")

model_name = 'AlexNet'
flops, params = get_model_complexity_info(net, (1, 224, 224), as_strings=True,
print_per_layer_stat=True)
print("%s |%s |%s" % (model_name, flops, params))
train(net, train_iter, test_iter, lr, my_sgd)

#train(net, train_iter, test_iter, lr, torch.optim.SGD(net.parameters(), lr=lr))
#train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
