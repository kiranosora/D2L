import torch
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

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU()),
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    #every pooling layer 2x2(stride=2) causes h/2, w/2
    #4 vgg causes h/(2^5), w/(2^5) = 224/32, 224/32 = 7, 7
    return nn.Sequential(*conv_blks, nn.Flatten(),
                         nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
                         nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                         nn.Linear(4096, 10)
                         )


net = vgg(conv_arch)
batch_size = 10
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)

ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
print(f"small_conv_arch: {small_conv_arch}")
net = vgg(small_conv_arch).to(device)

lr, num_epochs, batch_size = 0.05, 10, 128

def my_sgd(params, batch_size, lr):
    for param in params:
        param.data -= param.grad * lr
        param.grad.zero_()

loss = nn.CrossEntropyLoss()
def train_epoch(net, train_iter, lr, updater):
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
            updater(net.parameters(), batch_size, lr)
        #print(f"y_hat shape: {y_hat.shape}, y.shape: {y.shape}")
        preds = y_hat.reshape((-1, y_hat.shape[-1])).argmax(axis=1)
        cmp = preds.type(y.dtype) == y
        metric.add(l.sum(), cmp.sum(), X.shape[0])
    return metric[0]/metric[2], metric[1]/metric[2]

def train(net, train_iter, test_iter, lr, updater):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    for epoch in range(num_epochs):
        loss, train_acc = train_epoch(net, train_iter, lr, updater)
        metric = d2l.Accumulator(2)
        for X, y in test_iter:
            y_hat = net(X)
            preds = y_hat.reshape((-1, y_hat.shape[-1])).argmax(axis=1)
            cmp = preds.type(y.dtype) == y
            metric.add(cmp.sum(), y.numel())
        print(f"epoch={epoch}, loss = {loss:.3f}, train_acc: {train_acc:.3f}, test_acc: {metric[0]/metric[1]}")


train(net, train_iter, test_iter, lr, my_sgd)

