import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

import utils


class CNN(nn.Module):
    def __init__(self, in_shape, n_classes):
        super().__init__()
        c, w, h = in_shape
        pool_layers = 3
        fc_h = int(h / 2**pool_layers)
        fc_w = int(w / 2**pool_layers)
        self.features = nn.Sequential(
            *conv_bn_relu(c, 8, kernel_size=1, stride=1, padding=0),
            *conv_bn_relu(8, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2), #size/2
            *conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2), #size/2
            *conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2), #size/2
        )
        self.classifier = nn.Sequential(
            *linear_bn_relu_drop(64 * fc_h * fc_w, 256, dropout=0.25),
            nn.Linear(256, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Trainer():
    def __init__(self):
        self.metrics = {
            'loss': {
                'trn':[],
                'tst':[]
            },
            'accuracy': {
                'trn':[],
                'tst':[]
            },
        }

    def run(self, model, trn_loader, tst_loader, crit, optim, epochs):
        for epoch in range(1, epochs+1):
            trn_loss, trn_acc = train(model, trn_loader, crit, optim)
            tst_loss, tst_acc = test(model, tst_loader, crit)
            print('Epoch %d, TrnLoss: %.3f, TrnAcc: %.3f, TstLoss: %.3f, TstAcc: %.3f' % (
                epoch, trn_loss, trn_acc, tst_loss, tst_acc))
            self.metrics['loss']['trn'].append(trn_loss)
            self.metrics['loss']['tst'].append(tst_loss)
            self.metrics['accuracy']['trn'].append(trn_acc)
            self.metrics['accuracy']['tst'].append(tst_acc)


def train(net, loader, crit, optim):
    net.train()
    n_batches = len(loader)
    total_loss = 0
    total_acc = 0
    for data in loader:
        inputs = Variable(data[0].cuda())
        targets = Variable(data[1].cuda())

        output = net(inputs)
        loss = crit(output, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        preds = utils.get_argmax(output)
        accuracy = utils.get_accuracy(preds, targets.data.cpu().numpy())

        total_loss += loss.data[0]
        total_acc += accuracy

    mean_loss = total_loss / n_batches
    mean_acc = total_acc / n_batches
    return mean_loss, mean_acc

def test(net, tst_loader, criterion):
    net.eval()
    test_loss = 0
    test_acc = 0
    for data in tst_loader:
        inputs = Variable(data[0].cuda(), volatile=True)
        target = Variable(data[1].cuda())
        output = net(inputs)
        test_loss += criterion(output, target).data[0]
        pred = utils.get_argmax(output)
        test_acc += utils.get_accuracy(pred, target.data.cpu().numpy())
    test_loss /= len(tst_loader)
    test_acc /= len(tst_loader)
    return test_loss, test_acc

def conv_bn_relu(in_chans, out_chans, kernel_size=3, stride=1,
                 padding=1, bias=False):
    return [
        nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_chans),
        nn.ReLU(inplace=True),
    ]

def linear_bn_relu_drop(in_chans, out_chans, dropout=0.5, bias=False):
    layers = [
        nn.Linear(in_chans, out_chans, bias=bias),
        nn.BatchNorm1d(out_chans),
        nn.ReLU(inplace=True)
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers
