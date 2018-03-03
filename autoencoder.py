import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

import utils


class AE(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        c,h,w = in_shape
        self.encoder = nn.Sequential(
            nn.Linear(c*h*w, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, c*h*w),
            nn.Sigmoid())

    def forward(self, x):
        bs,c,h,w = x.size()
        x = x.view(bs, -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(bs, c, h, w)
        return x


class ConvAE(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        c,h,w = in_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1),  # b, 16, 32, 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # b, 16, 16, 16
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),  # b, 8, 16, 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # b, 8, 8, 8
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=0),  # 16, 17, 17
            nn.ReLU(),
            nn.ConvTranspose2d(16, c, kernel_size=3, stride=2, padding=1),  # 3, 33, 33
            utils.CenterCrop(h, w),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(model, loader, criterion, optim):
    model.train()
    total_loss = 0
    for img, _ in loader:
        inputs = Variable(img.cuda())

        output = model(inputs)
        loss = criterion(output, inputs)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.data[0]

    mean_loss = total_loss / len(loader)
    return mean_loss

def predict(model, img):
    model.eval()
    if len(img.size()) == 3:
        c,h,w = img.size()
        img = img.view(1,c,h,w)
    img = Variable(img.cuda())
    out = model(img).data.cpu()
    return out

def predict_batch(model, loader):
    inputs, _ = next(iter(loader))
    out = predict(model, inputs)
    return out

def run(model, trn_loader, crit, optim, epochs, plot_interval=1000):
    losses = []
    for epoch in range(epochs):
        loss = train(model, trn_loader, crit, optim)
        print('Epoch {:d} Loss: {:.4f}'.format(epoch+1, loss))
        if epoch % plot_interval == 0:
            samples = predict_batch(model, trn_loader)
            utils.plot_batch(samples)
        losses.append(loss)
    samples = predict_batch(model, trn_loader)
    utils.plot_batch(samples)
    return losses
