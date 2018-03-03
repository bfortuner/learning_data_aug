import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

import utils


class VAE(nn.Module):
    def __init__(self, in_shape, n_latent):
        super().__init__()
        self.in_shape = in_shape
        self.n_latent = n_latent
        c,h,w = in_shape
        self.z_dim = h//2**2 # receptive field downsampled 3 times
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(c),
            nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),  # 32, 16, 16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32, 8, 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.z_mean = nn.Linear(64 * self.z_dim**2, n_latent)
        self.z_var = nn.Linear(64 * self.z_dim**2, n_latent)
        self.z_develop = nn.Linear(n_latent, 64 * self.z_dim**2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1),
            utils.CenterCrop(h,w),
            nn.Sigmoid()
        )

    def sample_z(self, mean, var):
        std = torch.exp(0.5 * var)
        eps = Variable(torch.randn(std.size())).cuda()
        return (eps * std) + mean

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mean = self.z_mean(x)
        var = self.z_var(x)
        return mean, var

    def decode(self, z):
        out = self.z_develop(z)
        out = out.view(z.size(0), 64, self.z_dim, self.z_dim)
        out = self.decoder(out)
        return out

    def forward(self, x):
        mean, var = self.encode(x)
        z = self.sample_z(mean, var)
        out = self.decode(z)
        return out, mean, var

    def transform(self, x, eps=0.1):
        mean, var = self.encode(x)
        noise = Variable(1. + torch.randn(x.size(0), self.n_latent).cuda() * eps)
        z = self.sample_z(mean*noise, var*noise)
        out = self.decode(z)
        return out


def vae_loss(output, input, mean, var, criterion):
    recon_loss = criterion(output, input)
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(var) + mean**2 - 1. - var, 1))
    return recon_loss + kl_loss

def generate(model, mean, var):
    model.eval()
    mean = Variable(mean.cuda())
    var = Variable(var.cuda())
    z = model.sample_z(mean, var)
    out = model.decode(z)
    return out.data.cpu()

def predict(model, img):
    model.eval()
    if len(img.size()) == 3:
        c,h,w = img.size()
        img = img.view(1,c,h,w)
    img = Variable(img.cuda())
    out, mean, var = model(img)
    return out.data.cpu(), mean.data.cpu(), var.data.cpu()

def predict_batch(model, loader):
    inputs, _ = next(iter(loader))
    out, mu, logvar = predict(model, inputs)
    return out, mu, logvar

def train(model, dataloader, crit, optim):
    model.train()
    total_loss = 0
    for img, _ in dataloader:
        inputs = Variable(img.cuda())

        output, mean, var = model(inputs)
        loss = vae_loss(output, inputs, mean, var, crit)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.data[0]

    return total_loss / len(dataloader)

def test(model, tst_loader, crit):
    model.eval()
    test_loss = 0
    for inputs, targets in tst_loader:
        inputs = Variable(inputs.cuda(), volatile=True)
        target = Variable(targets.cuda())

        output, mean, var = model(inputs)

        loss = vae_loss(output, inputs, mean, var, crit)
        test_loss += loss.data[0]

    test_loss /= len(tst_loader)
    return test_loss

def run(model, trn_loader, tst_loader, crit, optim, epochs, plot_interval=1000):
    losses = {'trn': [], 'tst':[]}
    for epoch in range(epochs):
        trn_loss = train(model, trn_loader, crit, optim)
        tst_loss = test(model, tst_loader, crit)
        print('Epoch %d, TrnLoss: %.4f, TstLoss: %.4f' % (
            epoch+1, trn_loss, tst_loss))
        if epoch % plot_interval == 0:
            samples, mu, var = predict_batch(model, tst_loader)
            utils.plot_batch(samples)
        losses['trn'].append(trn_loss)
        losses['tst'].append(tst_loss)
    samples, mean, var = predict_batch(model, trn_loader)
    utils.plot_batch(samples)
    return losses
