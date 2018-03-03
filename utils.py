import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchvision


class CenterCrop(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width

    def forward(self, img):
        bs, c, h, w = img.size()
        xy1 = (w - self.width) // 2
        xy2 = (h - self.height) // 2
        img = img[:, :, xy2:(xy2 + self.height), xy1:(xy1 + self.width)]
        return img

def plot_tensor(img, fs=(10,10), title=""):
    if len(img.size()) == 4:
        img = img.squeeze(dim=0)
    npimg = img.numpy()
    plt.figure(figsize=fs)
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.title(title)
    plt.show()

def plot_batch(samples, title="", fs=(10,10)):
    plot_tensor(torchvision.utils.make_grid(samples), fs=fs, title=title)

def plot_metric(trn, tst, title):
    plt.plot(np.stack([trn, tst], 1));
    plt.title(title)
    plt.show()

def get_accuracy(preds, targets):
    correct = np.sum(preds==targets)
    return correct / len(targets)

def get_argmax(output):
    val,idx = torch.max(output, dim=1)
    return idx.data.cpu().view(-1).numpy()

def predict_batch(net, inputs):
    v = Variable(inputs.cuda(), volatile=True)
    return net(v).data.cpu().numpy()

def get_probabilities(model, loader):
    model.eval()
    return np.vstack(predict_batch(model, data[0]) for data in loader)

def get_predictions(probs, thresholds):
    preds = np.copy(probs)
    preds[preds >= thresholds] = 1
    preds[preds < thresholds] = 0
    return preds.astype('uint8')
