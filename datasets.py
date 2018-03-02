import torch
import torchvision
from torchvision import transforms


CIFAR_CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

def get_cifar_loader(bs=64, trn_size=50000, tst_size=10000):
    trainTransform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='data/', train=True, download=True, transform=trainTransform)
    trainset.train_data = trainset.train_data[:trn_size]
    trainset.train_labels = trainset.train_labels[:trn_size]
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=bs, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='data/', train=False, download=True, transform=testTransform)
    testset.test_data = testset.test_data[:tst_size]
    testset.test_labels = testset.test_labels[:tst_size]
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=bs, shuffle=False, num_workers=2)

    return trainloader, testloader
