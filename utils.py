import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os, random
import numpy as np

from IFSGD import IFSGD
from SGDM_baseline import SGDM_baseline

from torchvision.models import resnet18 as imagenet_resnet18
from torchvision.models import resnet50 as imagenet_resnet50

from torch.utils.data import DataLoader, random_split

from model import resnet, PRN

from model import repvgg


def load_data(dataset_name):
    trainloader = ""
    testloader = ""
    if dataset_name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    elif dataset_name == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    elif dataset_name == "cifar100_val":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        train_size = int(0.8 * len(dataset))  # 80% for training
        val_size = len(dataset) - train_size  # 20% for validation
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
        testloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    elif dataset_name == "imagenet1k":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = datasets.ImageFolder(
            root='/disk1/imagenet1k/train',
            transform=train_transforms
        )

        test_dataset = datasets.ImageFolder(
            root='/disk1/imagenet1k/val',
            transform=test_transforms
        )

        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=True, num_workers=16, pin_memory=True
        )

        testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=128, shuffle=False, num_workers=16, pin_memory=True
        )
    return trainloader, testloader


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def select_model(model_name, dataset, device):
    if dataset == "imagenet1k":
        if model_name == "resnet18":
            return imagenet_resnet18(num_classes=1000, weights=None).to(device)
        if model_name == "resnet50":
            return imagenet_resnet50(num_classes=1000, weights=None).to(device)
    if model_name == "resnet18":
        return resnet.ResNet18(num_classes=100).to(device)
    elif model_name == "resnet50":
        return resnet.ResNet50(num_classes=100).to(device)
    elif model_name == "PyramidNet110":
        return PRN.PyramidNet(dataset="cifar100",depth=101, alpha=64, num_classes=100).to(device)
    elif model_name == "RepVGG_A1":
        return repvgg.repvgg_a1(num_classes=100).to(device)


def select_optimizer(optimizer_name, model, lr):
    if optimizer_name == "SGDM":
        return SGDM_baseline(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == "IFSGD":
        return IFSGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


def test(net, testloader, device="cuda"):
    net.eval()
    correct = 0
    total = 0
    autocast = torch.cuda.amp.autocast
    with torch.no_grad():
        for data, label in testloader:
            data, label = data.to(device), label.to(device)
            with autocast():
                output = net(data)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
    return 100 * correct / total