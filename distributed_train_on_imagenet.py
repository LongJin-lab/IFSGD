import os
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from multiprocessing import cpu_count
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision.models import resnet18 as ResNet18
from torchvision.models import resnet50 as ResNet50
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from SGDM_baseline import SGDM_baseline
from IFSGD import IFSGD

from torch.optim.lr_scheduler import CosineAnnealingLR

import random
import numpy as np

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'  
    os.environ['MASTER_PORT'] = '29500'      
    dist.init_process_group(
        backend='nccl',        
        rank=rank,             
        world_size=world_size  
    )
    torch.cuda.set_device(rank)  

def load_dataset(dataset_name):
    traindataset = None
    testdataset = None
    if dataset_name == "imagenet1k":

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
        
        # 读取ImageNet训练集
        traindataset = datasets.ImageFolder(
            root='/imagenet/train',  # replace this with your ImageNet path
            transform=train_transforms
        )

        # 读取ImageNet验证集
        testdataset = datasets.ImageFolder(
            root='/imagenet/val',  # replace this with your ImageNet path
            transform=test_transforms
        )
    elif dataset_name == "cifar100":
        # 数据预处理
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

        traindataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

        testdataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    return traindataset, testdataset

world_size = 2
dataset_name = "imagenet1k"
optimizer_name = "IFSGD"
model_name = "resnet50"
lr = 0.1
epoch_num = 100

def train_and_test(rank, world_size):
    setup(rank, world_size)

    train_dataset, test_dataset = load_dataset(dataset_name)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=128, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=128, num_workers=8, pin_memory=True)

    if model_name == "resnet18":
        model = ResNet18(num_classes=1000)
    elif model_name == "resnet50":
        model = ResNet50(num_classes=1000)
    model.to(rank)

    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss(reduction='mean').to(rank)

    optimizer = None

    if optimizer_name == "SGDM":
        optimizer = SGDM_baseline(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == "IFSGD":
        optimizer = IFSGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # scheduler = StepLR(optimizer=optimizer, learning_rate=lr, total_epochs=epoch_num)

    scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=lr/1000, verbose=True)
  
    scaler = torch.amp.GradScaler("cuda") 

    accuracies = []
    losses = []

    for epoch in range(epoch_num):
        # Training phase
        ddp_model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(rank), targets.to(rank)
            
            # mix precision
            with torch.amp.autocast("cuda"):  
                outputs = ddp_model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()  
            scaler.step(optimizer, epoch=epoch)  
            scaler.update()  

            epoch_loss += loss.item() / len(train_loader)
        training_time = time.time() - start_time

        total_loss = torch.tensor(epoch_loss, device=rank)
        dist.reduce(total_loss, dst=0, op=dist.ReduceOp.SUM)  # reduce to rank 0
        if rank == 0:
            avg_loss = total_loss.item() / world_size
            losses.append(avg_loss)

        # Testing phase
        ddp_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(rank), targets.to(rank)
                
                # mix precision
                with torch.cuda.amp.autocast():  
                    outputs = ddp_model(inputs)
                    _, predicted = torch.max(outputs, 1)
                
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        correct_tensor = torch.tensor(correct, device=rank)
        total_tensor = torch.tensor(total, device=rank)
        dist.reduce(correct_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_tensor, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            accuracy = correct_tensor.item() / total_tensor.item()
            accuracies.append(accuracy)
        scheduler.step()  

        if rank == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f} Accuracy: {accuracy:.4f} Learning Rate: {scheduler.get_last_lr()[0]} Training Time: {training_time}")
    if rank == 0:
        with open(f'{optimizer_name}_accuracy_{epoch_num}_epochs_lr={lr}_{model_name}_{dataset_name}.pkl', 'wb') as file:
            pickle.dump(accuracies, file)
        
        with open(f'{optimizer_name}_loss_{epoch_num}_epochs_lr={lr}_{model_name}_{dataset_name}.pkl', 'wb') as file:
            pickle.dump(losses, file)

    dist.destroy_process_group()

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    seed_torch(42)

    for i in range(torch.cuda.device_count()):
        print(f"gpu{i}: {torch.cuda.get_device_name(i)}")

    print(f"Dataset:{dataset_name} Model: {model_name} Optimizer:{optimizer_name}")

    mp.spawn(train_and_test, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()

