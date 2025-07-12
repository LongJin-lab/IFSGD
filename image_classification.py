import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import utils
from torch.optim.lr_scheduler import CosineAnnealingLR


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="cifar100")
    parser.add_argument("--model_name", type=str, default="resnet18")

    args = parser.parse_args()

    utils.seed_torch(42)
    epoch_num = 200
    lr = 0.1

    dataset_name = args.dataset_name
    trainloader, testloader = utils.load_data(dataset_name)

    device = torch.device("cuda:0")

    model_name = args.model_name
    model = utils.select_model(model_name, dataset_name, device)

    optimizer_name = "SGDM"
    optimizer = utils.select_optimizer(optimizer_name, model, lr)

    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

    scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=lr/1000, verbose=True)

    print(f"Dataset:{dataset_name} Model: {model_name} Optimizer:{optimizer_name}")

    training_times = []
    losses = []
    accuracies = []
    for epoch in range(epoch_num):
        start_time = time.time()
        loss = train_one_epoch(epoch=epoch, model=model, training_loader=trainloader, optimizer=optimizer, criterion=criterion, device=device)
        training_time = time.time() - start_time
        training_times.append(training_time)
        accuracy = utils.test(model, testloader, device=device)
        losses.append(loss)
        accuracies.append(accuracy)
        scheduler.step()
        print(f"Epoch {epoch + 1}: Loss: {loss}, Accuracy:{accuracy} LR:{ scheduler.get_last_lr()[0]} Training Time:{training_time}")


def train_one_epoch(training_loader, model, criterion, epoch=0, optimizer=None, device=None):
    scaler = torch.amp.GradScaler()
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    model.train()
    running_loss = 0.0

    for i, (inputs, target) in enumerate(training_loader):
        optimizer.zero_grad()

        input_var = inputs.to(device)
        target_var = target.to(device)

        with autocast:
            output = model(input_var)
            loss = criterion(output, target_var)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
    return running_loss / len(training_loader)


if __name__ == '__main__':
    main()