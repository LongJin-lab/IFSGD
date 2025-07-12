import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from torch.cuda.amp import autocast, GradScaler
import utils
from SGDM_baseline import SGDM_baseline
from IFSGD import IFSGD
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
from torch.optim.lr_scheduler import CosineAnnealingLR


class Adapter(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()

        self.down_proj = nn.Linear(dim, dim // reduction)

        self.up_proj = nn.Linear(dim // reduction, dim)
        self.activation = nn.ReLU(inplace=True)

        nn.init.kaiming_normal_(self.down_proj.weight, nonlinearity='relu')

        nn.init.normal_(self.up_proj.weight, mean=0, std=1)
        nn.init.normal_(self.down_proj.bias, mean=0, std=1)
        nn.init.normal_(self.up_proj.bias, mean=0, std=1)

    def forward(self, x):
        identity = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return identity + x


class CustomCLIP(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(clip_model_name)
        for param in self.clip.parameters():
            param.requires_grad = False

        self.projection_dim = self.clip.projection_dim

        # use Adapter as the projector
        self.image_proj = Adapter(self.projection_dim)
        self.text_proj = Adapter(self.projection_dim)

        # use MLP as the projector
        # self.image_proj = nn.Sequential(
        #     nn.Linear(self.projection_dim, self.projection_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.projection_dim, self.projection_dim)
        # )
        # self.text_proj = nn.Sequential(
        #     nn.Linear(self.projection_dim, self.projection_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.projection_dim, self.projection_dim)
        # )

        self.class_names = CIFAR100(root='./data', train=True, download=True).classes
        self._prepare_text_features(clip_model_name)

    def _prepare_text_features(self, model_name):
        processor = CLIPProcessor.from_pretrained(model_name)
        text_inputs = [f"a photo of a {name}" for name in self.class_names]
        inputs = processor(text=text_inputs, return_tensors="pt", padding=True)

        with torch.no_grad():
            text_features = self.clip.get_text_features(**inputs)

        self.register_buffer("text_features", text_features)

    def forward(self, pixel_values):
        with torch.no_grad():
            image_features = self.clip.get_image_features(pixel_values=pixel_values)

        proj_image = self.image_proj(image_features)
        proj_text = self.text_proj(self.text_features)

        proj_image = F.normalize(proj_image, p=2, dim=-1)
        proj_text = F.normalize(proj_text, p=2, dim=-1)
        logits = proj_image @ proj_text.T
        logits *= self.clip.logit_scale.exp()

        return logits


if __name__ == '__main__':
    utils.seed_torch(1234)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    num_epochs = 100
    lr = 1e-1

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

    train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = CustomCLIP().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_name = "IFSGD"

    optimizer = None
    if optimizer_name == "SGDM":
        optimizer = SGDM_baseline([
            {'params': model.image_proj.parameters()},
            {'params': model.text_proj.parameters()}
        ], lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == "IFSGD":
        optimizer = IFSGD([
            {'params': model.image_proj.parameters()},
            {'params': model.text_proj.parameters()}
        ], lr=lr, momentum=0.9, weight_decay=5e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr / 1000, verbose=True)

    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        training_times = []

        start_time = time.time()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer, epoch=epoch)
            scaler.update()

            train_loss += loss.item() * images.size(0)
        training_time = time.time() - start_time
        training_times.append(training_time)

        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch + 1}/{num_epochs}] | "
              f"Train Loss: {train_loss / len(train_dataset):.4f} | "
              f"Test Acc: {100 * correct / total:.2f}% | "
              f"Training Time: {training_time}")
