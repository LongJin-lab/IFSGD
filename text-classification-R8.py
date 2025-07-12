import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    AlbertTokenizer, AlbertForSequenceClassification,
    ElectraTokenizer, ElectraForSequenceClassification,
)
import time
from SGDM_baseline import SGDM_baseline
from IFSGD import IFSGD

from utils import seed_torch


MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(text_file, label_file):
    with open(text_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f]

    labels = []
    split_info = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            label = '_'.join(parts[2:])
            split_info.append(parts[1])
            labels.append(label)


    df = pd.DataFrame({
        'text': texts,
        'label': labels,
        'split': split_info
    })

    unique_labels = df['label'].unique().tolist()
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    return train_df, test_df, label2id, id2label


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(epoch=epoch)
        optimizer.zero_grad()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return accuracy_score(true_labels, predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="albert-base")
    args = parser.parse_args()

    PRETRAINED_MODEL = args.model_name
    print("Model: ", PRETRAINED_MODEL)

    seed = 42
    seed_torch(seed)
    print("Seed: ", seed)

    train_df, test_df, label2id, id2label = load_data('./R8/R8.txt', './R8/R8_label.txt')

    tokenizer = None
    model = None
    if PRETRAINED_MODEL == "electra-small":
        tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")

        model = ElectraForSequenceClassification.from_pretrained(
            "google/electra-small-discriminator",
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        ).to(device)
    if PRETRAINED_MODEL == "albert-base":
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

        model = AlbertForSequenceClassification.from_pretrained(
            "albert-base-v2",
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        ).to(device)

    train_dataset = TextDataset(
        texts=train_df.text.values,
        labels=train_df.label.map(label2id).values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    test_dataset = TextDataset(
        texts=test_df.text.values,
        labels=test_df.label.map(label2id).values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = None

    optimizer_name = "IFSGD"

    print("Optimizer: ", optimizer_name)

    if optimizer_name == "SGDM":
        optimizer = SGDM_baseline(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4, momentum=0.9)
    elif optimizer_name == "IFSGD":
        optimizer = IFSGD(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4, momentum=0.9)

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE/1000, verbose=True)

    best_accuracy = 0

    training_times = []
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        training_time = time.time() - start_time
        print(f'Train loss: {train_loss:.4f}')
        print(f'Training time: {training_time:.4f}')
        training_times.append(training_time)
        scheduler.step()

        accuracy = evaluate(model, test_loader, device)
        print(f'Test Accuracy: {accuracy:.4f}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy

    print(f'\nBest Accuracy: {best_accuracy}')


if __name__ == '__main__':
    main()