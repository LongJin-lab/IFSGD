import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AlbertTokenizer, AlbertForSequenceClassification,
    ElectraTokenizer, ElectraForSequenceClassification,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import seed_torch
from SGDM_baseline import SGDM_baseline
from IFSGD import IFSGD
import time

MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2
RANDOM_SEED = 1234
DATA_PATH = "./MR"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(data_path):
    texts = []
    labels = []

    neg_dir = os.path.join(data_path, 'neg')
    for filename in os.listdir(neg_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(0)

    pos_dir = os.path.join(data_path, 'pos')
    for filename in os.listdir(pos_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, 
        test_size=TEST_SIZE, 
        stratify=labels,
        random_state=RANDOM_SEED
    )
    
    return train_texts, test_texts, train_labels, test_labels


class MRDataset(Dataset):
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


def train_epoch(model, dataloader, optimizer, scheduler, epoch):
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


def evaluate(model, dataloader):
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    with torch.no_grad():
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
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="albert-base")
    args = parser.parse_args()

    seed_torch(RANDOM_SEED)
    print("Seed: ", RANDOM_SEED)

    train_texts, test_texts, train_labels, test_labels = load_data(DATA_PATH)

    PRETRAINED_MODEL = args.model_name
    print("Model: ", PRETRAINED_MODEL)

    model = None
    tokenizer = None
    if PRETRAINED_MODEL == "electra-small":
        tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")

        model = ElectraForSequenceClassification.from_pretrained(
            "google/electra-small-discriminator",
            num_labels=2,
        ).to(device)
    if PRETRAINED_MODEL == "albert-base":
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

        model = AlbertForSequenceClassification.from_pretrained(
            "albert-base-v2",
            num_labels=2,
        ).to(device)

    train_dataset = MRDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    test_dataset = MRDataset(test_texts, test_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer_name = "IFSGD"
    print("Optimizer:", optimizer_name)

    optimizer = None
    if optimizer_name == "SGDM":
        optimizer = SGDM_baseline(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == "IFSGD":
        optimizer = IFSGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE/1000, verbose=True)

    training_times = []
    best_accuracy = 0
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch)
        training_time = time.time() - start_time
        test_loss, test_acc = evaluate(model, test_loader)
        training_times.append(training_time)
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')
        print(f'Test Accuracy: {test_acc:.4f}')
        print(f'Training Time: {training_time:.4f}')
        if test_acc > best_accuracy:
            best_accuracy = test_acc
        
    print(f'\nBest Accuracy: {best_accuracy:.4f}')


if __name__ == '__main__':
    main()