import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class IMDbDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

class BertClassifier(nn.Module):
    def __init__(self, model_name="prajjwal1/bert-tiny", num_classes=2, dropout_rate=0.1, from_scratch=False):
        super(BertClassifier, self).__init__()
        self.model_name = model_name
        self.from_scratch = from_scratch
        if from_scratch:
            config = BertConfig.from_pretrained(model_name)
            self.encoder = BertModel(config)
        else:
            self.encoder = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(cls_rep)
        logits = self.classifier(pooled)
        return logits

def set_encoder_frozen(model, frozen=True):
    for param in model.encoder.parameters():
        param.requires_grad = not frozen

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_confs = []
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_confs.extend(confs.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
    avg_loss = total_loss / max(len(dataloader), 1)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    conf_mat = confusion_matrix(all_labels, all_preds)
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": all_preds,
        "true_labels": all_labels,
        "confidences": all_confs,
        "probabilities": all_probs,
        "confusion_matrix": conf_mat
    }

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    sample_text = "The cats are playing."
    tokens = tokenizer(sample_text, return_tensors="pt")
    print("Token keys:", list(tokens.keys()))
    print("Input IDs:", tokens["input_ids"].tolist())
    print("Attention Mask:", tokens["attention_mask"].tolist())
    model = BertClassifier(model_name="prajjwal1/bert-tiny", num_classes=2)
    model.to(device)
    print("Total parameters:", count_total_parameters(model))
    print("Trainable parameters:", count_trainable_parameters(model))
    set_encoder_frozen(model, frozen=True)
    print("Trainable parameters after freezing encoder:", count_trainable_parameters(model))
    logits = model(tokens["input_ids"].to(device), tokens["attention_mask"].to(device))
    print("Logits shape:", logits.shape)
    probs = torch.softmax(logits, dim=-1)
    print("Probabilities:", probs.tolist())

