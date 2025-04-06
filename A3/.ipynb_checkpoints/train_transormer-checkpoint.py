import os, shutil, random
import glob
import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm 
try:
    import torchtext
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
    torchtext_available = True
except ImportError:
    torchtext_available = False
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models.dataset import *
from models.encoder_arch import *
from models.init_seed import initialize
from preprocessing import make_train_data, make_test_data, train_val_split

data_dir="Datasets"
make_train_data(dir_name=data_dir)
make_test_data(dir_name=data_dir)
train_val_split(dir_name=data_dir ,split_ratio=0.2)

initialize(seed=0)
    
max_length = 600       
max_tokens = 20000     
batch_size = 32
embed_dim = 256
num_heads = 2
dense_dim = 32
num_classes = 5 

def train_model(model, train_loader, val_loader, device, epochs=5, lr=1e-3):
    train_accuracy_array = []
    val_accuracy_array = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    model.to(device)
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False)
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total += inputs.size(0)
            
            train_bar.set_postfix(loss=loss.item(), acc=running_corrects / total if total > 0 else 0)
        
        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        train_accuracy_array.append(epoch_acc)
        
        model.eval()
        val_corrects = 0
        val_total = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation", leave=False)
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels).item()
                val_total += inputs.size(0)
                val_bar.set_postfix(val_acc=val_corrects / val_total if val_total > 0 else 0)
        
        val_acc = val_corrects / val_total
        val_accuracy_array.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
    
    print("Training complete. Best validation accuracy: {:.4f}".format(best_val_acc))
    return train_accuracy_array, val_accuracy_array

def test_model(model, test_loader, device):
    model.to(device)
    model.eval()
    test_corrects = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels).item()
            test_total += inputs.size(0)
    test_acc = test_corrects / test_total
    print(f"Test Accuracy: {test_acc:.4f}")
    
use_torchtext = True
if use_torchtext and torchtext_available:
    print("Using torchtext tokenizer (basic_english)")
    tokenizer = get_tokenizer("basic_english")
else:
    print("Using basic Python tokenizer (split on whitespace)")
    tokenizer = lambda x: x.lower().split()

train_dir = os.path.join("Datasets", "train")
vocab = build_vocab(train_dir, tokenizer, use_torchtext=use_torchtext, max_tokens=max_tokens)
vocab_size = len(vocab)
print("Vocabulary size:", vocab_size)

train_dataset = TextClassificationDataset(train_dir, vocab, tokenizer, max_length)
val_dir = os.path.join("Datasets", "val")
val_dataset = TextClassificationDataset(val_dir, vocab, tokenizer, max_length)
test_dir = os.path.join("Datasets", "test")
test_dataset = TextClassificationDataset(test_dir, vocab, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TextClassificationModel(sequence_length=max_length,
                                vocab_size=vocab_size,
                                embed_dim=embed_dim,
                                num_heads=num_heads,
                                dense_dim=dense_dim,
                                num_classes=num_classes)
nepochs=20
train_array, val_array = train_model(model, train_loader, val_loader, device, epochs=nepochs)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
test_model(model, test_loader, device)


