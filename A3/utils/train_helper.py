import os
import glob
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

def train_model(model, train_loader, val_loader, device, epochs=5, lr=1e-3, save_model_path="best_model.pth"):
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
            torch.save(model.state_dict(), save_model_path)
    
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

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return all_preds, all_labels
