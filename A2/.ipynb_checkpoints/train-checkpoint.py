import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from models.sam import SAM
from models.resnet import resnet
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.initialize import initialize

initialize(seed=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

def create_dataset(data_dir, batch_size=32, num_workers=1, val_split=0.2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    total_size = len(full_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return full_dataset, train_loader, val_loader

def smooth_crossentropy(pred, target, smoothing=0.1):
    confidence = 1.0 - smoothing
    logprobs = F.log_softmax(pred, dim=1)
    nll_loss = F.nll_loss(logprobs, target, reduction='none')
    loss = confidence * nll_loss + smoothing * (-logprobs.mean(dim=1))
    return loss

def evaluate_model(model, data_loader):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            outputs = model(inputs)
            loss = smooth_crossentropy(outputs, labels).mean()
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, save_weights_path, epoch_cap):
    model = model.float().to(device)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, adaptive=True,
                    momentum=0.9, lr=0.05, weight_decay=0.0005)
    scheduler = StepLR(optimizer, 0.05, epoch_cap)

    train_accuracies = []
    val_accuracies = []
    training_duration = 3570
    start_time = time.time()

    best_val_accuracy = 0

    epoch = 0
    while time.time() - start_time < training_duration and epoch < epoch_cap:
        epoch += 1
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)

            # First forward-backward step
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, labels)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # Second forward-backward step
            disable_running_stats(model)
            smooth_crossentropy(model(inputs), labels).mean().backward()
            optimizer.second_step(zero_grad=True)

            running_loss += loss.mean().item()
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler(epoch)

        val_loss, val_accuracy = evaluate_model(model, val_loader)
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        epoch_loss = running_loss / len(train_loader)
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch}, Time: {elapsed_time:.2f}s, Train Loss: {epoch_loss:.4f}, " 
              f"Val Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, " 
              f"Val Accuracy: {val_accuracy:.2f}%")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_weights_path)
            print(f"Saved new best model with validation accuracy: {best_val_accuracy:.2f}%")

    return model, best_val_accuracy

def main():
    if len(sys.argv) != 3:
        print("Usage: python train.py <train data dir> <weight checkpoint dir>")
        sys.exit(1)
    
    train_dir = sys.argv[1]
    weight_ckpt_dir = sys.argv[2]
    os.makedirs(weight_ckpt_dir, exist_ok=True)
    save_weights_path = os.path.join(weight_ckpt_dir, "resnet_model.pth")

    _, train_loader, val_loader = create_dataset(train_dir, batch_size=32, num_workers=1, val_split=0.2)
    
    n = 2
    num_classes = 100
    model = resnet(n, num_classes)
    
    epoch_cap = 100
    model, best_val_accuracy = train_model(model, train_loader, val_loader, save_weights_path, epoch_cap)

    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.2f}%")


if __name__ == '__main__':
    main()
