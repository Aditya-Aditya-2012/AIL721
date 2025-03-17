import pandas as pd
from dataset import WineDataset, create_dataset
from model import WineModel
import argparse
import torch
from torch.utils.data import DataLoader

seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    train_csv_path = "./Train.csv"
    train_dataset, val_dataset=create_dataset(train_csv_path, 0.2)

    model = WineModel().to(device)

    n_epochs = min(int(args.n_epochs), 200)

    #### implement the model training code here
    train_loader = DataLoader(dataset=train_dataset, batch_size=32)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
    loss_fn = torch.nn.CrossEntropyLoss()
    lr=0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5 )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.5)
    best_model = model.state_dict()
    best_acc = 0.0
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            x = X_batch.to(device)
            y = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        scheduler.step()
        average_loss = epoch_loss / len(train_loader)  # Calculate average loss for the epoch
        losses.append(average_loss)
        # Print epoch number and loss
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {average_loss:.4f}")
        print(f"Training Accuracy: {100 * correct / total:.2f}%")
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for x_val, y_val in val_loader:
                val_x = x_val.to(device)
                val_y = y_val.to(device)
                val_pred = model(val_x)
                val_loss += loss_fn(val_pred, val_y).item()
                
                _, predicted = torch.max(val_pred, 1)
                total += val_y.size(0)
                correct += (predicted == val_y).sum().item()
            
            acc = 100 * correct / total
            print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
            print(f"Validation Accuracy: {100 * correct / total:.2f}%")
            if(acc>best_acc):
                best_acc = acc
                best_model = model.state_dict()

    #### please donot hard code the number of epochs
    #### number of epochs should be passed as an argument and should be less than 200

    print(f'best accuracy: {best_acc}')
    model_path = "./model_weights.pth"
    torch.save(best_model, model_path)