import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pickle
import torch.optim as optim
from sklearn.model_selection import train_test_split

#Loading the training data
with open('train.pkl', 'rb') as file:
    train_data = pickle.load(file)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False



# Step 1:Dataset Creation
##############################################
class MNISTDataset(Dataset):
    def __init__(self, data):
        self.data = train_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image.reshape(784, ), label
#Implement your logic here
##############################################

# Step 2: MLP creation
##############################################
class MLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_sz1=512, hidden_sz2=256, output_sz=10) -> None:
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_sz1)
        self.act1 = nn.GELU()
        self.hidden2 = nn.Linear(hidden_sz1, hidden_sz2)
        self.act2 = nn.GELU()
        self.hidden3 = nn.Linear(hidden_sz2, output_sz)
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.hidden3(x)
        return x
#Implement your logic here 
# (make sure the model returns logits only not the prediction)
##############################################

# Not to be changed
epochs = 50

# Step 3: Loading the data to Dataloader and hyperparammeters selection
##############################################
model = MLP()

#Implement your logic here
##############################################
model.to(device)
train_data, val_data = train_test_split(train_data, test_size = 0.2, random_state=42)
train_dataset = MNISTDataset(train_data)
val_dataset = MNISTDataset(val_data)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
# Step 4: Training the model and saving it.
##############################################


#Implement your logic here
##############################################
loss_fn = torch.nn.CrossEntropyLoss()
lr=0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5 )
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
best_model = None
best_acc = 0.0
losses = []
for epoch in range(epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        x = X_batch.to(device)
        y = y_batch.to(device)
        optimizer.zero_grad()
        y_pred = torch.softmax(model(x), dim=1)
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
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")
    print(f"Training Accuracy: {100 * correct / total:.2f}%")
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        correct = 0
        total = 0
        for x_val, y_val in val_loader:
            val_x = x_val.to(device)
            val_y = y_val.to(device)
            val_pred = torch.softmax(model(val_x), dim=1)
            val_loss += loss_fn(val_pred, val_y).item()
            
            _, predicted = torch.max(val_pred, 1)
            total += val_y.size(0)
            correct += (predicted == val_y).sum().item()
        
        acc = 100 * correct / total
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
        print(f"Validation Accuracy: {100 * correct / total:.2f}%")
        if(acc>best_acc):
            best_acc = acc
            best_model = model


#Inference (Don't change the code)
def evaluate(model,test_data_path):
    with open(test_data_path, 'rb') as file:
        test_data = pickle.load(file)
    test_dataset = MNISTDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) #MOdel returns logits
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100*correct/total #returns accuracy in percentage
    return accuracy

accuracy = evaluate(best_model,'test.pkl')
print(f"accuracy: {accuracy}%")
