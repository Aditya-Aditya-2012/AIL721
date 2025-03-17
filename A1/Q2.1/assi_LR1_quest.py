import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import pickle

import matplotlib.pyplot as plt
# Loading data
file_path = "train.dat"
columns = [
    "frequency", 
    "attack_angle", 
    "chord_length", 
    "free_stream_velocity", 
    "suction_side_displacement_thickness", 
    "scaled_sound_pressure"
]
data = pd.read_csv(file_path, sep="\t", header=None, names=columns)

# Step: 1 Features and target values
##############################################
X = data[["frequency", "attack_angle", "chord_length", "free_stream_velocity", "suction_side_displacement_thickness"]]
y = data["scaled_sound_pressure"]
##############################################


# Step: 2 Training-Validation split
##############################################
    # Implement your Logic
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
##############################################



# Step: 3 Normalizing features
##############################################
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
##############################################


# Converting to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)


# Step: 4 Defining the NN
##############################################
class LinearRegressionModel(nn.Module):
    # Implement your Logic
    def __init__(self, input_sz) -> None:
        super().__init__()
        self.hidden1 = nn.Linear(input_sz, 3)
        self.act1 = nn.Sigmoid()
        self.hidden2 = nn.Linear(3, 1)
    
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.hidden2(x)
        return x
##############################################
    

# Step: 5 Hyperparameters (learning_rate, batch_size, Loss_function, optimizer)
##############################################
input_dim = 5
model = LinearRegressionModel(input_dim)
    # Implement your Logic
learning_rate = 0.1
batch_size = 10
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
##############################################


# Not to be changed
epochs = 300

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)


# Step: 6 Training loop
################################################################
    # Implement your Logic
train_loss = []
val_loss =[]
for epoch in range(epochs):
    epoch_loss = 0
    batch_cnt = 0
    for i in range(0, len(X), batch_size):
        batch_cnt += 1
        Xbatch = X_train[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y_train[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        epoch_loss+=loss.mean().item()
    
    with torch.no_grad():
        y_pred_train = model(X_train)
        train_loss.append(loss_fn(y_pred_train, y_train)/len(X_train))
        y_pred_val = model(X_val)
        val_loss.append(loss_fn(y_pred_val, y_val)/len(X_val))
################################################################


# Step: 7 Train and Val Loss plot
##############################################
    # Implement your Logic
def plot_loss(loss_array, label):
    '''
    :param train_loss: list of training losses
    :return: saves a plot of the training losses vs epochs with file name loss-pytorch.png on current working directory
    '''
    # ========== Please do not edit this function

    plt.xlabel(f'{label} Epoch')
    plt.ylabel(f'{label} Loss')
    plt.plot(loss_array)
    plt.savefig(f'./{label}-loss-pytorch.png')

plot_loss(train_loss, 'Training')
plot_loss(val_loss, 'Validation')
##############################################


# Step: 8 Save the Model (Format: EntryNumber_model.pkl )
##############################################
    # Implement your Logic
torch.save(model.state_dict(), '2021CE10494_model.pkl')
##############################################








