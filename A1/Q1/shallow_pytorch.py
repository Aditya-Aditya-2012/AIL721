# ================================ Imports ================================ #
import matplotlib.pyplot as plt
import torch
import numpy as np
import logging
plt.style.use('seaborn-v0_8')
from torch import nn

# =============================== Variables ================================== #
torch.manual_seed(100) # Do not change the seed
np.random.seed(100) # Do not change the seed
torch.set_default_dtype(torch.float64)
logging.basicConfig(filename="avg-error-pytorch.log", filemode='w', format='%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
EPOCH = 10000

# ============================================================================ #


class shallow_network(nn.Module):

    def __init__(self, ip_size, hidden_size, op_size) -> None:
        super().__init__()
        # ========== Write code for creating neural network model
        self.hidden1 = nn.Linear(ip_size, hidden_size)
        self.act1 = nn.Sigmoid()
        self.hidden2 = nn.Linear(hidden_size, op_size)
        self.act_output = nn.Sigmoid()

        # raise NotImplementedError

    def forward(self, x):
        # ========== Implement forward pass of pytorch
        x = self.act1(self.hidden1(x))
        x = self.act_output(self.hidden2(x))
        return x


def plot_loss(train_loss):
    '''
    :param train_loss: list of training losses
    :return: saves a plot of the training losses vs epochs with file name loss-pytorch.png on current working directory
    '''
    # ========== Please do not edit this function

    plt.xlabel("Training Epoch")
    plt.ylabel("Training Loss")
    plt.plot(train_loss)
    plt.savefig("./loss-pytorch.png")


def main():

    # ========= Input Data
    X = [[1, 1, 1], [1, 0, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0]]

    # ========= Ouput Labels
    Y = [[1], [0], [0], [0], [0], [0], [0], [0]]
    X = np.array(X)
    X = torch.tensor(X.reshape(8, 3), dtype=torch.float64)
    Y = torch.tensor(Y, dtype=torch.float64)


    # =========== Write code to build neural net model
    input_size = 3
    num_hidden_units = 1
    output_size = 1
    model = shallow_network(input_size, num_hidden_units, output_size)

    # =========== Write code for training the model
    train_loss = [] # Use this list to store training loss per epoch
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1) 
    for i in range(EPOCH): # Total training epoch is 10000
        '''
        Write code for training loop here.
        '''
        y_pred = model(X)
        loss = loss_fn(y_pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item()) # Log the training loss for the epoch


    # =========== Plot Epoch Losses
    plot_loss(train_loss) # Do not change

    # =========== Predict
    X = torch.tensor(X).double()
    Y = torch.tensor(Y).double()
    error = []
    logger.info("===================")
    logger.info("   X       Y   Y' ")
    logger.info("===================")
    for i in range(Y.shape[0]):
        tstr = ""
        x = X[i]
        y_target = Y[i]
        y_pred = model(x)
        # loss = # use the same loss function used during training to get the error
        loss = loss_fn(y_pred, y_target)
        error.append(loss.item())
        x = x.data.numpy()
        y_target = int(y_target.item())
        y_pred = round(y_pred.item(), 1)
        tstr += str(x) + " "+ str(y_target)+" "+str(y_pred)
        logger.info(tstr)
    logger.info("Average Error: " + str(round(np.mean(error), 5)))




# =============================================================================== #

if __name__ == '__main__':
    main()
