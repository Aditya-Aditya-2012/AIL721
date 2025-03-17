# ================================ Imports ================================ #
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import logging

# =============================== Variables ================================== #
logging.basicConfig(filename="avg-error-numpy.log", filemode='w', format='%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
np.random.seed(100)
EPOCH = 10000

# ============================================================================ #

def sigmoid(z):
    '''
    Write your code for sigmoid implementation.
    '''
    return 1 / (1 + np.exp(-z))
    # raise NotImplementedError

def sigmoid_derivative(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

class shallow_network:

    def __init__(self, ip_size, hidden_size, op_size, lr=0.0):

        self.lr = lr
        self.w1 = self.init_w(ip_size, hidden_size)
        self.w2 = self.init_w(hidden_size, op_size)

    def init_w(self, x, y):
        '''
        Write your code for initializing the weight.
        '''
        return np.random.randn(x, y)
        # raise NotImplementedError

    def forward(self, x):
        '''
        Write your code for forward pass.
        '''
        self.z1 = np.dot(x.T, self.w1)
        self.a1 = sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.w2)
        self.a2 = sigmoid(self.z2)

        return self.a2
        # raise NotImplementedError

    def backward(self, x, y):

        '''
        Write your code for backward pass that computes the gradient and updates the weights.
        '''
        d2 = self.forward(x) - y
        d1 = np.multiply(self.w2.dot(d2.T).T, sigmoid_derivative(self.z1))  

        w1_gradient = x.dot(d1)  
        w2_gradient = self.a1.T.dot(d2)  

        self.w1 -= self.lr * w1_gradient  
        self.w2 -= self.lr * w2_gradient  
        # raise NotImplementedError

def loss(y_pred, y_targt):
    '''
    Write your code for mse loss.
    '''
    return np.mean(np.square(y_pred - y_targt))

    # raise NotImplementedError

def plot_loss(epoch_loss):

    '''
    :param train_loss: list of training losses
    :return: saves a plot of the training losses vs epochs with file name loss-numpy.png on current working directory
    '''
    # ========== Please do not edit this function
    plt.xlabel("Training Epoch")
    plt.ylabel("Training Loss")
    plt.plot(epoch_loss)
    plt.savefig("./loss-numpy.png")

def main():

    # ========= Input
    X = [[1, 1, 1, 1], [1, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 0, 1]]
    # ========= Ouput
    Y = [[1], [1], [0], [1], [1], [1], [1], [1]]
    X = np.array(X)
    X = X.reshape(8, 4, 1)
    Y = np.array(Y)
    # =========== Build Neural Net Model
    input_size = 4
    hidden_size = 2
    output_size = 1
    model = shallow_network(input_size, hidden_size, output_size, lr=0.5)

    # =========== Write code for training the model
    train_loss = [] # Use this list to store training loss per epoch
    for i in range(EPOCH): # Total training epoch is 10000
        epoch_loss = []
        for j in range(len(X)):
            x = X[j]
            y_target = Y[j]
            y_pred = model.forward(x)
            _loss = loss(y_pred, y_target)
            epoch_loss.append(_loss)
            model.backward(x, y_target)
        train_loss.append(sum(epoch_loss) / len(X))

    # =========== Plot Epoch Losses
    plot_loss(train_loss)

    # =========== Predict
    error = []
    logger.info("===================")
    logger.info("   X    Y  Y' ")
    logger.info("===================")
    for i in range(Y.shape[0]):
        tstr = ""
        x = X[i]
        y_target = Y[i]
        y_pred = model.forward(x)
        tstr += str(x) + " "+ str(y_target[0])+" "+str((np.round(y_pred[0]), 1))
        logger.info(tstr)
        error.append(loss(y_pred, y_target))
    logger.info("Average Error: " + str(round(np.mean(error), 5)))




# =============================================================================== #

if __name__ == '__main__':
    main()
