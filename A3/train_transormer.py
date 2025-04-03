import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from models.encoder_arch import SinusodialEmbedding, EncoderBlock, Transformer
from models.dataset import CustomDataset, tokenize, create_vocab, sentence_to_indices, sentences_to_indices, TokenDrop, CustomTestDataset

def makeplots(epoch, training_acc_logger, test_acc_logger, save_path_prefix="acc_plot"):
    plt.figure(figsize=(10, 8))
    epochs = range(1, len(training_acc_logger) + 1)
    plt.plot(epochs, training_acc_logger, label='Training Accuracy', marker='o')
    plt.plot(epochs, test_acc_logger, label='Testing Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.grid(True)
    # Save the plot with a file name indicating the current epoch.
    save_path = f"{save_path_prefix}.png"
    plt.savefig(save_path)
    plt.close()


device = torch.device(0 if torch.cuda.is_available() else 'cpu')
data_path = '/home/civil/btech/ce1210494/AIL721/A3/Datasets/TrainData.csv'
data_test_path = '/home/civil/btech/ce1210494/AIL721/A3/Datasets/TestLabels.csv'
nepochs=100
dataset_train = CustomDataset(data_path)
dataset_test = CustomTestDataset(data_test_path)
batch_sz=128
trainloader = DataLoader(dataset_train, batch_size=batch_sz, shuffle=True, drop_last=True)
testloader = DataLoader(dataset_test, batch_size=batch_sz, shuffle=True, drop_last=True)
vocab, word_counts, tokens_list = create_vocab(data_path)
# test_vocab, test_word_counts, test_tokens_list = create_vocab(data_test_path)
hidden_sz = 256
tf_classifier = Transformer(num_emb=len(vocab), output_sz=5, hidden_sz=hidden_sz,
                            num_layers=4, num_heads=8).to(device)

optimizer = optim.Adam(tf_classifier.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()
td = TokenDrop(prob=0.1)

training_loss_logger = []
test_loss_logger = []
training_acc_logger = []
test_acc_logger = []

pbar = trange(0, nepochs, leave=False, desc="Epoch")
train_acc = 0
test_acc = 0
for epoch in pbar:
    train_acc_count = 0
    test_acc_count = 0
    
    # Update the progress bar with current training and testing accuracy
    pbar.set_postfix_str('Accuracy: Train %.2f%%, Test %.2f%%' % (train_acc * 100, test_acc * 100))
    # pbar.set_postfix_str('Accuracy: Train %.2f%%' % (train_acc * 100))
    # Set the model to training mode
    tf_classifier.train()
    steps = 0
    
    # Loop over each batch in the training dataset
    for label, text in tqdm(trainloader, desc="Training", leave=False):
        bs = label.shape[0]
        # Transform the text to tokens and move to the GPU
        text_tokens = sentences_to_indices(text, vocab, 256)
        label = label.to(device)
        text_tokens = td(text_tokens)

        pred = tf_classifier(text_tokens)

        # Compute the loss using cross-entropy loss
        loss = loss_fn(pred, label)
        
        # Backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log the training loss
        training_loss_logger.append(loss.item())
        
        # Update training accuracy
        train_acc_count += (pred.argmax(1) == label).sum()
        steps += bs
    
    # Calculate average training accuracy
    train_acc = (train_acc_count / steps).item()
    print(f'training accuracy: {train_acc}')
    training_acc_logger.append(train_acc)
    
    # # Set the model to evaluation mode
    tf_classifier.eval()
    steps = 0
    
    # Loop over each batch in the testing dataset
    with torch.no_grad():
        for label, text in tqdm(testloader, desc="Testing", leave=False):
            bs = label.shape[0]
            
            # Transform the text to tokens and move to the GPU
            text_tokens = sentences_to_indices(text, vocab, 256)
            label = label.to(device)

            # Get the model predictions
            pred = tf_classifier(text_tokens)

            # Compute the loss using cross-entropy loss
            loss = loss_fn(pred, label)
            test_loss_logger.append(loss.item())

            # Update testing accuracy
            test_acc_count += (pred.argmax(1) == label).sum()
            steps += bs

        # Calculate average testing accuracy
        test_acc = (test_acc_count / steps).item()
        print(f'testing accuracy: {test_acc}')
        test_acc_logger.append(test_acc)
    makeplots(epoch, training_acc_logger, test_acc_logger, save_path_prefix="acc_plot")