### This code is only provided for referrence purposes, please do not make any changes to this

import argparse
import pandas as pd

from dataset import WineDataset
from model import WineModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--test_csv_path')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    args = parser.parse_args()

    # device = torch.device(f"cuda:{args.gpu}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WineModel().to(device)
    model.load_state_dict(torch.load(args.model_path, weights_only = True))
    model = model.to(device)

    test_data = pd.read_csv(args.test_csv_path)
    test_dataset = WineDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle=False)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    return accuracy



if __name__ == '__main__':

    print('Accuracy:', main())

