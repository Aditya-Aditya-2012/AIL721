"""has functions to create the vocab and embeddings
by training a skipgram model from scratch"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import re
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomTestDataset(Dataset):
    def __init__(self, path_to_data):
        self.df = pd.read_csv(path_to_data)
        self.df.fillna('', inplace=True)
        categories = self.df["Label - (business, tech, politics, sport, entertainment)"].unique()
        self.category2idx = {cat: idx for idx, cat in enumerate(categories)}
    
    def __getitem__(self, index):
        text = self.df.loc[index]["Text"]
        class_index = self.category2idx[self.df.loc[index]["Label - (business, tech, politics, sport, entertainment)"]]
        return class_index, text
    def __len__(self):
        return len(self.df)


class CustomDataset(Dataset):
    def __init__(self, path_to_dataset):
        self.df = pd.read_csv(path_to_dataset)
        self.df.fillna('', inplace=True)
        categories = self.df["Category"].unique()
        self.category2idx = {cat: idx for idx, cat in enumerate(categories)}

    def __getitem__(self, index):
        text = self.df.loc[index]["Text"]
        class_index = self.category2idx[self.df.loc[index]["Category"]]
        return class_index, text
    
    def __len__(self):
        return len(self.df)


def _clean_data(sent):
    sent = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sent)
    sent = re.sub(r"\'s", " \'s", sent)
    sent = re.sub(r"\'ve", " \'ve", sent)
    sent = re.sub(r"n\'t", " n\'t", sent)
    sent = re.sub(r"\'re", " \'re", sent)
    sent = re.sub(r"\'d", " \'d", sent)
    sent = re.sub(r"\'ll", " \'ll", sent)
    sent = re.sub(r",", " , ", sent)
    sent = re.sub(r"!", " ! ", sent)
    sent = re.sub(r"\(", " \( ", sent)
    sent = re.sub(r"\)", " \) ", sent)
    sent = re.sub(r"\?", " \? ", sent)
    sent = re.sub(r"\s{2,}", " ", sent)
    return sent

def tokenize(text):
    text = text.lower()
    text = _clean_data(text)
    tokens = text.split()
    tokens = ["<sos>"] + tokens + ["<eos>"]
    return tokens

def create_vocab(path, min_freq=1, specials=['<pad>', '<sos>', '<eos>', '<unk>']):
    df = pd.read_csv(path)
    df['tokens'] = df['Text'].apply(tokenize)
    all_tokens = [token for tokens in df['tokens'] for token in tokens]
    word_counts = Counter(all_tokens)
    vocab = {}
    for token in specials:
        vocab[token] = len(vocab)
    for token, count in word_counts.items():
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab, word_counts, df['tokens']

def sentence_to_indices(sent, vocab, max_seq_len):
    tokens=tokenize(sent)
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    if len(indices) > max_seq_len:
        indices = indices[:max_seq_len]
    else:
        indices = indices + [vocab['<pad>']] * (max_seq_len - len(indices))
    return indices

def sentences_to_indices(sentences, vocab, max_seq_len):
    """
    Processes a batch (list) of sentences and returns a tensor of shape
    (batch_size, max_seq_len).
    """
    all_indices = [sentence_to_indices(sentence, vocab, max_seq_len) for sentence in sentences]
    return torch.tensor(all_indices, dtype=torch.long).to(device)

class TokenDrop(nn.Module):
    def __init__(self, prob=0.1, pad_token=0, num_special=4):
        self.prob = prob
        self.num_special = num_special
        self.pad_token = pad_token

    def __call__(self, sample):
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.long).to(device)
        mask = torch.bernoulli(self.prob * torch.ones_like(sample)).long()
        can_drop = (sample >= self.num_special).long()
        mask = mask * can_drop
        replace_with = (self.pad_token * torch.ones_like(sample)).long()
        sample_out = (1 - mask) * sample + mask * replace_with
        
        return sample_out