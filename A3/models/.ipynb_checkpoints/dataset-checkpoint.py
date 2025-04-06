import os
import glob
import random
import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

try:
    import torchtext
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
    torchtext_available = True
except ImportError:
    torchtext_available = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextClassificationDataset(Dataset):
    """
    A custom Dataset that reads text files from a directory structure.
    Expects subdirectories for each category containing .txt files.
    """
    def __init__(self, root_dir, vocab, tokenizer, max_length=600):
        self.samples = []
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2idx = {}
        self.idx2label = {}
        # Get categories sorted alphabetically
        categories = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for i, cat in enumerate(categories):
            self.label2idx[cat] = i
            self.idx2label[i] = cat
            cat_dir = os.path.join(root_dir, cat)
            files = glob.glob(os.path.join(cat_dir, "*.txt"))
            for f in files:
                with open(f, "r", encoding="utf-8") as fp:
                    text = fp.read().strip()
                self.samples.append((text, i))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        tokens = self.tokenizer(text)
        # Convert tokens to indices; if token not found, use index 0 (<unk>)
        indices = [self.vocab.get(token, 0) for token in tokens]
        # Pad or truncate the sequence to max_length
        if len(indices) < self.max_length:
            indices = indices + [0] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def build_vocab(data_dir, tokenizer, use_torchtext=True, max_tokens=20000):
    """
    Build a vocabulary from text files located in data_dir (assumed to be the training directory).
    """
    texts = []
    categories = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    for cat in categories:
        cat_dir = os.path.join(data_dir, cat)
        files = glob.glob(os.path.join(cat_dir, "*.txt"))
        for f in files:
            with open(f, "r", encoding="utf-8") as fp:
                text = fp.read().strip()
            texts.append(text)
    
    def yield_tokens(texts):
        for t in texts:
            yield tokenizer(t)
    
    if use_torchtext and torchtext_available:
        # Use torchtext's vocabulary builder
        vocab = build_vocab_from_iterator(yield_tokens(texts), specials=["<unk>"], max_tokens=max_tokens)
        vocab.set_default_index(vocab["<unk>"])
        # Convert torchtext Vocab to a regular dict mapping token -> index
        vocab_dict = {token: vocab[token] for token in vocab.get_itos()}
        return vocab_dict
    else:
        # Manually build vocabulary using collections.Counter
        counter = Counter()
        for t in texts:
            tokens = tokenizer(t)
            counter.update(tokens)
        most_common = counter.most_common(max_tokens - 1)  # reserve index 0 for <unk>
        vocab = {"<unk>": 0}
        for idx, (token, count) in enumerate(most_common, start=1):
            vocab[token] = idx
        return vocab