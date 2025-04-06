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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEmbedding(nn.Module):
    """
    Adds token embeddings and learned positional embeddings.
    """
    def __init__(self, sequence_length, vocab_size, embed_dim):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(sequence_length, embed_dim)
        self.sequence_length = sequence_length

    def forward(self, x):
        # x: (batch_size, seq_length)
        batch_size, seq_length = x.size()
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand(batch_size, seq_length)
        token_emb = self.token_embeddings(x)          # (batch_size, seq_length, embed_dim)
        pos_emb = self.position_embeddings(positions)   # (batch_size, seq_length, embed_dim)
        return token_emb + pos_emb

class TransformerEncoder(nn.Module):
    """
    A single Transformer encoder block with multi-head self-attention, a feedforward network, 
    and residual connections with layer normalization.
    """
    def __init__(self, embed_dim, dense_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dense_dim)
        self.linear2 = nn.Linear(dense_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, x, src_mask=None):
        # x: (batch_size, seq_length, embed_dim)
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=src_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.linear2(self.activation(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_output))
        return x

class TextClassificationModel(nn.Module):
    """
    Complete model: embeddings, transformer encoder, global max pooling, dropout, and a classification head.
    """
    def __init__(self, sequence_length, vocab_size, embed_dim, num_heads, dense_dim, num_classes):
        super().__init__()
        self.embedding = PositionalEmbedding(sequence_length, vocab_size, embed_dim)
        self.transformer = TransformerEncoder(embed_dim, dense_dim, num_heads)
        # Global max pooling is implemented using adaptive max pooling over the time dimension.
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_length)
        x = self.embedding(x)            # (batch_size, seq_length, embed_dim)
        x = self.transformer(x)          # (batch_size, seq_length, embed_dim)
        # Permute to (batch_size, embed_dim, seq_length) for pooling
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(2)      # (batch_size, embed_dim)
        x = self.dropout(x)
        x = self.fc(x)
        return x

