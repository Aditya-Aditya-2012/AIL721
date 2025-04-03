import numpy as np
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SinusodialEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(1000) / (half_dim -1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None]*emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb 

class EncoderBlock(nn.Module):
    def __init__(self, hidden_sz=128, num_heads=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_sz)
        self.multihead_attn = nn.MultiheadAttention(hidden_sz, num_heads=num_heads, batch_first=True, dropout=0.25)
        self.ln2 = nn.LayerNorm(hidden_sz)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_sz, hidden_sz),
            nn.LayerNorm(hidden_sz),
            nn.GELU(),
            nn.Linear(hidden_sz, hidden_sz)
        )
    
    def forward(self, x, key_padding_mask):
        norm_x = self.ln1(x)
        attn_out, _ = self.multihead_attn(norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask)
        x = attn_out + x
        norm_x = self.ln2(x)

        mlp_out = self.mlp(norm_x)

        out = mlp_out + x
        return out

class Transformer(nn.Module):
    def __init__(self, num_emb, output_sz, hidden_sz=128, num_layers=3, num_heads=8):
        super(Transformer, self).__init__()

        self.embedding = nn.Embedding(num_emb, hidden_sz)
        self.pos_emb = SinusodialEmbedding(hidden_sz)

        self.blocks = nn.ModuleList([
            EncoderBlock(hidden_sz, num_heads) for _ in range(num_layers)
        ])

        self.out_vec = nn.Parameter(torch.zeros(1, 1, hidden_sz))
        self.fc_out = nn.Linear(hidden_sz, output_sz)
    
    def forward(self, input_seq):
        batch_sz = input_seq.shape[0]
        key_padding_mask = input_seq == 0
        key_padding_mask = torch.cat((torch.zeros(batch_sz, 1, device=device).bool(), key_padding_mask), 1)

        input_embs = self.embedding(input_seq)

        input_embs = torch.cat((self.out_vec.expand(batch_sz, 1, -1), input_embs), 1)
        batch_sz, l, h = input_embs.shape

        seq_idx = torch.arange(l, device=device)
        pos_emb = self.pos_emb(seq_idx).reshape(1, l, h).expand(batch_sz, l, h)

        embs = input_embs + pos_emb

        for block in self.blocks:
            embs = block(embs, key_padding_mask)
        
        return self.fc_out(embs[:, 0])