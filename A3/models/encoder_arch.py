import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    """
    Simple token embedding (without positional embeddings).
    """
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, x):
        return self.embedding(x)

class PositionalEmbedding(nn.Module):
    """
    Adds token embeddings and fixed sinusoidal positional embeddings.
    
    Args:
        sequence_length (int): Maximum sequence length.
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimension.
    """
    def __init__(self, sequence_length, vocab_size, embed_dim):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim

        # Create the sinusoidal positional encoding matrix
        pe = torch.zeros(sequence_length, embed_dim)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) *
                             (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register pe as a buffer so it's saved in the state_dict but not updated by gradients.
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, sequence_length, embed_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length) with token indices.
        Returns:
            Tensor of shape (batch_size, seq_length, embed_dim) where token embeddings are added to fixed sinusoidal positional embeddings.
        """
        batch_size, seq_length = x.size()
        token_emb = self.token_embeddings(x)  # (batch_size, seq_length, embed_dim)
        pos_emb = self.pe[:, :seq_length, :].expand(batch_size, seq_length, self.embed_dim)
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
    Complete model:
      - Embedding (either token-only or token + positional)
      - A stack of transformer encoder layers
      - Global max pooling, dropout, and a classification head.
    
    Args:
        sequence_length (int): Maximum sequence length.
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        dense_dim (int): Number of neurons in the feedforward layer.
        num_classes (int): Number of output classes.
        num_layers (int): Number of stacked transformer encoder layers.
        use_positional_embedding (bool): If True, use positional embeddings.
    """
    def __init__(self, sequence_length, vocab_size, embed_dim, num_heads, dense_dim, num_classes,
                 num_layers=1, use_positional_embedding=True):
        super().__init__()
        self.sequence_length = sequence_length
        self.use_positional_embedding = use_positional_embedding

        # Choose embedding layer
        if self.use_positional_embedding:
            self.embedding = PositionalEmbedding(sequence_length, vocab_size, embed_dim)
        else:
            self.embedding = TokenEmbedding(vocab_size, embed_dim)

        # Stack multiple Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(embed_dim, dense_dim, num_heads) for _ in range(num_layers)
        ])

        # Global max pooling is done over the sequence dimension
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, embed_dim)
        for encoder in self.encoder_layers:
            x = encoder(x)  # (batch_size, seq_length, embed_dim)
        # Permute to (batch_size, embed_dim, seq_length) for pooling
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(2)  # (batch_size, embed_dim)
        x = self.dropout(x)
        logits = self.fc(x)         # (batch_size, num_classes)
        return logits

if __name__ == "__main__":
    # Quick test of the model
    batch_size = 8
    seq_length = 600
    vocab_size = 20000
    embed_dim = 256
    num_heads = 2
    dense_dim = 32
    num_classes = 5
    num_layers = 3  # stacking 3 encoder layers

    model = TextClassificationModel(sequence_length, vocab_size, embed_dim, num_heads,
                                    dense_dim, num_classes, num_layers=num_layers, use_positional_embedding=True)
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    logits = model(x)
    print("Logits shape:", logits.shape)  # Expect (batch_size, num_classes)
