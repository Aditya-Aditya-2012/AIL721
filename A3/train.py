import argparse
import os
import glob
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

try:
    import torchtext
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
    torchtext_available = True
except ImportError:
    torchtext_available = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models.clstm import CLSTM
from models.encoder_arch import TextClassificationModel 
from models.dataset import TextClassificationDataset, build_vocab
from models.init_seed import initialize
from utils.preprocessing import make_train_data, make_test_data, train_val_split
from utils.train_helper import train_model, test_model, evaluate_model
from utils.make_plots import make_plot


def main():
    parser = argparse.ArgumentParser(
        description="Train a text classification model using transformer, CLSTM or BiLSTM with either torchtext or basic tokenization."
    )
    parser.add_argument("--model_name", type=str, choices=["transformer", "clstm", "biclstm"],
                        default="transformer", help="Select model: transformer, clstm, or biclstm.")
    parser.add_argument("--use_positional", type=bool, default=True, help="Use positional embedding or not")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of encoder layers (used for transformer).")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--num_heads", type=int, default=64, help="Number of heads (for transformer).")
    parser.add_argument("--dense_dim", type=int, default=32, help="Dense dimension in transformer.")
    # CLSTM parameters
    parser.add_argument("--num_filters", type=int, default=100, help="Number of filters for each convolution in CLSTM.")
    parser.add_argument("--filter_sizes", type=int, nargs='+', default=[3,4,5],
                        help="Filter sizes for CLSTM convolutions.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for LSTM in CLSTM.")
    
    # Tokenization & Data parameters
    parser.add_argument("--tokenizer_name", type=str, choices=["torchtext", "basic"],
                        default="basic", help="Choose 'torchtext' tokenizer (if available) or 'basic' tokenizer.")
    parser.add_argument("--max_length", type=int, default=600, help="Maximum sequence length.")
    parser.add_argument("--max_tokens", type=int, default=20000, help="Maximum vocabulary tokens.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of output classes.")
    
    # Paths and training parameters
    parser.add_argument("--save_model_dir", type=str, default="weights",
                        help="File path to save the best model.")
    parser.add_argument("--plot_dir", type=str, default="plots",
                        help="Directory where plots will be saved.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning_rate")
    args = parser.parse_args()

    model_name = args.model_name
    num_layers = args.num_layers
    embed_dim = args.embed_dim
    num_heads = args.num_heads
    dense_dim = args.dense_dim
    num_filters = args.num_filters
    filter_sizes = args.filter_sizes
    hidden_dim = args.hidden_dim
    tokenizer_name = args.tokenizer_name
    max_length = args.max_length
    max_tokens = args.max_tokens
    batch_size = args.batch_size
    num_classes = args.num_classes
    save_model_dir = args.save_model_dir
    plot_dir = args.plot_dir
    nepochs = args.epochs
    use_positional_embedding = args.use_positional

    
    
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
        
    if model_name=='transformer':
        save_model_path = f'{save_model_dir}/{model_name}_{int(use_positional_embedding)}_{num_layers}_{tokenizer_name}'
    else:
        save_model_path = f'{save_model_dir}/{model_name}_{tokenizer_name}'
    use_torchtext = (tokenizer_name == "torchtext") and torchtext_available

    initialize(seed=0)
    data_dir = "Datasets"
    make_train_data(dir_name=data_dir)
    make_test_data(dir_name=data_dir)
    train_val_split(dir_name=data_dir, split_ratio=0.2)

    if use_torchtext:
        print("Using torchtext tokenizer (basic_english)")
        tokenizer = get_tokenizer("basic_english")
    else:
        print("Using basic Python tokenizer (split on whitespace)")
        tokenizer = lambda x: x.lower().split()

    train_dir = os.path.join(data_dir, "train")
    vocab = build_vocab(train_dir, tokenizer, use_torchtext=use_torchtext, max_tokens=max_tokens)
    vocab_size = len(vocab)

    train_dataset = TextClassificationDataset(train_dir, vocab, tokenizer, max_length)
    val_dir = os.path.join(data_dir, "val")
    val_dataset = TextClassificationDataset(val_dir, vocab, tokenizer, max_length)
    test_dir = os.path.join(data_dir, "test")
    test_dataset = TextClassificationDataset(test_dir, vocab, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    def init_model(model_name):
        if model_name == "transformer":
            model = TextClassificationModel(
                sequence_length=max_length,
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dense_dim=dense_dim,
                num_classes=num_classes,
                num_layers=num_layers,
                use_positional_embedding=use_positional_embedding
            )
        elif model_name == "clstm":
            model = CLSTM(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_filters=num_filters,
                filter_sizes=filter_sizes,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                bidirectional=False
            )
        elif model_name == "biclstm":
            model = CLSTM(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_filters=num_filters,
                filter_sizes=filter_sizes,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                bidirectional=True
            )
        else:
            raise ValueError("Invalid model name provided.")
        return model

    model = init_model(model_name)
    print("Using device:", device)
    model.to(device)

    train_array, val_array = train_model(model, train_loader, val_loader, device, epochs=nepochs, save_model_path=save_model_path, lr=args.lr)
    
    model.load_state_dict(torch.load(save_model_path, map_location=device))
    test_model(model, test_loader, device)
    preds, labels = evaluate_model(model, test_loader, device)
    micro_f1 = f1_score(labels, preds, average="micro")
    print("Micro-average F1 score on test set: {:.4f}".format(micro_f1))
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    make_plot(train_array, val_array, tokenizer_name, model_name, num_layers, use_positional_embedding, plot_dir)


if __name__ == "__main__":
    main()
