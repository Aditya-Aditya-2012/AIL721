import torch
import torch.nn as nn

class CLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_filters: int,
        filter_sizes: list,
        hidden_dim: int,
        num_classes: int,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1) 
            )
            for fs in filter_sizes
        ])
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional
        )
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear((num_filters * len(filter_sizes)) + lstm_output_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  
        embedded_cnn = embedded.transpose(1, 2) 
        cnn_outs = []
        for conv_block in self.convs:
            c = conv_block(embedded_cnn)
            c = c.squeeze(-1)
            cnn_outs.append(c)
        cnn_cat = torch.cat(cnn_outs, dim=1)
        lstm_out, (h, c) = self.lstm(embedded)
        if self.bidirectional:
            h = torch.cat([h[-2,:,:], h[-1,:,:]], dim=1)
        else:
            h = h[-1,:,:]
        concat_out = torch.cat([cnn_cat, h], dim=1) 
        logits = self.fc(concat_out)
        return logits
