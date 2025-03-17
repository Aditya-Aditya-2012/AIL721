import torch
import torch.nn as nn

class WineModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden1 = nn.Linear(13, 64)
        self.act1 = nn.GELU()
        self.dropout = nn.Dropout(0.5)
        self.hidden2 = nn.Linear(64, 32)
        self.act2 = nn.GELU()
        self.hidden3 = nn.Linear(32, 7)
    
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.dropout(x)
        x = self.act2(self.hidden2(x))
        x = self.dropout(x)
        x = self.hidden3(x)
        return x
