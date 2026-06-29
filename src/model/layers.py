import math
import torch
import torch.nn as nn

# Position-wise Feed-Forward Networks
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # two linear layers + dropout
        # remember: d_model → d_ff → d_model
        l1 = nn.Linear(d_model, d_ff)
        l2 = nn.Linear(d_ff, d_model)
        self.layers = nn.Sequential(l1,nn.ReLU(), l2)

    def forward(self, x):
        # linear → relu → dropout → linear
        return self.layers(x)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)          # not a learnable param, but moves with .to(device)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:x.size(1)].unsqueeze(0)
    
  
