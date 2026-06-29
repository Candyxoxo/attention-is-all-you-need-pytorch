import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.dropout = nn.Dropout(p=0.1)

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        # x: (batch, seq_len, d_model)
        # → (batch, n_heads, seq_len, d_k)
        x = x.view(batch_size, -1, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, x, context=None, mask=None):
        batch_size = x.size(0)
        kv = context if context is not None else x

        q = self.wq(x)
        k = self.wk(kv)
        v = self.wv(kv)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled Dot-Product Attention
        # scores = QKᵀ / √dₖ
        scores = torch.matmul(q, k.transpose(-2,-1))/(self.d_k ** 0.5)

        # apply mask (causal or padding) before softmax
        if mask is not None:
          scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax over last axis → attention weights
        weights = F.softmax(scores, dim = -1)
        weights = self.dropout(weights)

        # weighted sum of values
        output = torch.matmul(weights, v)
        # end Scaled Dot-Product Attention

        output = output.transpose(1,2).contiguous().view(batch_size, -1, self.d_k * self.n_heads)
        output = self.wo(output)
        return output