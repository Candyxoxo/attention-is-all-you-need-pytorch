import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .layers import FeedForward, PositionalEncoding
from ..data.dataset import PAD_IDX 

# Encoder
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # MultiHeadAttention, FeedForward, 2x LayerNorm, dropout
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x = LayerNorm(x + dropout(MHA(x)))
        # x = LayerNorm(x + dropout(FFN(x)))
        x = self.norm1(x + self.dropout(self.mha(x, mask = mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])  # n_layers EncoderLayers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # pass x through each layer sequentially
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
     
# Decoder
# Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
      super().__init__()
      # you need: 2x MultiHeadAttention, 1x FeedForward, 3x LayerNorm
      self.mha1 = MultiHeadAttention(d_model, n_heads)
      self.mha2 = MultiHeadAttention(d_model, n_heads)
      self.ff = FeedForward(d_model, d_ff)
      self.ln1 = nn.LayerNorm(d_model)
      self.ln2 = nn.LayerNorm(d_model)
      self.ln3 = nn.LayerNorm(d_model)
      self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, trg_mask=None):
      # sublayer 1 — masked self-attention
      x = self.ln1(x + self.dropout(self.mha1(x, mask = trg_mask)))

      # sublayer 2 — cross-attention (Q from decoder, K,V from encoder)
      x = self.ln2(x + self.dropout(self.mha2(x, context = enc_out, mask = src_mask)))

      # sublayer 3 — feed-forward
      x = self.ln3(x + self.dropout(self.ff(x)))
      return x

# Decoder
class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask=None, trg_mask=None):
        # each layer needs both x and encoder_output
        # pass x through each layer sequentially
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, trg_mask)
        return self.norm(x)
    
# Full Transformer Architecture
class Transformer(nn.Module):
    def __init__(self,  src_vocab_size, tgt_vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # 1. two embedding matrices (src and tgt) — shape (vocab_size, d_model)
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        # 2. positional encoding — already have the function
        self.pe = PositionalEncoding(d_model, max_seq_len)
        # 3. Encoder
        self.encoder = Encoder(d_model, n_heads, d_ff, n_layers, dropout)
        # 4. Decoder
        self.decoder = Decoder(d_model, n_heads, d_ff, n_layers, dropout)
        # 5. final linear projection — shape (d_model, vocab_size)
        self.proj = nn.Linear(d_model, tgt_vocab_size)
        # 6. Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src, tgt are integer token indices, shape (seq_len,)
        # 1. embed src, add positional encoding → feed to Encoder
        # 2. embed tgt, add positional encoding → feed to Decoder with enc_output
        src_emb = self.dropout(self.pe(self.src_emb(src) * (self.d_model ** 0.5)))
        tgt_emb = self.dropout(self.pe(self.tgt_emb(tgt) * (self.d_model ** 0.5)))

        enc_output = self.encoder(src_emb, src_mask)
        dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)
        # 3. project decoder output to vocab_size
        return self.proj(dec_output)  # (batch, tgt_seq_len, vocab_size)