import json
import os
import yaml      
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .model.transformer import Transformer
from .data.dataset import (
    PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX,
    build_vocab, get_dataloaders,
    make_src_mask, make_tgt_mask,
    tokenize_en, tokenize_de, get_pairs
)

# load config
with open('configs/base.yaml') as f: cfg = yaml.safe_load(f)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
# token iterators for vocab building
def en_tokens(split):
  for en, _ in get_pairs(split):
    yield tokenize_en(en)

def de_tokens(split):
  for _, de in get_pairs(split):
    yield tokenize_de(de)

# vocab_en = build_vocab(en_tokens('train'), min_freq=2)
# vocab_de = build_vocab(de_tokens('train'), min_freq=2)

# # Load dataset
# train_dataloader, val_dataloader = get_dataloaders(vocab_en, vocab_de, batch_size=128)

# with this
vocab_en = build_vocab(en_tokens('train'), min_freq=cfg['data']['min_freq'])
vocab_de = build_vocab(de_tokens('train'), min_freq=cfg['data']['min_freq'])
train_dataloader, val_dataloader = get_dataloaders(vocab_en, vocab_de, batch_size=cfg['training']['batch_size'])

# Model
# paper hyperparameters
# SRC_VOCAB_SIZE = len(vocab_en)
# TGT_VOCAB_SIZE = len(vocab_de)
# D_MODEL    = 256   # paper uses 512 but 256 trains faster on Multi30k
# N_HEADS    = 8
# D_FF       = 512   # paper uses 2048
# N_LAYERS   = 3     # paper uses 6
# DROPOUT    = 0.1
# MAX_SEQ_LEN = 100
# N_EPOCHS   = 100

SRC_VOCAB_SIZE = len(vocab_en)
TGT_VOCAB_SIZE = len(vocab_de)
D_MODEL     = cfg['model']['d_model']
N_HEADS     = cfg['model']['n_heads']
D_FF        = cfg['model']['d_ff']
N_LAYERS    = cfg['model']['n_layers']
DROPOUT     = cfg['model']['dropout']
MAX_SEQ_LEN = cfg['model']['max_seq_len']
N_EPOCHS    = cfg['training']['n_epochs']

model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    d_ff=D_FF,
    n_layers=N_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
    dropout=DROPOUT
).to(device)

# paper uses Adam with specific betas
optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'], betas=(0.9, 0.98), eps=1e-9)

# ignore padding in loss — we don't want to penalize pad token predictions
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

# Train function
def train(model, loader, optimizer, criterion, device):
  model.train()
  total_loss = 0

  for src, tgt in loader:
    src = src.to(device)
    tgt = tgt.to(device)

    # right shift: decoder input is tgt without last token
    tgt_input = tgt[:, :-1]  # <bos> I am fine
    tgt_output = tgt[:, 1:]  # I am fine <eos>  ← what we predict

    src_mask = make_src_mask(src, device)
    tgt_mask = make_tgt_mask(tgt_input, device)

    # forward pass → (batch, tgt_seq_len, tgt_vocab_size)
    logits = model(src, tgt_input, src_mask, tgt_mask)

    # reshape for cross entropy
    # logits:     (batch, tgt_seq_len, vocab_size) → (batch*tgt_seq_len, vocab_size)
    # tgt_output: (batch, tgt_seq_len)             → (batch*tgt_seq_len)
    loss = criterion(
        logits.reshape(-1, logits.shape[-1]),
        tgt_output.reshape(-1)
    )

    # backprop
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # prevent exploding gradients
    optimizer.step()

    total_loss += loss.item()

  return total_loss / len(loader)

# Evaluation function
def evaluate(model, loader, criterion, device):
  model.eval()
  total_loss = 0

  with torch.no_grad():
    for src, tgt in loader:
      src = src.to(device)
      tgt = tgt.to(device)

      tgt_input  = tgt[:, :-1]
      tgt_output = tgt[:, 1:]

      src_mask = make_src_mask(src, device)
      tgt_mask = make_tgt_mask(tgt_input, device)

      logits = model(src, tgt_input, src_mask, tgt_mask)
      loss = criterion(
          logits.reshape(-1, logits.size(-1)),
          tgt_output.reshape(-1)
       )
      total_loss += loss.item()
  return total_loss / len(loader)

if __name__ == "__main__":
  from inference import translate
  
  MODEL_PATH    = cfg['training']['model_path']
  VOCAB_EN_PATH = cfg['training']['vocab_en_path']
  VOCAB_DE_PATH = cfg['training']['vocab_de_path']
  os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
  
  if os.path.exists(MODEL_PATH):
    print("Found saved model — loading...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    train_losses, val_losses = [], []
        
  else:
    print("No saved model — training from scratch...")
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = cfg['training']['patience']
  
    epochs_no_improve = 0

    for epoch in range(N_EPOCHS):
      train_loss = train(model, train_dataloader, optimizer, criterion, device)
      val_loss   = evaluate(model, val_dataloader, criterion, device)

      train_losses.append(train_loss)
      val_losses.append(val_loss)

      if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODEL_PATH)
        with open(VOCAB_EN_PATH, 'w') as f: json.dump(vocab_en, f)   
        with open(VOCAB_DE_PATH, 'w') as f: json.dump(vocab_de, f)
      else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
          print(f"Early stopping at epoch {epoch+1}")
          break

      print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")
  
  if train_losses:
    # plot after training
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses)+1),   val_losses,   label='Val Loss',   marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Transformer Training — Multi30k En→De')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(MODEL_PATH), 'loss_curve.png'), dpi=150)
    plt.show()
  else:
    print("Skipping plot — model loaded from checkpoint.")

    print("\n--- Translations ---")
    test_sentences = [
        "A man is playing guitar.",
        "Two dogs are running in the park.",
        "A woman is reading a book."
    ]
    for sentence in test_sentences:
      translation = translate(model, sentence, vocab_en, vocab_de, device)
      print(f"EN: {sentence}")
      print(f"DE: {translation}")
      print()