from collections import Counter
import spacy
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Special tokens
special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

# load spacy tokenizers
spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

# Build vocabularies

def get_pairs(split):
    """yields (en_sentence, de_sentence) string pairs"""
    # load Multi30k via HuggingFace
    df = load_dataset("bentrevett/multi30k")
    for row in df[split]:
        yield row['en'], row['de']
        
def build_vocab(token_iter, min_freq=2):
  counter = Counter()
  for tokens in token_iter:
    counter.update(tokens)

  # start with special tokens
  vocab = {tok:i for i, tok in enumerate(special_tokens)}

  # add tokens meeting min_freq threshold
  for token, freq in counter.items():
    if freq >= min_freq and token not in vocab:
      vocab[token] = len(vocab)

  return vocab

def lookup(vocab, token):
    return vocab.get(token, UNK_IDX)  # falls back to <unk>

# Building dataset class
class Multi30kDataset(Dataset):
    def __init__(self, split, vocab_en, vocab_de):
      self.pairs = list(get_pairs(split))
      self.vocab_en = vocab_en
      self.vocab_de = vocab_de

    def __len__(self):
      return len(self.pairs)

    def __getitem__(self, idx):
      en, de = self.pairs[idx]

      # tokenize → numericalize → add <bos> and <eos>
      en_ids = [BOS_IDX] + [lookup(self.vocab_en, token) for token in tokenize_en(en)] + [EOS_IDX]
      de_ids = [BOS_IDX] + [lookup(self.vocab_de, token) for token in tokenize_de(de)] + [EOS_IDX]

      return torch.tensor(en_ids), torch.tensor(de_ids)
         
def collate_fn(batch):
    """pads sequences in a batch to the same length"""
    en_batch, de_batch = zip(*batch)
    en_batch = pad_sequence(en_batch, batch_first=True, padding_value=PAD_IDX)
    de_batch = pad_sequence(de_batch, batch_first=True, padding_value=PAD_IDX)
    return en_batch, de_batch

def get_dataloaders(vocab_en, vocab_de, batch_size=128):
    train_ds = Multi30kDataset('train',      vocab_en, vocab_de)
    val_ds   = Multi30kDataset('validation', vocab_en, vocab_de)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader

# Write mask generation functions
def make_src_mask(src, device):
  # True where token is NOT padding — shape: (batch, 1, 1, seq_len)
  # the 1,1 dims broadcast across heads and query positions
  mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
  return mask.to(device)

def make_tgt_mask(tgt, device):
    batch_size, tgt_len = tgt.shape

    # 1. padding mask — True where not padding
    pad_mask = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)
    # shape: (batch, 1, 1, tgt_len)

    # 2. causal mask — move to device FIRST, then reshape
    causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()
    # shape: (tgt_len, tgt_len)  →  (1, 1, tgt_len, tgt_len)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    # combine — both now on the same device
    mask = pad_mask & causal_mask
    return mask.to(device)