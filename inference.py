import json
import torch
import yaml
from src.model import Transformer
from src.data.dataset import (
    PAD_IDX, BOS_IDX, EOS_IDX, lookup,
    tokenize_en, tokenize_de,
    make_src_mask, make_tgt_mask)  

def translate(model, sentence, vocab_en, vocab_de, device, max_len=50):
    model.eval()

    # 1. tokenize and numericalize source
    tokens = [BOS_IDX] + [lookup(vocab_en, t) for t in tokenize_en(sentence)] + [EOS_IDX]
    src = torch.tensor(tokens).unsqueeze(0).to(device)  # (1, src_len)
    src_mask = make_src_mask(src, device)

    # 2. encode source once
    with torch.no_grad():
        src_emb = model.dropout(model.pe(model.src_emb(src) * (model.d_model ** 0.5)))
        enc_out = model.encoder(src_emb, src_mask)

    # 3. decode autoregressively — one token at a time
    tgt_tokens = [BOS_IDX]

    for _ in range(max_len):
        tgt = torch.tensor(tgt_tokens).unsqueeze(0).to(device)  # (1, current_len)
        tgt_mask = make_tgt_mask(tgt, device)

        with torch.no_grad():
            tgt_emb = model.dropout(model.pe(model.tgt_emb(tgt) * (model.d_model ** 0.5)))
            dec_out = model.decoder(tgt_emb, enc_out, src_mask, tgt_mask)
            logits  = model.proj(dec_out)  # (1, current_len, tgt_vocab_size)

        # take the last token's prediction
        next_token = logits[0, -1, :].argmax().item()
        tgt_tokens.append(next_token)

        # stop if <eos> predicted
        if next_token == EOS_IDX:
            break

    # 4. convert indices back to words
    idx_to_de = {v: k for k, v in vocab_de.items()}
    translated = [idx_to_de.get(i, '<unk>') for i in tgt_tokens[1:-1]]  # strip <bos> and <eos>
    return ' '.join(translated)

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load config
    with open('configs/base.yaml') as f: cfg = yaml.safe_load(f)

    MODEL_PATH    = cfg['training']['model_path']
    VOCAB_EN_PATH = cfg['training']['vocab_en_path']
    VOCAB_DE_PATH = cfg['training']['vocab_de_path']

    # load vocabs
    with open(VOCAB_EN_PATH) as f: vocab_en = json.load(f)
    with open(VOCAB_DE_PATH) as f: vocab_de = json.load(f)

    # instantiate model from config
    model = Transformer(
        src_vocab_size=len(vocab_en),
        tgt_vocab_size=len(vocab_de),
        d_model=cfg['model']['d_model'],
        n_heads=cfg['model']['n_heads'],
        d_ff=cfg['model']['d_ff'],
        n_layers=cfg['model']['n_layers'],
        max_seq_len=cfg['model']['max_seq_len'],
        dropout=cfg['model']['dropout']
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
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