import json
import yaml
import torch
from tqdm import tqdm
import sacrebleu

from src.model import Transformer
from src.data.dataset import get_pairs
from inference import translate


def evaluate_bleu(model, vocab_en, vocab_de, device, split='test', max_len=50, limit=None):
    """
    Runs greedy translation over a dataset split and computes corpus BLEU
    against the reference German sentences.
    """
    pairs = list(get_pairs(split))
    if limit:
        pairs = pairs[:limit]

    hypotheses = []
    references = []

    with torch.no_grad():
        for en_sentence, de_sentence in tqdm(pairs, desc=f"Translating '{split}' split"):
            hypothesis = translate(model, en_sentence, vocab_en, vocab_de, device, max_len=max_len)
            hypotheses.append(hypothesis)
            references.append(de_sentence)

    # lowercase=True: the model is trained on lowercased tokens (see tokenize_de),
    # so hypotheses are always lowercase while references retain original casing.
    # Scoring case-sensitively would penalize every capitalized word as a mismatch.
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], lowercase=True)

    return bleu, hypotheses, references


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load config
    with open('configs/base.yaml') as f:
        cfg = yaml.safe_load(f)

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
    model.eval()

    bleu, hypotheses, references = evaluate_bleu(model, vocab_en, vocab_de, device, split='test')

    print(f"\nCorpus BLEU: {bleu.score:.2f}")
    print(bleu.format())

    print("\nSample translations")
    for i in range(min(5, len(hypotheses))):
        print(f"HYP: {hypotheses[i]}")
        print(f"REF: {references[i]}")
        print()
