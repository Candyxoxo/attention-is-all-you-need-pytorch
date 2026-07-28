# Attention Is All You Need - From Scratch

PyTorch implementation of "Attention Is All You Need" (Vaswani et al., 2017) — trained on Multi30k En→De translation task.

No pre-built `nn.Transformer` or `nn.MultiheadAttention` modules — every component (scaled dot-product attention, multi-head attention, positional encoding, encoder/decoder stacks, masking) is implemented manually to build a deep, working understanding of the architecture.

---

## Trained Model

The model trained in this repo has been uploaded to HuggingFace Hub. Weights and vocab files are available there:

🤗 [Shadow895/attention-is-all-you-need-pytorch](https://huggingface.co/Shadow895/attention-is-all-you-need-pytorch)

To download and run inference:

```python
from huggingface_hub import hf_hub_download

model_path    = hf_hub_download(repo_id="Shadow895/attention-is-all-you-need-pytorch", filename="transformer_model_v1.pt")
vocab_en_path = hf_hub_download(repo_id="Shadow895/attention-is-all-you-need-pytorch", filename="vocab_en.json")
vocab_de_path = hf_hub_download(repo_id="Shadow895/attention-is-all-you-need-pytorch", filename="vocab_de.json")
```

---

## Overview

This project implements the full encoder-decoder Transformer for sequence-to-sequence translation, including:

- **Scaled dot-product & multi-head attention** — implemented from first principles, supporting both self-attention and cross-attention through a shared module
- **Sinusoidal positional encoding**
- **Position-wise feed-forward networks**
- **Encoder and decoder stacks** with residual connections, layer normalization, and dropout
- **Padding and causal masking** for both encoder self-attention and decoder autoregressive generation
- **Greedy decoding** for inference
- **Corpus BLEU evaluation** on the held-out test set (`sacrebleu`)
- **Attention weight visualization** for interpretability (self-attention and cross-attention heatmaps)

The model is trained on the **Multi30k** English–German parallel corpus, tokenized with spaCy, using the training setup described in the original paper (Adam with the paper's beta values, label smoothing, gradient clipping, and learning-rate-driven convergence).

---

## Architecture

| Component | Details |
|---|---|
| Attention | Multi-head scaled dot-product attention, implemented from scratch |
| Positional Encoding | Fixed sinusoidal encoding (sin/cos), added to token embeddings |
| Feed-Forward | Two-layer MLP with ReLU, applied position-wise |
| Normalization | Post-norm residual connections (`LayerNorm(x + Sublayer(x))`) |
| Masking | Padding mask (encoder) + combined padding & causal mask (decoder) |
| Embeddings | Shared special-token indices, scaled by `√d_model` before adding positional encoding |

### Default Hyperparameters (`configs/base.yaml`)

The paper's original hyperparameters (`d_model=512`, `n_layers=6`) were scaled down to train efficiently on the smaller Multi30k dataset:

| Parameter | Value |
|---|---|
| `d_model` | 256 |
| `n_heads` | 8 |
| `d_ff` | 512 |
| `n_layers` | 3 |
| `dropout` | 0.1 |
| `max_seq_len` | 100 |
| Optimizer | Adam (β₁=0.9, β₂=0.98, ε=1e-9) |
| Learning rate | 1e-4 |
| Label smoothing | 0.1 |
| Batch size | 128 |
| Early stopping patience | 10 epochs |

---

## Project Structure

```
Attention_is_all_you_need/
├── configs/
│   └── base.yaml              # Model & training hyperparameters
├── notebooks/
│   └── replicate_attention.ipynb   # Exploratory notebook + attention visualizations
├── src/
│   ├── data/
│   │   └── dataset.py          # Multi30k loading, tokenization, vocab, masking
│   ├── model/
│   │   ├── __init__.py
│   │   ├── attention.py        # Multi-head scaled dot-product attention
│   │   ├── layers.py           # FeedForward + PositionalEncoding
│   │   └── transformer.py      # Encoder, Decoder, full Transformer
│   └── training.py             # Training & evaluation loop
├── Trained Model/
│   ├── transformer_model_v1.pt # Trained model checkpoint
│   ├── vocab_en.json           # English vocabulary
│   ├── vocab_de.json           # German vocabulary
│   └── loss_curve.png          # Train/val loss curve
├── inference.py                # Greedy autoregressive translation
├── evaluate_bleu.py             # Corpus BLEU evaluation on the test split
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Setup

```bash
git clone <repo-url>
cd Attention_is_all_you_need
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Download the spaCy tokenizer models:

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

---

## Usage

### Training

```bash
python -m src.training
```

- Loads hyperparameters from `configs/base.yaml`
- Builds English/German vocabularies from the Multi30k training split
- Trains with early stopping on validation loss
- Saves the best checkpoint, vocabularies, and a loss curve plot to `Trained Model/`
- If a checkpoint already exists at the configured path, training is skipped and the saved model is loaded directly

### Inference

```bash
python inference.py
```

Loads the trained checkpoint and vocabularies, then greedily decodes translations for a set of sample sentences.

```python
from inference import translate

translation = translate(model, "A man is playing guitar.", vocab_en, vocab_de, device)
```

### Attention Visualization

The `notebooks/replicate_attention.ipynb` notebook contains attention weight heatmaps for both encoder self-attention and decoder cross-attention, useful for inspecting what the model attends to at each layer/head.

### BLEU Evaluation

```bash
python evaluate_bleu.py
```

Runs greedy decoding over the full Multi30k test split (1,000 sentence pairs) and computes corpus BLEU using `sacrebleu`, printing the overall score, n-gram precision breakdown, and a handful of sample translations.

---

## Results

### Training Loss

Training and validation loss over epochs (with early stopping) are plotted automatically and saved to `Trained Model/loss_curve.png`.

### BLEU Score

Evaluated on the full Multi30k test split (1,000 sentence pairs) using greedy decoding:

```
Corpus BLEU: 23.79
BLEU = 23.79  54.7/29.5/18.1/11.0  (BP = 1.000, ratio = 1.126, hyp_len = 13626, ref_len = 12106)
```

Published Multi30k En→De Transformer baselines typically fall in the ~30–38 BLEU range. This model scores below that range, for a few identifiable reasons rather than a broken pipeline, qualitatively, translations are largely fluent and semantically correct (see samples below):

- **Reduced model capacity**: `d_model=256`, 3 layers, vs. the paper's 512/6 — a deliberate choice for faster iteration on limited hardware, at some cost to translation quality.
- **Greedy decoding, not beam search**: the current `translate()` picks the single highest-probability token at each step. Beam search typically adds a few BLEU points on top of the same trained weights by exploring multiple candidate sequences instead of committing early.
- **Vocabulary coverage (`min_freq=2`)**: rarer tokens (e.g. multi-word proper nouns like "Boston Terrier") fall back to `<unk>`, directly costing precision whenever they appear in the reference.

Sample translations (greedy decode):

| Source (EN) | Model Output (DE) | Reference (DE) |
|---|---|---|
| A man in an orange hat starring at something. | ein mann mit einem orangefarbenen hut betrachtet etwas an . | Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt. |
| A Boston Terrier is running on lush green grass in front of a white fence. | ein `<unk>` mit einem weißen zaun rennt auf dem rasen vor einem zaun . | Ein Boston Terrier läuft über saftig-grünes Gras vor einem weißen Zaun. |
| People are fixing the roof of a house. | leute reparieren das dach eines hauses . | Leute Reparieren das Dach eines Hauses. |

---

## Key Implementation Notes

- **Shared attention module**: `MultiHeadAttention` handles both self-attention and cross-attention through an optional `context` argument, avoiding duplicated code between encoder and decoder.
- **Masking**: Padding masks prevent attention to `<pad>` tokens; the decoder additionally uses a causal (lower-triangular) mask combined with the padding mask to enforce autoregressive generation.
- **Embedding scaling**: Token embeddings are scaled by `√d_model` before positional encoding is added, as specified in the paper.
- **Label smoothing** and **gradient clipping** are used during training for stability, matching the original paper's setup.

---

## References

- Vaswani, A. et al. (2017). [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).
- [Multi30k Dataset](https://github.com/multi30k/dataset)

---

## License

See [LICENSE](LICENSE) for details.
