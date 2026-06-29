PyTorch implementation of "Attention Is All You Need" (Vaswani et al., 2017) — trained on Multi30k En→De translation task

## Pretrained Model

Model weights, vocab files available on HuggingFace:

🤗 [Shadow895/attention-is-all-you-need-pytorch](https://huggingface.co/Shadow895/attention-is-all-you-need-pytorch)

To download and run inference:
```python
from huggingface_hub import hf_hub_download

model_path    = hf_hub_download(repo_id="Shadow895/attention-is-all-you-need-pytorch", filename="transformer_model_v1.pt")
vocab_en_path = hf_hub_download(repo_id="Shadow895/attention-is-all-you-need-pytorch", filename="vocab_en.json")
vocab_de_path = hf_hub_download(repo_id="Shadow895/attention-is-all-you-need-pytorch", filename="vocab_de.json")
```
