"""
Run a single forward pass of the BiGRU tagger to inspect logits and argmax outputs.
"""

import torch

from data_module import make_dataloader
from dataset import data
from model import BiGRUTagger
from vocab import hanzi_itos, hanzi_stoi, pinyin_stoi


def main():
    torch.manual_seed(0)

    dataloader = make_dataloader(data[:4], batch_size=2, shuffle=False)
    batch = next(iter(dataloader))

    model = BiGRUTagger(
        vocab_size=len(pinyin_stoi),
        embed_dim=32,
        hidden_dim=32,
        num_labels=len(hanzi_stoi),
        pad_idx=pinyin_stoi["<pad>"],
    )

    logits = model(batch["src"])
    print("logits shape (batch, seq, vocab):", tuple(logits.shape))

    # Take argmax to see the predicted hanzi for the first sample (ignoring padding).
    pred_ids = logits.argmax(dim=-1)
    first_pred = pred_ids[0]
    pad_mask = batch["tgt_out"][0].eq(hanzi_stoi["<pad>"])
    valid_len = int((~pad_mask).sum().item())
    decoded = "".join(hanzi_itos[i] for i in first_pred[:valid_len].tolist())
    print("Predicted hanzi string for sample 0:", decoded)


if __name__ == "__main__":
    main()
