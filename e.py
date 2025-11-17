"""
Tiny training smoke test on a small subset to ensure the loop runs end-to-end.
"""

import torch
import torch.nn as nn

from data_module import make_dataloader
from dataset import data
from model import BiGRUTagger
from vocab import hanzi_stoi, pinyin_stoi


def main():
    torch.manual_seed(0)
    subset = data[:256]
    dataloader = make_dataloader(subset, batch_size=16, shuffle=True)

    model = BiGRUTagger(
        vocab_size=len(pinyin_stoi),
        embed_dim=64,
        hidden_dim=64,
        num_labels=len(hanzi_stoi),
        pad_idx=pinyin_stoi["<pad>"],
    )
    criterion = nn.CrossEntropyLoss(ignore_index=hanzi_stoi["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    total_loss = 0.0
    steps = 0
    for batch in dataloader:
        src = batch["src"]
        tgt = batch["tgt_out"]

        optimizer.zero_grad()
        logits = model(src)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1
        if steps >= 5:  # keep it quick
            break

    avg_loss = total_loss / steps if steps else float("nan")
    print(f"Ran {steps} training steps; avg loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
