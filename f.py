"""
Minimal training runner with tqdm progress and a quick inference demo.
Trains a small BiGRU tagger on a subset (configurable) and predicts hanzi for a sample pinyin sentence.
"""

import torch
import torch.nn as nn

from data_module import make_dataloader
from dataset import data
from model import BiGRUTagger
from vocab import hanzi_itos, hanzi_stoi, pinyin_stoi


try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x


DEVICE = torch.device("cpu")


def train_model(
    num_epochs: int = 2,
    batch_size: int = 64,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    max_samples: int = 5000,
):
    """Train on a subset of the data for speed; returns the trained model."""
    subset = data[:max_samples] if max_samples else data
    dataloader = make_dataloader(subset, batch_size=batch_size, shuffle=True)

    model = BiGRUTagger(
        vocab_size=len(pinyin_stoi),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_labels=len(hanzi_stoi),
        pad_idx=pinyin_stoi["<pad>"],
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=hanzi_stoi["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in tqdm(dataloader, desc=f"epoch {epoch}"):
            src = batch["src"].to(DEVICE)
            tgt = batch["tgt_out"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(src)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1
        avg_loss = total_loss / max(steps, 1)
        print(f"epoch {epoch} avg loss: {avg_loss:.4f}")
    return model


def predict_sentence(model: BiGRUTagger, pinyin_sentence: str) -> str:
    """Map a space-separated pinyin sentence to a hanzi string using argmax."""
    tokens = pinyin_sentence.strip().split()
    unk = pinyin_stoi["<unk>"]
    src_ids = [pinyin_stoi.get(tok, unk) for tok in tokens]
    src = torch.tensor(src_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(src)
        pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()
    return "".join(hanzi_itos[i] for i in pred_ids)


if __name__ == "__main__":
    # Quick run: small subset and few epochs for a fast demo.
    model = train_model(num_epochs=2, batch_size=64, max_samples=5000)
    demo_inputs = ["ni3 hao3 ma3", "wo3 men qu4 you3 yong4"]
    for d_i in demo_inputs:
        prediction = predict_sentence(model, d_i)
        print(f"\nInput pinyin: {d_i}")
        print(f"Predicted hanzi: {prediction}")
