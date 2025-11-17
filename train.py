import torch
import torch.nn as nn

from data_module import make_dataloader
from dataset import data
from model import BiGRUTagger
from vocab import hanzi_stoi, pinyin_stoi


# Training is intended to run on CPU for simplicity/reproducibility.
DEVICE = torch.device("cpu")


def train_model(
    num_epochs: int = 5,
    batch_size: int = 64,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    lr: float = 1e-3,
):
    dataloader = make_dataloader(data, batch_size=batch_size, shuffle=True)
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
        for batch in dataloader:
            src = batch["src"].to(DEVICE)
            tgt = batch["tgt_out"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(src)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"epoch {epoch:02d} - loss: {avg_loss:.4f}")

    return model


if __name__ == "__main__":
    train_model()
