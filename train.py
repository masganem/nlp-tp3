import os
import torch
import torch.nn as nn
from tqdm import tqdm

from data_module import make_dataloader
from dataset import train_data, eval_data
from model import BiGRUTagger
from vocab import hanzi_stoi, pinyin_stoi, pinyin_hanzi_map


# Training is intended to run on CPU for simplicity/reproducibility.
DEVICE = torch.device("cpu")
CHECKPOINT_DIR = "checkpoints"


def compute_batch_loss_and_metrics(candidate_logits, candidate_ids, tgt, pad_idx):
    tgt_expanded = tgt.unsqueeze(-1)
    matches = candidate_ids == tgt_expanded
    target_indices = matches.long().argmax(dim=-1)
    non_pad_mask = tgt != pad_idx
    valid_mask = matches.any(dim=-1) & non_pad_mask

    predictions = candidate_logits.argmax(dim=-1)
    pred_hanzi = torch.gather(candidate_ids, dim=-1, index=predictions.unsqueeze(-1)).squeeze(-1)

    tokens_with_target = int(valid_mask.sum().item())

    if valid_mask.any():
        logits_flat = candidate_logits[valid_mask]
        targets_flat = target_indices[valid_mask]
        loss = nn.functional.cross_entropy(logits_flat, targets_flat, reduction="sum")
        total_correct = int(((predictions == target_indices) & valid_mask).sum().item())
    else:
        loss = candidate_logits.sum() * 0.0
        total_correct = 0

    bigram_mask = non_pad_mask[:, :-1] & non_pad_mask[:, 1:]
    bigram_correct = (
        (pred_hanzi[:, :-1] == tgt[:, :-1])
        & (pred_hanzi[:, 1:] == tgt[:, 1:])
        & bigram_mask
    )
    total_bigram_correct = int(bigram_correct.sum().item())
    total_bigrams = int(bigram_mask.sum().item())

    return (
        loss,
        tokens_with_target,
        total_correct,
        total_bigram_correct,
        total_bigrams,
    )


def evaluate(model, dataloader, pad_idx):
    model.eval()
    total_loss = 0.0
    tokens_with_target = 0
    total_correct = 0
    total_bigram_correct = 0
    total_bigrams = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch["src"].to(DEVICE)
            tgt = batch["tgt_out"].to(DEVICE)
            candidate_logits, candidate_ids = model(src)
            (
                loss,
                batch_tokens_with_target,
                correct,
                bigram_correct,
                bigrams,
            ) = compute_batch_loss_and_metrics(candidate_logits, candidate_ids, tgt, pad_idx)

            total_loss += loss.item()
            tokens_with_target += batch_tokens_with_target
            total_correct += correct
            total_bigram_correct += bigram_correct
            total_bigrams += bigrams

    avg_loss = total_loss / max(tokens_with_target, 1)

    token_precision = total_correct / max(tokens_with_target, 1)
    bigram_precision = total_bigram_correct / max(total_bigrams, 1)

    return avg_loss, token_precision * 100, bigram_precision * 100


def plot_training_progress(history, save_path: str = "progress.png"):
    if not history:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping progress plot")
        return

    epochs = list(range(1, len(history) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # Precision metrics
    axes[1].plot(epochs, [h["train_prec"] for h in history], label="train token")
    axes[1].plot(epochs, [h["val_prec"] for h in history], label="val token", linestyle="--")
    axes[1].plot(epochs, [h["train_bi_prec"] for h in history], label="train bigram")
    axes[1].plot(epochs, [h["val_bi_prec"] for h in history], label="val bigram", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Precision (%)")
    axes[1].set_title("Token & bigram precision")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def train_model(
    num_epochs: int = 5,
    batch_size: int = 64,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    max_samples: int = None,
):
    subset = train_data[:max_samples] if max_samples else train_data
    dataloader = make_dataloader(subset, batch_size=batch_size, shuffle=True)
    eval_loader = make_dataloader(eval_data, batch_size=batch_size, shuffle=False)
    model = BiGRUTagger(
        vocab_size=len(pinyin_stoi),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        pad_idx=pinyin_stoi["<pad>"],
        pinyin_hanzi_map=pinyin_hanzi_map,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pad_idx = hanzi_stoi["<pad>"]
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        tokens_with_target = 0
        total_targets = 0
        total_bigram_correct = 0
        total_bigrams = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}"):
            src = batch["src"].to(DEVICE)
            tgt = batch["tgt_out"].to(DEVICE)  # [batch, seq_len] true hanzi IDs

            optimizer.zero_grad()
            candidate_logits, candidate_ids = model(src)
            (
                loss,
                batch_tokens_with_target,
                correct,
                bigram_correct,
                bigrams,
            ) = compute_batch_loss_and_metrics(candidate_logits, candidate_ids, tgt, pad_idx)

            if batch_tokens_with_target > 0:
                (loss / batch_tokens_with_target).backward()
                optimizer.step()

                total_loss += loss.item()
                tokens_with_target += batch_tokens_with_target
                total_correct += correct
                total_bigram_correct += bigram_correct
                total_bigrams += bigrams

        avg_loss = total_loss / max(tokens_with_target, 1)
        token_prec = total_correct / max(tokens_with_target, 1)
        bigram_prec = total_bigram_correct / max(total_bigrams, 1)

        eval_loss, eval_prec, eval_bi_prec = evaluate(model, eval_loader, pad_idx)

        train_prec_pct = token_prec * 100
        train_bi_prec_pct = bigram_prec * 100

        history.append(
            {
                "train_loss": avg_loss,
                "val_loss": eval_loss,
                "train_prec": train_prec_pct,
                "val_prec": eval_prec,
                "train_bi_prec": train_bi_prec_pct,
                "val_bi_prec": eval_bi_prec,
            }
        )
        print(
            f"epoch {epoch:02d} - "
            f"loss: {avg_loss:.4f} - token precision: {train_prec_pct:.2f}% - "
            f"bigram precision: {train_bi_prec_pct:.2f}% - "
            f"val loss: {eval_loss:.4f} - val token precision: {eval_prec:.2f}% - "
            f"val bigram precision: {eval_bi_prec:.2f}%"
        )

    plot_training_progress(history, save_path="progress.png")

    # Save checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    samples_str = f"{max_samples}" if max_samples else "full"
    checkpoint_name = f"model_e{num_epochs}_b{batch_size}_emb{embed_dim}_h{hidden_dim}_lr{lr}_s{samples_str}.pt"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nCheckpoint saved: {checkpoint_path}")

    return model


if __name__ == "__main__":
    train_model()
