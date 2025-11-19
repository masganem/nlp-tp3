import os
import torch
import torch.nn as nn
from tqdm import tqdm

from data_module import make_dataloader
from dataset import data
from model import BiGRUTagger
from vocab import hanzi_stoi, pinyin_stoi, pinyin_hanzi_map


# Training is intended to run on CPU for simplicity/reproducibility.
DEVICE = torch.device("cpu")
CHECKPOINT_DIR = "checkpoints"


def train_model(
    num_epochs: int = 5,
    batch_size: int = 64,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    max_samples: int = None,
):
    subset = data[:max_samples] if max_samples else data
    dataloader = make_dataloader(subset, batch_size=batch_size, shuffle=True)
    model = BiGRUTagger(
        vocab_size=len(pinyin_stoi),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        pad_idx=pinyin_stoi["<pad>"],
        pinyin_hanzi_map=pinyin_hanzi_map,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pad_idx = hanzi_stoi["<pad>"]

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        total_bigram_correct = 0
        total_bigrams = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}"):
            src = batch["src"].to(DEVICE)
            tgt = batch["tgt_out"].to(DEVICE)  # [batch, seq_len] true hanzi IDs

            optimizer.zero_grad()

            # Model returns candidate logits and their corresponding hanzi IDs
            candidate_logits, candidate_ids = model(src)
            # candidate_logits: [batch, seq_len, max_candidates]
            # candidate_ids: [batch, seq_len, max_candidates]

            # For each position, find which candidate index matches the true target
            # Expand tgt to compare: [batch, seq_len, 1]
            tgt_expanded = tgt.unsqueeze(-1)  # [batch, seq_len, 1]

            # Find which candidate matches the target: [batch, seq_len, max_candidates]
            matches = (candidate_ids == tgt_expanded)  # Boolean tensor

            # Convert to target indices for CrossEntropyLoss
            # For each position, find the index of the matching candidate
            target_indices = matches.long().argmax(dim=-1)  # [batch, seq_len]

            # Mask for non-padding positions
            non_pad_mask = tgt != pad_idx

            # Collect predictions for bigram accuracy
            predictions = candidate_logits.argmax(dim=-1)  # [batch, seq_len]
            pred_hanzi = torch.gather(candidate_ids, dim=-1, index=predictions.unsqueeze(-1)).squeeze(-1)

            # Only compute loss on non-padding positions
            loss = 0.0
            batch_size, seq_len = src.shape
            for b in range(batch_size):
                for s in range(seq_len):
                    if non_pad_mask[b, s]:
                        # Check if target exists in candidates
                        if matches[b, s].any():
                            target_idx = target_indices[b, s]
                            logits_at_pos = candidate_logits[b, s]  # [max_candidates]
                            loss += nn.functional.cross_entropy(
                                logits_at_pos.unsqueeze(0),
                                target_idx.unsqueeze(0),
                                reduction='sum'
                            )
                            total_tokens += 1

                            # Accuracy tracking
                            pred_idx = candidate_logits[b, s].argmax()
                            if pred_idx == target_idx:
                                total_correct += 1

                    # Bigram accuracy tracking
                    if s < seq_len - 1 and non_pad_mask[b, s] and non_pad_mask[b, s + 1]:
                        # Check if both positions in the bigram are correct
                        if pred_hanzi[b, s] == tgt[b, s] and pred_hanzi[b, s + 1] == tgt[b, s + 1]:
                            total_bigram_correct += 1
                        total_bigrams += 1

            if total_tokens > 0:
                loss = loss / total_tokens
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / max(total_tokens, 1) * 100
        bigram_accuracy = total_bigram_correct / max(total_bigrams, 1) * 100
        print(f"epoch {epoch:02d} - loss: {avg_loss:.4f} - token acc: {accuracy:.2f}% - bigram acc: {bigram_accuracy:.2f}%")

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
