import torch
import torch.nn as nn


class BiGRUTagger(nn.Module):
    """Bidirectional GRU tagger for pinyin→hanzi mapping."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_labels: int, pad_idx: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: LongTensor of shape [batch, seq_len] containing pinyin token ids.

        Returns:
            Logits of shape [batch, seq_len, num_labels] for each hanzi position.
        """
        x = self.embed(src)
        enc_out, _ = self.encoder(x)
        logits = self.proj(enc_out)
        return logits
