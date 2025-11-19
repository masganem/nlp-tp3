import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class BiGRUTagger(nn.Module):
    """
    Bidirectional GRU tagger with dynamic per-position candidate selection.

    Instead of projecting to 20k hanzi, we project to max_candidates dimensions
    and use the pinyin_hanzi_map to interpret which candidates those correspond to.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        pad_idx: int,
        pinyin_hanzi_map: Dict[int, List[int]],
        context_window: int = 1,  # window=1 means bigram context: [i-1, i, i+1]
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.context_window = context_window
        self.hidden_dim = hidden_dim

        # Determine max candidates across all pinyin
        self.max_candidates = max(len(candidates) for candidates in pinyin_hanzi_map.values())

        # Project from concatenated context window to max_candidates
        # context_window=2 means [h_{i-2}, h_{i-1}, h_i, h_{i+1}, h_{i+2}] → 5 * (hidden_dim * 2)
        context_size = (2 * context_window + 1) * hidden_dim * 2
        self.proj = nn.Linear(context_size, self.max_candidates)

        # Store the pinyin→hanzi mapping
        self.pinyin_hanzi_map = pinyin_hanzi_map

        # Pre-build candidate lookup tensor: [vocab_size, max_candidates]
        # candidates[pinyin_id, i] = hanzi_id for i-th valid candidate
        # Pad with -1 for pinyin with fewer than max_candidates options
        self.register_buffer("candidates", self._build_candidate_tensor(pinyin_hanzi_map))

    def _build_candidate_tensor(self, pinyin_hanzi_map: Dict[int, List[int]]) -> torch.Tensor:
        """
        Build [vocab_size, max_candidates] tensor of candidate hanzi IDs.
        Padded with -1 where pinyin has fewer than max_candidates options.
        """
        vocab_size = max(pinyin_hanzi_map.keys()) + 1
        candidates = torch.full((vocab_size, self.max_candidates), -1, dtype=torch.long)

        for pinyin_id, valid_hanzi_ids in pinyin_hanzi_map.items():
            for i, hanzi_id in enumerate(valid_hanzi_ids):
                candidates[pinyin_id, i] = hanzi_id

        return candidates

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src: LongTensor of shape [batch, seq_len] containing pinyin token ids.

        Returns:
            candidate_logits: [batch, seq_len, max_candidates] logits over candidates
            candidate_ids: [batch, seq_len, max_candidates] actual hanzi IDs for each candidate
        """
        batch_size, seq_len = src.shape

        # Encode context
        x = self.embed(src)
        enc_out, _ = self.encoder(x)  # [batch, seq_len, hidden_dim * 2]

        # Build context windows by concatenating neighboring hidden states
        # Pad enc_out with zeros at boundaries
        pad_size = self.context_window
        enc_padded = torch.nn.functional.pad(
            enc_out,
            (0, 0, pad_size, pad_size),  # Pad seq_len dimension
            mode='constant',
            value=0
        )  # [batch, seq_len + 2*pad_size, hidden_dim * 2]

        # Collect context windows for each position
        context_features = []
        for offset in range(-self.context_window, self.context_window + 1):
            # Shift by offset and extract the relevant slice
            start_idx = pad_size + offset
            end_idx = start_idx + seq_len
            context_features.append(enc_padded[:, start_idx:end_idx, :])

        # Concatenate all context positions: [batch, seq_len, context_size]
        context = torch.cat(context_features, dim=-1)

        # Project to candidate scores
        candidate_logits = self.proj(context)  # [batch, seq_len, max_candidates]

        # Look up which hanzi IDs these candidates correspond to
        candidate_ids = self.candidates[src]  # [batch, seq_len, max_candidates]

        # Mask out padding candidates (where candidate_id == -1)
        mask = candidate_ids == -1
        candidate_logits = candidate_logits.masked_fill(mask, float("-inf"))

        return candidate_logits, candidate_ids
