from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from vocab import hanzi_stoi, pinyin_stoi


class PinyinHanziDataset(Dataset):
    def __init__(self, pairs: Sequence[tuple]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        hanzi, pinyin = self.pairs[idx]
        src_tokens = pinyin.split()
        tgt_tokens = list(hanzi)
        unk_py = pinyin_stoi["<unk>"]
        unk_hz = hanzi_stoi["<unk>"]
        src_ids = [pinyin_stoi.get(t, unk_py) for t in src_tokens]
        tgt_ids = [hanzi_stoi.get(c, unk_hz) for c in tgt_tokens]
        # Pure per-token alignment; no extra boundary tokens.
        tgt_in = tgt_ids
        tgt_out = tgt_ids
        return {"src": src_ids, "tgt_in": tgt_in, "tgt_out": tgt_out}


def collate_batch(batch: List[Dict[str, List[int]]], pad_idx_src: int, pad_idx_tgt: int):
    src = [torch.tensor(b["src"], dtype=torch.long) for b in batch]
    tgt_in = [torch.tensor(b["tgt_in"], dtype=torch.long) for b in batch]
    tgt_out = [torch.tensor(b["tgt_out"], dtype=torch.long) for b in batch]
    src_pad = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=pad_idx_src)
    tgt_in_pad = torch.nn.utils.rnn.pad_sequence(tgt_in, batch_first=True, padding_value=pad_idx_tgt)
    tgt_out_pad = torch.nn.utils.rnn.pad_sequence(tgt_out, batch_first=True, padding_value=pad_idx_tgt)
    return {
        "src": src_pad,
        "tgt_in": tgt_in_pad,
        "tgt_out": tgt_out_pad,
        "src_key_padding_mask": src_pad.eq(pad_idx_src),
        "tgt_key_padding_mask": tgt_in_pad.eq(pad_idx_tgt),
    }


def make_dataloader(
    pairs: Sequence[tuple],
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader with padding-aware collate."""
    dataset = PinyinHanziDataset(pairs)
    collate = lambda batch: collate_batch(batch, pinyin_stoi["<pad>"], hanzi_stoi["<pad>"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
