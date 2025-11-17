"""
Probe the collate function and padding/masks in the fixed no-<sos>/<eos> setup.
"""

import torch

from data_module import make_dataloader
from dataset import data
from vocab import hanzi_stoi, pinyin_stoi


def describe_batch():
    dataloader = make_dataloader(data[:4], batch_size=2, shuffle=False)
    batch = next(iter(dataloader))
    print("\nBatch shapes (no <sos>/<eos> tokens added):")
    print("src shape:", tuple(batch["src"].shape))
    print("tgt_in shape:", tuple(batch["tgt_in"].shape))
    print("tgt_out shape:", tuple(batch["tgt_out"].shape))
    print("src_key_padding_mask shape:", tuple(batch["src_key_padding_mask"].shape))
    print("tgt_key_padding_mask shape:", tuple(batch["tgt_key_padding_mask"].shape))

    # Show the first sample's ids for a quick glance.
    pad_src = pinyin_stoi["<pad>"]
    pad_tgt = hanzi_stoi["<pad>"]
    first_src = batch["src"][0].tolist()
    first_tgt_in = batch["tgt_in"][0].tolist()
    first_tgt_out = batch["tgt_out"][0].tolist()
    print("first src ids:", first_src)
    print("first tgt_in ids:", first_tgt_in)
    print("first tgt_out ids:", first_tgt_out)
    print("first src lengths (no pad):", len([i for i in first_src if i != pad_src]))
    print("first tgt lengths (no pad):", len([i for i in first_tgt_out if i != pad_tgt]))


if __name__ == "__main__":
    torch.manual_seed(0)
    describe_batch()
