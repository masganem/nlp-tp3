## Task & Requirements Recap

- **Task**: Given a full pinyin sentence, predict the corresponding hanzi sequence. The main difficulty is **polyphonic pinyin** (one pinyin token can map to multiple hanzi), where the correct character is determined by **context**, including future tokens.
- **Data**:
  - Dataset size is ~75 MB of pinyin–hanzi pairs, mostly with **1:1 alignment** between pinyin tokens and hanzi characters (plus punctuation).
  - The data pipeline already:
    - Loads the CSV correctly.
    - Truncates long entries by percentile to control sequence length.
    - Builds shared vocabularies for pinyin and hanzi, with special tokens and stable JSON persistence.
    - Converts sequences to ID lists and uses a collate function for padding and masks.
- **Model constraints**:
  - Must train on a **normal CPU** (no GPU), so the architecture must be **small and efficient**.
  - Must be **simple to implement and explain** in a university report (clear diagrams/equations; no overly complex architectures).
- **Context handling requirement**:
  - The model must use **local and global pinyin context**, including **next tokens**, to choose the correct hanzi for each pinyin.
  - It is acceptable and natural to use the **entire pinyin sentence** as context, since the full input is known before prediction.
- **Complexity target**:
  - Avoid a full Transformer with multi-head self-attention and many layers.
  - Prefer a **classical RNN-based solution** that is lighter and more directly matched to the mostly 1:1 alignment structure in the data.

## Minimal Architecture Plan: BiGRU Tagger

Given these requirements, a realistic “minimal but solid” model is a **bidirectional GRU tagger** that treats the task as contextual sequence tagging over pinyin:

- **Input**:
  - Use the existing `batch["src"]` pinyin ID sequences from the data pipeline.
  - For a pure tagging setup, we can drop `<sos>`/`<eos>` on the pinyin side and keep one pinyin token per hanzi character (plus padding).

- **Model**:
  - `nn.Embedding` for pinyin:
    - Maps each pinyin token ID to a dense vector (e.g., dimension 128–256).
  - 1-layer **bidirectional GRU encoder**:
    - Hidden size ≈ 128–256 per direction.
    - Reads the embedded pinyin sequence in both forward and backward directions.
  - Contextual representation per position:
    - For each time step `t`, concatenate forward and backward hidden states → `h_t`.
    - `h_t` now encodes both **past and future pinyin context** around position `t`.
  - Output projection:
    - Apply a linear layer + softmax from `h_t` to the hanzi vocabulary:
      - `logits_t = W h_t + b`
      - `P(hanzi_t | pinyins) = softmax(logits_t)`

- **Target / Loss**:
  - Align each hanzi character with its corresponding pinyin token position-wise.
  - Train with `nn.CrossEntropyLoss(ignore_index=pad_idx)`:
    - `pad_idx` is the hanzi `<pad>` index; padded positions are ignored in loss.
    - The loss is summed/averaged over all non-pad positions in the batch.

- **Why this fits the project**:
  - **Uses past + future context**:
    - The bidirectional GRU ensures that `h_t` at each position sees both previous and following pinyin tokens, which is exactly what we need to disambiguate polyphonic syllables.
  - **No decoder, no explicit attention**:
    - We avoid a separate decoder and encoder–decoder attention module.
    - The alignment is mostly 1:1, so a per-position classifier on top of a contextual encoder is enough.
    - This keeps the code and conceptual explanation significantly simpler.
  - **CPU-friendly**:
    - With a single BiGRU layer and moderate hidden size (e.g. 128–256), training on CPU with batch sizes like 32–64 is feasible.
    - The model is small enough to train in a reasonable time for a university project.

- **Extension (if an encoder–decoder is required)**:
  - If the course expects an explicit encoder–decoder architecture, you can:
    - Keep the same BiGRU encoder.
    - Add a simple GRU decoder that, at each step, consumes the previous hanzi and uses the encoder’s sequence of states (optionally with a simple attention mechanism).
  - Conceptually, however, for this specific pinyin→hanzi task with near 1:1 alignment, the **bidirectional encoder + per-step classifier** is the most natural fit and directly addresses the need to look at **future tokens** without the overhead of a full attention-based Transformer.

---

## How to Implement (files in this folder)

Relevant files: `dataset.py` (loads and truncates data), `vocab.py` (builds/saves vocab), `data_module.py` (dataset + collate/dataloader). Below is a sketch of the minimal BiGRU tagger and a training loop using these components. You can place the model in a new `model.py` (or inline in your training script).

### Model: bidirectional GRU tagger

```py
# model.py
import torch
import torch.nn as nn

class BiGRUTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels, pad_idx):
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

    def forward(self, src, src_key_padding_mask=None):
        # src: [B, S]
        x = self.embed(src)  # [B, S, E]
        enc_out, _ = self.encoder(x)  # [B, S, 2H]
        logits = self.proj(enc_out)   # [B, S, V_hanzi]
        return logits
```

### Training loop (CPU-friendly)

```py
# train.py (sketch)
import torch
import torch.nn as nn
from dataset import data
from vocab import pinyin_stoi, hanzi_stoi
from data_module import make_dataloader, PinyinHanziDataset
from model import BiGRUTagger

device = torch.device("cpu")

# Build dataloader (drop <sos>/<eos> on src if you prefer pure tagging; keep as-is for simplicity)
dataloader = make_dataloader(data, batch_size=64, shuffle=True, add_sos_eos=False)

model = BiGRUTagger(
    vocab_size=len(pinyin_stoi),
    embed_dim=128,
    hidden_dim=128,
    num_labels=len(hanzi_stoi),
    pad_idx=pinyin_stoi["<pad>"],
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=hanzi_stoi["<pad>"])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        src = batch["src"].to(device)            # [B, S]
        tgt = batch["tgt_out"].to(device)        # [B, S]

        optimizer.zero_grad()
        logits = model(src)                      # [B, S, V]
        loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"epoch {epoch} loss {total_loss/len(dataloader):.4f}")
```

Notes:
- If you drop `<sos>/<eos>` for tagging, ensure `data_module` returns aligned lengths (one pinyin per hanzi); the current pipeline already keeps padding/masks for batching.
- For evaluation/inference, take `logits.argmax(-1)` at each position, then map indices back to hanzi via `hanzi_itos` from `vocab.py`.
- You can reduce `batch_size` or `hidden_dim` if CPU memory/time is tight; increase modestly if training is too slow to converge.
