# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an NLP project (TP3) that implements a Pinyin-to-Hanzi sequence tagger using a Bidirectional GRU neural network. The model learns to convert romanized Chinese (pinyin with tone numbers) into Chinese characters (hanzi).

**Task**: Given a space-separated pinyin sequence like `"ni3 hao3 ma3"`, predict the corresponding hanzi characters `"你好吗"`.

## Core Architecture

### Data Flow

1. **Dataset Loading** (`dataset.py`):
   - Fetches Pinyin-Hanzi pairs from HuggingFace: `Duyu/Pinyin-Hanzi`
   - Caches dataset to `data/pinyin2hanzi.csv`
   - Filters to 99th percentile by length to reduce padding overhead
   - Enforces strict alignment: only keeps pairs where `len(hanzi) == len(pinyin.split())`
   - Exposes `raw_data` (unfiltered) and `data` (filtered)

2. **Vocabulary Building** (`vocab.py`):
   - Builds separate vocabularies for pinyin tokens and hanzi characters
   - Special tokens: `<pad>` (index 0), `<unk>` (index 1)
   - Caches vocabularies to `data/hanzi_vocab.json` and `data/pinyin_vocab.json`
   - Exports: `pinyin_stoi`, `pinyin_itos`, `hanzi_stoi`, `hanzi_itos`

3. **Data Module** (`data_module.py`):
   - `PinyinHanziDataset`: PyTorch Dataset that tokenizes (hanzi, pinyin) pairs
   - `collate_batch`: Pads sequences to batch max length
   - `make_dataloader`: Creates DataLoader with custom collation
   - Returns batches with `src` (pinyin ids), `tgt_in`, `tgt_out` (hanzi ids), and padding masks

4. **Model** (`model.py`):
   - `BiGRUTagger`: Embedding → Bidirectional GRU (1 layer) → Linear projection
   - Input: `[batch, seq_len]` pinyin token ids
   - Output: `[batch, seq_len, num_labels]` logits for hanzi prediction
   - Padding-aware via `padding_idx` in embedding layer

5. **Training** (`train.py` and `f.py`):
   - `train.py`: Minimal training loop (5 epochs, no progress bars)
   - `f.py`: Full-featured trainer with tqdm, subset training, and inference demo
   - Loss: CrossEntropyLoss with `ignore_index=hanzi_stoi["<pad>"]`
   - Device: CPU (for simplicity/reproducibility)
   - `f.py` includes `predict_sentence()` for inference

### Validation Dataset

- `data/npcr.csv`: New Practical Chinese Reader dataset with curated (hanzi, pinyin) pairs
- Format: CSV with columns `hanzi,pinyin` (aligned sequences)
- Use this for validation/testing the model on educational-quality examples

## Commands

### Training

```bash
# Quick training (5 epochs, full dataset, no progress bars)
python train.py

# Full-featured training with tqdm and inference demo (2 epochs, 5000 samples)
python f.py
```

### Dataset Exploration

```bash
# Print sample data and vocabulary sets
python dataset.py

# Plot sentence length distributions (raw vs filtered)
python plot/entry_lengths.py

# Plot hanzi frequency distribution (Zipf's law)
python plot/hanzi_frequency.py
```

## Key Design Decisions

1. **Per-token alignment**: The model assumes strict 1:1 correspondence between pinyin tokens and hanzi characters. Data is filtered to enforce this constraint (see `dataset.py:42`).

2. **No sequence-to-sequence**: Unlike typical seq2seq, there are no `<bos>`/`<eos>` tokens. The model operates on direct token-level alignment: `tgt_in = tgt_out = hanzi_ids` (see `data_module.py:25-26`).

3. **Percentile truncation**: Training uses only the shortest 99% of sentences to minimize padding waste. Check `dataset.py:31-37` for the truncation logic.

4. **Cached vocabularies**: Vocabularies are built once and cached to JSON. On subsequent runs, they're loaded from disk for consistency across experiments.

5. **Padding strategy**: Sequences are dynamically padded to the max length in each batch (not global max), controlled by `collate_batch` in `data_module.py:30-43`.

## File Organization

- `model.py`: Neural network architecture
- `train.py`, `f.py`: Training scripts (minimal vs full-featured)
- `dataset.py`: Dataset fetching and preprocessing
- `vocab.py`: Vocabulary construction and persistence
- `data_module.py`: PyTorch Dataset/DataLoader integration
- `data/`: Cached datasets and vocabularies
- `plot/`: Visualization scripts for data analysis
