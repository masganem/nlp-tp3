import json
import os
from collections import defaultdict
from typing import Iterable, List, Tuple, Dict

from dataset import data

SPECIALS: List[str] = ["<pad>", "<unk>"]
_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_REPO_DIR, "data")
HANZI_VOCAB_PATH = os.path.join(DATA_DIR, "hanzi_vocab.json")
PINYIN_VOCAB_PATH = os.path.join(DATA_DIR, "pinyin_vocab.json")
PINYIN_HANZI_MAP_PATH = os.path.join(DATA_DIR, "pinyin_hanzi_map.json")


def build_vocab(tokens: Iterable[str]) -> Tuple[dict, List[str]]:
    """
    Build string↔index mappings from an iterable of tokens.

    Returns:
        stoi: token to index map
        itos: index to token list (the vocabulary itself)
    """
    vocab = SPECIALS + sorted(set(tokens))
    stoi = {tok: i for i, tok in enumerate(vocab)}
    itos = list(vocab)
    return stoi, itos


def save_vocab(itos: List[str], path: str) -> None:
    """Persist vocabulary list to JSON for reuse across runs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(itos, f, ensure_ascii=False, indent=2)


def load_vocab(path: str) -> Tuple[dict, List[str]]:
    """Load vocab JSON and rebuild the index mappings."""
    with open(path, "r", encoding="utf-8") as f:
        itos = json.load(f)
    stoi = {tok: i for i, tok in enumerate(itos)}
    return stoi, itos


def build_pinyin_hanzi_map(data_pairs, pinyin_stoi: dict, hanzi_stoi: dict) -> Dict[int, List[int]]:
    """
    Build mapping from pinyin token indices to valid hanzi character indices.

    Returns:
        Dict mapping pinyin_id -> list of valid hanzi_ids
    """
    # Build string-level mapping
    pinyin_to_hanzi = defaultdict(set)
    for hanzi_str, pinyin_str in data_pairs:
        pinyin_tokens = pinyin_str.split()
        hanzi_chars = list(hanzi_str)
        if len(pinyin_tokens) != len(hanzi_chars):
            continue
        for py, hz in zip(pinyin_tokens, hanzi_chars):
            pinyin_to_hanzi[py].add(hz)

    # Convert to index-level mapping
    pinyin_hanzi_map = {}
    for py_token, py_id in pinyin_stoi.items():
        if py_token in SPECIALS:
            # Special tokens map to themselves or pad
            if py_token == "<pad>":
                pinyin_hanzi_map[py_id] = [hanzi_stoi["<pad>"]]
            else:
                pinyin_hanzi_map[py_id] = [hanzi_stoi["<unk>"]]
        else:
            valid_hanzi = pinyin_to_hanzi.get(py_token, set())
            hanzi_ids = [hanzi_stoi[hz] for hz in valid_hanzi if hz in hanzi_stoi]
            pinyin_hanzi_map[py_id] = sorted(hanzi_ids) if hanzi_ids else [hanzi_stoi["<unk>"]]

    return pinyin_hanzi_map


def save_pinyin_hanzi_map(mapping: Dict[int, List[int]], path: str) -> None:
    """Save the mapping to JSON (keys must be strings for JSON)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert int keys to strings for JSON
    str_mapping = {str(k): v for k, v in mapping.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(str_mapping, f, indent=2)


def load_pinyin_hanzi_map(path: str) -> Dict[int, List[int]]:
    """Load the mapping from JSON and convert keys back to ints."""
    with open(path, "r", encoding="utf-8") as f:
        str_mapping = json.load(f)
    return {int(k): v for k, v in str_mapping.items()}


def _build_default_vocabs():
    """Build or load the shared Hanzi/Pinyin vocabularies and mapping."""

    def _ensure_vocab(path: str, tokens: Iterable[str]) -> Tuple[dict, List[str]]:
        stoi, itos = build_vocab(tokens)
        save_vocab(itos, path)
        return stoi, itos

    hanzi_tokens = (ch for hanzi, _ in data for ch in hanzi)
    pinyin_tokens = (tok for _, pinyin in data for tok in pinyin.split())

    hanzi_stoi, hanzi_itos = _ensure_vocab(HANZI_VOCAB_PATH, hanzi_tokens)
    pinyin_stoi, pinyin_itos = _ensure_vocab(PINYIN_VOCAB_PATH, pinyin_tokens)

    # Build and cache pinyin→hanzi mapping
    pinyin_hanzi_map = build_pinyin_hanzi_map(data, pinyin_stoi, hanzi_stoi)
    save_pinyin_hanzi_map(pinyin_hanzi_map, PINYIN_HANZI_MAP_PATH)

    return hanzi_stoi, hanzi_itos, pinyin_stoi, pinyin_itos, pinyin_hanzi_map


hanzi_stoi, hanzi_itos, pinyin_stoi, pinyin_itos, pinyin_hanzi_map = _build_default_vocabs()
