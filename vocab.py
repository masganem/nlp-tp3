import json
import os
from typing import Iterable, List, Tuple

from dataset import data

SPECIALS: List[str] = ["<pad>", "<sos>", "<eos>", "<unk>"]
_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
HANZI_VOCAB_PATH = os.path.join(_REPO_DIR, "hanzi_vocab.json")
PINYIN_VOCAB_PATH = os.path.join(_REPO_DIR, "pinyin_vocab.json")


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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(itos, f, ensure_ascii=False, indent=2)


def load_vocab(path: str) -> Tuple[dict, List[str]]:
    """Load vocab JSON and rebuild the index mappings."""
    with open(path, "r", encoding="utf-8") as f:
        itos = json.load(f)
    stoi = {tok: i for i, tok in enumerate(itos)}
    return stoi, itos


def _build_default_vocabs():
    """Build or load the shared Hanzi/Pinyin vocabularies."""

    def _ensure_vocab(path: str, tokens: Iterable[str]) -> Tuple[dict, List[str]]:
        if os.path.exists(path):
            return load_vocab(path)
        stoi, itos = build_vocab(tokens)
        save_vocab(itos, path)
        return stoi, itos

    hanzi_tokens = (ch for hanzi, _ in data for ch in hanzi)
    pinyin_tokens = (tok for _, pinyin in data for tok in pinyin.split())

    hanzi_stoi, hanzi_itos = _ensure_vocab(HANZI_VOCAB_PATH, hanzi_tokens)
    pinyin_stoi, pinyin_itos = _ensure_vocab(PINYIN_VOCAB_PATH, pinyin_tokens)
    return hanzi_stoi, hanzi_itos, pinyin_stoi, pinyin_itos


hanzi_stoi, hanzi_itos, pinyin_stoi, pinyin_itos = _build_default_vocabs()

