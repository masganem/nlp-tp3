"""Dataset exploration utilities for pinyin↔hanzi coverage."""

import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

from dataset import data


def build_pinyin_hanzi_dict(pairs: Sequence[Tuple[str, str]]) -> Dict[str, Set[str]]:
    """Return mapping of pinyin token -> set of valid hanzi characters."""
    mapping: Dict[str, Set[str]] = {}
    for hanzi_str, pinyin_str in pairs:
        tokens = pinyin_str.split()
        for hz, py in zip(hanzi_str, tokens):
            bucket = mapping.setdefault(py, set())
            bucket.add(hz)
    return mapping


def list_hanzi_for_pinyin(pinyin_token: str, mapping: Dict[str, Set[str]]) -> List[str]:
    """List all hanzi observed for a pinyin token (sorted for readability)."""
    return sorted(mapping.get(pinyin_token, set()))


def dataset_counts(mapping: Dict[str, Set[str]]) -> Tuple[int, int]:
    """Return counts of unique pinyin tokens and unique hanzi characters."""
    all_hanzi = set().union(*mapping.values()) if mapping else set()
    return len(mapping), len(all_hanzi)


def plot_hanzi_per_pinyin(mapping: Dict[str, Set[str]], save_path: Optional[str] = None, show: bool = False):
    """Plot number of hanzi candidates per pinyin token (sorted desc by ambiguity)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError("matplotlib is required for plotting") from exc

    counts = sorted((len(hz_set) for hz_set in mapping.values()), reverse=True)
    if not counts:
        raise ValueError("mapping is empty; nothing to plot")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(counts)), counts, linewidth=1.5)
    ax.set_xlabel("Pinyin tokens (sorted by ambiguity)")
    ax.set_ylabel("Number of hanzi candidates")
    ax.set_title("Hanzi candidates per pinyin")
    ax.grid(True, linestyle="--", alpha=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return counts


if __name__ == "__main__":
    mapping = build_pinyin_hanzi_dict(data)
    num_pinyin, num_hanzi = dataset_counts(mapping)
    print(f"Unique pinyin tokens: {num_pinyin}")
    print(f"Unique hanzi characters: {num_hanzi}")

    if len(sys.argv) > 1:
        for token in sys.argv[1:]:
            hanzi_list = list_hanzi_for_pinyin(token, mapping)
            print(f"{token}: {''.join(hanzi_list)}")
        sys.exit(0)

    # Example usage for interactive probing
    sample_tokens = ["ni3", "wo3", "de", "ma3"]
    for token in sample_tokens:
        hanzi_list = list_hanzi_for_pinyin(token, mapping)
        print(f"{token}: {''.join(hanzi_list)}")

    print("\nPlotting hanzi-per-pinyin distribution (close the window to exit)...")
    plot_hanzi_per_pinyin(mapping, show=True)
