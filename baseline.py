"""Most-frequent hanzi baseline for each pinyin token."""

from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

from dataset import eval_data, train_data


def build_most_freq_mapping(pairs: Sequence[Tuple[str, str]]) -> Dict[str, str]:
    """Map each pinyin token to its most frequent hanzi in the data."""
    counts: Dict[str, Counter] = defaultdict(Counter)
    for hanzi_str, pinyin_str in pairs:
        for hz, py in zip(hanzi_str, pinyin_str.split()):
            counts[py][hz] += 1
    return {py: counter.most_common(1)[0][0] for py, counter in counts.items() if counter}


def evaluate_mapping(mapping: Dict[str, str], pairs: Sequence[Tuple[str, str]]):
    total_tokens = 0
    covered_tokens = 0
    correct_tokens = 0
    total_bigrams_possible = 0
    total_bigrams_predicted = 0
    correct_bigrams = 0

    for hanzi_str, pinyin_str in pairs:
        py_tokens = pinyin_str.split()
        preds: List[str | None] = []
        for py in py_tokens:
            pred = mapping.get(py)
            preds.append(pred)
            if pred is not None:
                covered_tokens += 1
        total_tokens += len(py_tokens)

        for idx, (gt, pred) in enumerate(zip(hanzi_str, preds)):
            if pred is None:
                continue
            if pred == gt:
                correct_tokens += 1

        total_bigrams_possible += max(len(preds) - 1, 0)
        for i in range(len(preds) - 1):
            if preds[i] is None or preds[i + 1] is None:
                continue
            total_bigrams_predicted += 1
            if preds[i] == hanzi_str[i] and preds[i + 1] == hanzi_str[i + 1]:
                correct_bigrams += 1

    covered_pct = covered_tokens / max(total_tokens, 1) * 100
    token_precision = correct_tokens / max(covered_tokens, 1) * 100
    bigram_precision = correct_bigrams / max(total_bigrams_predicted, 1) * 100

    return {
        "total_tokens": total_tokens,
        "covered_tokens": covered_tokens,
        "covered_pct": covered_pct,
        "token_precision": token_precision,
        "bigram_precision": bigram_precision,
        "total_bigrams_predicted": total_bigrams_predicted,
        "total_bigrams_possible": total_bigrams_possible,
    }


if __name__ == "__main__":
    mapping = build_most_freq_mapping(train_data)
    stats = evaluate_mapping(mapping, eval_data)

    print("Most-frequent hanzi baseline (train→eval):")
    print(f"pinyin types in baseline: {len(mapping)}")
    print(f"eval tokens: {stats['total_tokens']}")
    print(f"covered tokens: {stats['covered_tokens']} ({stats['covered_pct']:.2f}%)")
    print(f"token precision (on covered): {stats['token_precision']:.2f}%")
    print(f"bigram precision (on covered): {stats['bigram_precision']:.2f}%")
    print(f"bigram count (covered): {stats['total_bigrams_predicted']} / {stats['total_bigrams_possible']} possible")
