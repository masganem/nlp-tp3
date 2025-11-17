#!/usr/bin/env python3
"""Plot Hanzi frequency (rank vs. occurrences) using cached dataset data."""

from collections import Counter

import matplotlib.pyplot as plt

from dataset import raw_data


def main():
    counts = Counter(hz for hz_string, *_ in raw_data for hz in hz_string)
    if not counts:
        raise SystemExit("Dataset contains no entries.")

    ranked = counts.most_common()
    ranks = range(1, len(ranked) + 1)
    freq = [count for _, count in ranked]

    fig, ax_linear = plt.subplots(figsize=(10, 5))
    ax_log = ax_linear.twinx()

    linear_line, = ax_linear.plot(ranks, freq, linewidth=1.5, color="tab:blue", label="linear scale")
    log_line, = ax_log.plot(ranks, freq, linewidth=1.5, color="tab:orange", linestyle="--", label="log scale")
    ax_log.set_yscale("log")

    ax_linear.set_title("Hanzi frequency by rank (most to least common)")
    ax_linear.set_xlabel("Rank")
    ax_linear.set_ylabel("Occurrences (linear)")
    ax_log.set_ylabel("Occurrences (log)")

    ax_linear.grid(True, linestyle="--", alpha=0.6)
    ax_linear.legend(handles=[linear_line, log_line], loc="upper right")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
