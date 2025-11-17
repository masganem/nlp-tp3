#!/usr/bin/env python3
"""Plot Hanzi sentence-length histograms for raw and filtered entries."""

import matplotlib.pyplot as plt

from dataset import data, raw_data


def _build_hist(ax, lengths, title, color):
    if not lengths:
        ax.text(0.5, 0.5, "No entries", ha="center", va="center")
        ax.set_title(title)
        ax.set_ylabel("Number of entries")
        return

    bins = max(1, min(max(lengths), 200))
    ax.hist(
        lengths,
        bins=bins,
        edgecolor="black",
        color=color,
    )
    ax.set_title(title)
    ax.set_ylabel("Number of entries")


def main():
    raw_lengths = [len(item[0]) for item in raw_data]
    filtered_lengths = [len(item[0]) for item in data]
    if not raw_lengths:
        raise SystemExit("Dataset contains no entries.")

    fig, (ax_raw, ax_filtered) = plt.subplots(
        2, 1, figsize=(10, 8), constrained_layout=True
    )

    _build_hist(ax_raw, raw_lengths, "Raw dataset sentence lengths", "skyblue")
    _build_hist(
        ax_filtered, filtered_lengths, "Filtered dataset sentence lengths", "orange"
    )

    ax_filtered.set_xlabel("Length (characters)")
    ax_raw.set_yscale("log")
    ax_raw.set_ylim(bottom=0.8)
    plt.show()


if __name__ == "__main__":
    main()
