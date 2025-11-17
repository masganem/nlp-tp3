"""
Quick sanity checks on the dataset truncation and basic statistics.
"""

from dataset import data, raw_data


def main():
    print("Raw dataset size:", len(raw_data))
    print("Truncated dataset size (99th percentile length cap):", len(data))

    hanzi_lengths = [len(pair[0]) for pair in data]
    pinyin_lengths = [len(pair[1].split()) for pair in data]
    print("Max hanzi length (truncated):", max(hanzi_lengths))
    print("Max pinyin length (truncated):", max(pinyin_lengths))
    print("Example entry:", data[0])


if __name__ == "__main__":
    main()
