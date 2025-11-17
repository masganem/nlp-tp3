"""
Inspect vocabulary construction and special tokens.
"""

from vocab import HANZI_VOCAB_PATH, PINYIN_VOCAB_PATH, hanzi_itos, hanzi_stoi, pinyin_itos, pinyin_stoi


def main():
    print("Hanzi vocab size:", len(hanzi_itos))
    print("Pinyin vocab size:", len(pinyin_itos))
    print("Hanzi vocab path:", HANZI_VOCAB_PATH)
    print("Pinyin vocab path:", PINYIN_VOCAB_PATH)
    print("Special tokens (hanzi):", {tok: hanzi_stoi[tok] for tok in ("<pad>", "<unk>")})
    print("Special tokens (pinyin):", {tok: pinyin_stoi[tok] for tok in ("<pad>", "<unk>")})
    print("First 10 hanzi tokens:", hanzi_itos[:10])
    print("First 10 pinyin tokens:", pinyin_itos[:10])


if __name__ == "__main__":
    main()
