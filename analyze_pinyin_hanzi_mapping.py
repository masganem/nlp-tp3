"""Analyze the pinyin→hanzi mapping to understand output space constraints."""

from collections import defaultdict
from dataset import data

# Build pinyin → set of valid hanzi mapping
pinyin_to_hanzi = defaultdict(set)

for hanzi_str, pinyin_str in data:
    pinyin_tokens = pinyin_str.split()
    hanzi_chars = list(hanzi_str)

    # Ensure alignment (should already be filtered)
    if len(pinyin_tokens) != len(hanzi_chars):
        continue

    for py, hz in zip(pinyin_tokens, hanzi_chars):
        pinyin_to_hanzi[py].add(hz)

# Statistics
num_pinyin = len(pinyin_to_hanzi)
hanzi_counts = [len(hanzi_set) for hanzi_set in pinyin_to_hanzi.values()]
avg_hanzi_per_pinyin = sum(hanzi_counts) / len(hanzi_counts)
max_hanzi_per_pinyin = max(hanzi_counts)
min_hanzi_per_pinyin = min(hanzi_counts)

print(f"Total unique pinyin syllables: {num_pinyin}")
print(f"Average hanzi per pinyin: {avg_hanzi_per_pinyin:.2f}")
print(f"Max hanzi per pinyin: {max_hanzi_per_pinyin}")
print(f"Min hanzi per pinyin: {min_hanzi_per_pinyin}")
print()

# Show examples
print("Examples of pinyin with most hanzi options:")
sorted_pinyin = sorted(pinyin_to_hanzi.items(), key=lambda x: len(x[1]), reverse=True)
for py, hz_set in sorted_pinyin[:10]:
    print(f"  {py}: {len(hz_set)} hanzi → {', '.join(sorted(hz_set)[:20])}{'...' if len(hz_set) > 20 else ''}")

print()
print("Examples of pinyin with fewest hanzi options:")
for py, hz_set in sorted_pinyin[-10:]:
    print(f"  {py}: {len(hz_set)} hanzi → {', '.join(sorted(hz_set))}")

# Compare to full output space
from vocab import hanzi_stoi
total_hanzi = len(hanzi_stoi) - 2  # Exclude <pad> and <unk>
print()
print(f"Full output space: {total_hanzi} hanzi")
print(f"Average constrained space: {avg_hanzi_per_pinyin:.2f} hanzi")
print(f"Reduction factor: {total_hanzi / avg_hanzi_per_pinyin:.1f}x smaller")
