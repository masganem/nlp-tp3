"""Check the maximum number of valid hanzi candidates for any pinyin."""

from vocab import pinyin_hanzi_map

max_candidates = max(len(candidates) for candidates in pinyin_hanzi_map.values())
avg_candidates = sum(len(candidates) for candidates in pinyin_hanzi_map.values()) / len(pinyin_hanzi_map)

print(f"Max candidates for any pinyin: {max_candidates}")
print(f"Average candidates per pinyin: {avg_candidates:.2f}")
print(f"Total pinyin tokens: {len(pinyin_hanzi_map)}")

# Show distribution
from collections import Counter
candidate_counts = [len(candidates) for candidates in pinyin_hanzi_map.values()]
distribution = Counter(candidate_counts)
print(f"\nDistribution of candidate counts:")
for count in sorted(distribution.keys())[:20]:
    print(f"  {count} candidates: {distribution[count]} pinyin")
