import math
import os
import re
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader

DATASET_URL = "https://huggingface.co/datasets/Duyu/Pinyin-Hanzi/resolve/main/pinyin2hanzi.csv"
DATASET_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pinyin2hanzi.csv")
_PINYIN_TOKEN_RE = re.compile(r"^[A-Za-z]+[0-9]$")


def fetch_dataset():
    if not os.path.exists(DATASET_CACHE_PATH):
        with urllib.request.urlopen(DATASET_URL) as response:
            content = response.read()
        with open(DATASET_CACHE_PATH, "wb") as cache_file:
            cache_file.write(content)

    with open(DATASET_CACHE_PATH, "r", encoding="utf-8") as cache_file:
        return [
            row.strip().split(',')
            for row in cache_file
            if row.strip()
        ]

# Large sentences force large padding which slows training,
# considering 99% of the entries are much much shorter
# (see skewed distribution in https://huggingface.co/datasets/Duyu/Pinyin-Hanzi/tree/main), we can cut off from the 99th percentile.
def _truncate_to_percentile(data, percentile=0.99):
    if not data:
        return data
    lengths = sorted(len(item[0]) for item in data)
    cutoff_index = max(0, min(len(lengths) - 1, math.ceil(percentile * len(lengths)) - 1))
    cutoff_length = lengths[cutoff_index]
    return [item for item in data if len(item[0]) <= cutoff_length]

raw_data = fetch_dataset()
data = _truncate_to_percentile(raw_data)

hanzi = set(
    [
        hz_char
        for hz_string in [item[0] for item in data]
        for hz_char in hz_string
    ]
)

pinyin = set(
    token
    for py_string in [item[1] for item in data]
    for token in py_string.split()
    if _PINYIN_TOKEN_RE.fullmatch(token)
)

if __name__ == "__main__":
    print(data[:100])
    print(hanzi)
    print(pinyin)
