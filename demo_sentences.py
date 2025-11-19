"""Run inference on demo sentences using a trained model."""

import torch
from model import BiGRUTagger
from vocab import hanzi_itos, pinyin_stoi, pinyin_hanzi_map


DEVICE = torch.device("cpu")


def load_model(checkpoint_path: str) -> BiGRUTagger:
    """Load a trained model from checkpoint."""
    # Recreate model with same architecture as training
    model = BiGRUTagger(
        vocab_size=len(pinyin_stoi),
        embed_dim=128,
        hidden_dim=128,
        pad_idx=pinyin_stoi["<pad>"],
        pinyin_hanzi_map=pinyin_hanzi_map,
    ).to(DEVICE)

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


def predict_sentence(model: BiGRUTagger, pinyin_sentence: str) -> str:
    """Convert pinyin sentence to hanzi."""
    tokens = pinyin_sentence.strip().split()
    unk = pinyin_stoi["<unk>"]
    src_ids = [pinyin_stoi.get(tok, unk) for tok in tokens]
    src = torch.tensor(src_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        candidate_logits, candidate_ids = model(src)
        best_candidate_idx = candidate_logits.argmax(dim=-1)
        pred_hanzi_ids = torch.gather(
            candidate_ids,
            dim=-1,
            index=best_candidate_idx.unsqueeze(-1)
        ).squeeze(-1).squeeze(0).tolist()

    return "".join(hanzi_itos[i] for i in pred_hanzi_ids)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python demo_sentences.py <checkpoint_path>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    print(f"Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path)

    print("\n" + "="*60)
    print("Running inference on demo sentences:")
    print("="*60)

    demo_sentences = [
        "ni3 hao3 ma3",
        "wo3 men qu4 you2 yong3",
        "ming2 tian1 wo3 men qu4 you2 yong3 , hao3 ma ?",
        "jin1 tian1 de tian1 qi4 zen3 me yang4 ?",
        "wo3 yao4 ka1 fei1",
    ]

    for pinyin in demo_sentences:
        hanzi = predict_sentence(model, pinyin)
        print(f"\n{pinyin}")
        print(f"→ {hanzi}")
