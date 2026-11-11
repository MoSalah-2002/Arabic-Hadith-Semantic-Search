"""Arabic hadith semantic retrieval using GATE-AraBert-v1."""
import argparse, re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL = "Omartificial-Intelligence-Space/GATE-AraBert-v1"

def clean(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"[\u200f\u0060“’”‘ﷺ]", "", str(text))
    text = re.sub(r"""[`"']""", "", text)
    return re.sub(r"\s+", " ", text).strip()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    if "text_ar" not in df.columns:
        raise ValueError("CSV must contain a text_ar column")
    texts = df["text_ar"].map(clean)
    texts = texts[texts.str.len() > 0].reset_index(drop=True)
    model = SentenceTransformer(MODEL)
    corpus = model.encode(texts.tolist(), convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=True)
    query = model.encode([clean(args.query)], convert_to_numpy=True,
                         normalize_embeddings=True)[0]
    scores = corpus @ query
    k = min(max(args.top_k, 1), len(scores))
    for i in np.argsort(scores)[-k:][::-1]:
        print(f"{scores[i]:.4f} | {texts.iloc[i]}")

if __name__ == "__main__":
    main()
