# Arabic Hadith Semantic Search

Semantic retrieval over Arabic hadith text using Sentence Transformers and `Omartificial-Intelligence-Space/GATE-AraBert-v1`.

## Pipeline
1. Load `data/hadith/kaggle_hadiths_clean.csv` from Kaggle dataset [alizahidraja/quran-nlp](https://www.kaggle.com/datasets/alizahidraja/quran-nlp).
2. Clean Arabic text and normalize whitespace.
3. Encode hadiths and a user query with GATE-AraBert-v1.
4. Rank corpus entries by cosine similarity and return top-k matches.

## Run
`pip install -r requirements.txt`

`python search.py --csv data/hadith/kaggle_hadiths_clean.csv --query "أصل كل فعل النية" --top-k 4`

The dataset is not included. The first run downloads the model.

## Limitations
Similarity is only a retrieval signal. It does not establish hadith authenticity, chain reliability, legal meaning, or correctness of a religious interpretation. No labelled benchmark is claimed.