import random
import os
from pandas.core.interchange.dataframe_protocol import DataFrame
from transformers import AutoTokenizer, BertModel
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch


def similarity_anime(genres1, genres2):
    set1 = set([g.strip().lower() for g in genres1.split(",") if g.strip()])
    set2 = set([g.strip().lower() for g in genres2.split(",") if g.strip()])

    if not set1 or not set2:
        return 0

    jaccard = len(set1 & set2) / len(set1 | set2)

    return 1 if jaccard >= 0.4 else 0


def preprocessing_data(titles, genres, synopsis, num_pairs=5000):
    text = [f"{titles}.{synopsis}" for titles, synopsis in zip(titles, synopsis)]
    n = len(titles)
    pairs = []
    while len(pairs) < num_pairs:

        anime_1, anime_2 = random.sample(range(n), 2)
        print(f"id1:{anime_1}, id2:{anime_2}")
        label = similarity_anime(genres[anime_1], genres[anime_2])
        print(f"label:{label}, genre_1:{genres[anime_1]}, genre_2: {genres[anime_2]}")

        if label == 1 and sum(l == 1 for _, _, l in pairs) >= num_pairs // 2:
            continue
        if label == 0 and sum(l == 0 for _, _, l in pairs) >= num_pairs // 2:
            continue

        pairs.append((text[anime_1], text[anime_2], label))
        print(f"Overall:{pairs}")
    df = pd.DataFrame(pairs, columns=["text1", "text2", "label"])
    df.to_parquet("data/anime_pairs.parquet", index=False, compression="gzip")

    print(df["label"].value_counts(normalize=True))


def move_to_device(item_1, item_2, y, device):
    item_1 = {k: v.to(device) for k, v in item_1.items()}
    item_2 = {k: v.to(device) for k, v in item_2.items()}
    y = y.to(device)
    return item_1, item_2, y


def save_embedding_of_all_anime():
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    class PredictionBert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bert_model = BertModel.from_pretrained("google-bert/bert-base-uncased")
            self.dropout = torch.nn.Dropout(0.3)
            self.linear = torch.nn.Linear(768, 256)

        def forward(self, **x):
            outputs = self.bert_model(**x)
            x = self.dropout(outputs.pooler_output)  # [batch, 768]
            x = self.linear(x)  # [batch, 256]
            x = F.normalize(x, p=2, dim=1)  # нормализация (единичная длина вектора)
            return x

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(BASE_DIR, "../../data/embeddings/anime_recommender.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PredictionBert().to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()

    df = pd.read_parquet(
        os.path.join(BASE_DIR, "../../data/processed/parsed_anime_data.parquet")
    )

    anime_texts = [f"{t}. {s}" for t, s in zip(df["title"], df["synopsis"])]
    anime_titles = df["title"].tolist()

    embeddings_list = []

    with torch.no_grad():
        for text in anime_texts:
            tokenized = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt",
            ).to(device)

            emb = model(**tokenized)
            embeddings_list.append(emb.cpu().numpy())

    embeddings_matrix = np.vstack(embeddings_list).astype("float32")
    np.save(
        os.path.join(BASE_DIR, "../../data/embeddings/embedding_of_all_anime.npy"),
        embeddings_matrix,
    )
