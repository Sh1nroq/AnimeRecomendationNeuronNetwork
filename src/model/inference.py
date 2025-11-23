import os.path
from src.utils.utils import get_anime
import faiss
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, BertModel
import torch.nn.functional as F
import numpy as np


class PredictionBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256)
        )

    def forward(self, **x):
        outputs = self.bert_model(**x)
        x = outputs.pooler_output
        x = self.head(x)
        x = F.normalize(x, p=2, dim=1)
        return x

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(BASE_DIR, "../../data/embeddings/anime_recommender_alpha.pt")
filepath_anime = os.path.join(
    BASE_DIR, "../../data/processed/parsed_anime_data.parquet"
)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = PredictionBert().to(device)
model.load_state_dict(torch.load(filepath, map_location=device))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

embeddings_matrix = np.load(
    os.path.join(BASE_DIR, "../../data/embeddings/embedding_of_all_anime.npy")
)

anime = pd.read_parquet(filepath_anime)
anime_titles = anime["title"]

dim = embeddings_matrix.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings_matrix)
print(f"Добавлено {index.ntotal} аниме в FAISS индекс")

name = 'Teekyuu'
query = get_anime(name, filepath_anime, 'synopsis')

tokens = tokenizer(
    query, return_tensors="pt", truncation=True
).to(device)

with torch.no_grad():
    query_emb = model(**tokens).cpu().numpy()

distances, indices = index.search(query_emb, k=100)

genres_query = get_anime(name, filepath_anime, 'genres')
title = []

print("\nРекомендации по запросу:")
for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
    genres_anime = get_anime(anime_titles[idx], filepath_anime, 'genres')
    if any(g in genres_query for g in genres_anime):
        title = anime_titles[idx]
        print(f"{rank}. {title} (similarity={dist:.4f})")
