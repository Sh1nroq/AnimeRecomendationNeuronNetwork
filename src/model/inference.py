import os.path

import faiss
import pandas as pd
import torch
from transformers import AutoTokenizer, BertModel
import torch.nn.functional as F
import numpy as np


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

query = "Ichigo Kurosaki is an ordinary high schooler—until his family is attacked by a Hollow, a corrupt spirit that seeks to devour human souls. It is then that he meets a Soul Reaper named Rukia Kuchiki, who gets injured while protecting Ichigo's family from the assailant. To save his family, Ichigo accepts Rukia's offer of taking her powers and becomes a Soul Reaper as a result."

tokens = tokenizer(
    query, return_tensors="pt", truncation=True, padding="max_length", max_length=128
).to(device)

with torch.no_grad():
    query_emb = model(**tokens).cpu().numpy()

distances, indices = index.search(query_emb, k=5)

print("\nРекомендации по запросу:")
for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
    title = anime_titles[idx]
    print(f"{rank}. {title} (similarity={dist:.4f})")
