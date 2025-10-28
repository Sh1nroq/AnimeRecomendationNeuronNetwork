import os.path

import torch
from torch import nn, no_grad
from transformers import AutoTokenizer, BertModel
from src.model.my_anime_dataset import MyAnimeDataset
from src.utils.utils import move_to_device
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(BASE_DIR, "../../data/processed/anime_pairs.parquet")

df = pd.read_parquet(filename)

train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")


class AnimeRecommender(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(768, 256)

    def forward(self, **x):
        outputs = self.bert_model(**x)
        x = self.dropout(outputs.pooler_output)
        x = self.linear(x)
        x = F.normalize(x, p=2, dim=1)
        return x


model = AnimeRecommender().to(device)

for name, param in model.bert_model.named_parameters():
    if "encoder.layer" in name:
        param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

train_dataset = MyAnimeDataset(train_df, tokenizer)
val_dataset = MyAnimeDataset(val_df, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

learning_rate = 3e-5
weight_decay = 0.01
batch_size = 32
num_epochs = 5
margin = 0.5
max_length = 256
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)

total_steps = len(train_dataloader) * num_epochs
warmup_steps = int(0.1 * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)


def unfreeze_layers(model, num_layers_to_unfreeze):
    total_layers = 12  # у bert-base-uncased 12 encoder layers

    for name, param in model.bert_model.named_parameters():
        if "encoder.layer." in name:
            parts = name.split(".")
            try:
                layer_num = int(parts[2])
            except ValueError:
                continue
            if layer_num >= total_layers - num_layers_to_unfreeze:
                param.requires_grad = True


def fine_tuning(model, train_dataloader, optimizer, margin):
    model.train()
    total_loss = 0
    num_batches = 0

    for item_1, item_2, y in tqdm(train_dataloader, desc="Training"):
        item_1, item_2, y = move_to_device(item_1, item_2, y, device)

        token_text_1 = model(**item_1)
        token_text_2 = model(**item_2)
        # print(token_text_1.shape, token_text_2.shape)

        cosine_similarity = F.cosine_similarity(token_text_1, token_text_2, dim=-1)
        loss = (
            y * (1 - cosine_similarity) ** 2
            + (1 - y) * (torch.clamp(cosine_similarity - margin, min=0)) ** 2
        )

        loss = loss.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_dataloader, margin):
    model.eval()
    total_loss = 0
    num_batches = 0

    with no_grad():
        for item_1, item_2, y in tqdm(val_dataloader):
            item_1, item_2, y = move_to_device(item_1, item_2, y, device)

            token_text_1 = model(**item_1)
            token_text_2 = model(**item_2)

            cosine_similarity = F.cosine_similarity(token_text_1, token_text_2, dim=-1)
            loss = (
                y * (1 - cosine_similarity) ** 2
                + (1 - y) * (torch.clamp(cosine_similarity - margin, min=0)) ** 2
            )

            total_loss += loss.mean().item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        return avg_loss


for epoch in range(num_epochs):
    if epoch % 2 == 0 and epoch > 0:
        num_to_unfreeze = epoch // 2 * 2
        print(f"Размораживаем верхние {num_to_unfreeze} слоя(ев)")
        unfreeze_layers(model, num_to_unfreeze)

    train_loss = fine_tuning(model, train_dataloader, optimizer, margin)
    val_loss = validate(model, val_dataloader, margin)
    print(
        f"\nEpoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}"
    )

torch.save(model.state_dict(), os.path.join(BASE_DIR, "../model/anime_recommender.pt"))
