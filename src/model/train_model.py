import os.path
from torch.amp import autocast, GradScaler
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
filename = os.path.join(BASE_DIR, "../../data/processed/anime_pairs_augmented.parquet")

df = pd.read_parquet(filename)

train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class AnimeRecommender(nn.Module):
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

model = AnimeRecommender().to(device)

for name, param in model.bert_model.named_parameters():
    if "encoder.layer" in name:
        param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

train_dataset = MyAnimeDataset(train_df, tokenizer)
val_dataset = MyAnimeDataset(val_df, tokenizer)

batch_size = 16

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

learning_rate = 3e-5
weight_decay = 0.01
num_epochs = 5
margin = 0.5
max_length = 256
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)

total_steps = len(train_dataloader) * num_epochs
warmup_steps = int(0.2 * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

def fine_tuning(model, train_dataloader, optimizer, scheduler, device, margin=0.5, accum_steps=4, epoch=0):
    num_layers = len(model.bert_model.encoder.layer)
    layers_to_unfreeze = min(2 * (epoch + 1), num_layers)
    for i, layer in enumerate(model.bert_model.encoder.layer):
        requires_grad = i >= num_layers - layers_to_unfreeze
        for param in layer.parameters():
            param.requires_grad = requires_grad

    criterion = nn.CosineEmbeddingLoss(margin=margin)
    model.train()
    scaler = GradScaler()
    optimizer.zero_grad()
    total_loss = 0.0
    num_batches = 0

    for i, (item_1, item_2, y) in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
        item_1, item_2, y = move_to_device(item_1, item_2, y, device)

        with autocast(device_type="cuda"):
            token_text_1 = model(**item_1)
            token_text_2 = model(**item_2)
            loss = criterion(token_text_1, token_text_2, y)

        loss = loss / accum_steps
        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(train_dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * accum_steps
        num_batches += 1
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_dataloader, margin):
    model.eval()
    total_loss = 0
    num_batches = 0
    criterion = nn.CosineEmbeddingLoss(margin=margin)

    with no_grad():
        for item_1, item_2, y in tqdm(val_dataloader):
            item_1, item_2, y = move_to_device(item_1, item_2, y, device)

            token_text_1 = model(**item_1)
            token_text_2 = model(**item_2)

            cosine_similarity = F.cosine_similarity(token_text_1, token_text_2, dim=-1)
            loss = criterion(token_text_1, token_text_2, y)

            total_loss += loss.mean().item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        return avg_loss


for epoch in range(num_epochs):
    if epoch % 2 == 0 and epoch > 0:
        num_to_unfreeze = epoch // 2 * 2
        print(f"Размораживаем верхние {num_to_unfreeze} слоя(ев)")

    train_loss = fine_tuning(model, train_dataloader, optimizer, scheduler, device, margin, epoch=epoch)
    val_loss = validate(model, val_dataloader, margin)
    print(
        f"\nEpoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}"
    )

torch.save(model.state_dict(), os.path.join(BASE_DIR, "../../data/embeddings/anime_recommender_alpha.pt"))
