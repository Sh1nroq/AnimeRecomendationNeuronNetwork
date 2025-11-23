import torch
import random

class MyAnimeDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        # self.full_text = pd.read_parquet(filepath_parquet)
        self.full_text = data
        self.label = self.full_text["label"].tolist()
        synopsis_text_1 = self.full_text["text1"].tolist()
        synopsis_text_2 = self.full_text["text2"].tolist()

        self.tokenized_text_1 = tokenizer(
            synopsis_text_1, truncation=True, padding="max_length", return_tensors="pt"
        )
        self.tokenized_text_2 = tokenizer(
            synopsis_text_2, truncation=True, padding="max_length", return_tensors="pt"
        )

    def __getitem__(self, index):
        item_1 = {k: v[index] for k, v in self.tokenized_text_1.items()}
        item_2 = {k: v[index] for k, v in self.tokenized_text_2.items()}
        label = torch.tensor(self.label[index], dtype=torch.long)
        return item_1, item_2, label

    def __len__(self):
        return len(self.label)