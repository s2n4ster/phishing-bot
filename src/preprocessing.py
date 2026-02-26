"""Подготовка данных ссылок для модели."""

import pandas as pd
import torch
from torch.utils.data import Dataset

from config import ModelConfig


class URLTokenizer:
    def __init__(self):
        self.char_to_idx = {char: i + 2 for i, char in enumerate(ModelConfig.VOCAB)}

    def encode(self, url: str) -> list:
        value = url.lower().strip()

        for prefix in ("https://", "http://", "www."):
            if value.startswith(prefix):
                value = value[len(prefix):]
                break

        result = []
        for char in value[: ModelConfig.MAX_URL_LENGTH]:
            result.append(self.char_to_idx.get(char, ModelConfig.UNK_IDX))

        while len(result) < ModelConfig.MAX_URL_LENGTH:
            result.append(ModelConfig.PAD_IDX)

        return result


def load_dataset(filepath: str):
    print(f"Загружаю датасет: {filepath}")
    df = pd.read_csv(filepath)

    urls = df["url"].tolist()
    labels = [0 if t == "benign" else 1 for t in df["type"].tolist()]

    print(f"Всего URL: {len(urls)}")
    print(f"Безопасных: {labels.count(0)}")
    print(f"Опасных: {labels.count(1)}")

    return urls, labels


class PhishingDataset(Dataset):
    def __init__(self, urls: list, labels: list):
        self.urls = urls
        self.labels = labels
        self.tokenizer = URLTokenizer()

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer.encode(self.urls[idx])
        label = self.labels[idx]

        url_tensor = torch.tensor(encoded, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return url_tensor, label_tensor
