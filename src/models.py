"""Модели для классификации ссылок."""

import torch
import torch.nn as nn

from config import ModelConfig


class PhishingMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=ModelConfig.VOCAB_SIZE,
            embedding_dim=ModelConfig.EMBEDDING_DIM,
            padding_idx=ModelConfig.PAD_IDX,
        )

        input_size = ModelConfig.MAX_URL_LENGTH * ModelConfig.EMBEDDING_DIM
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(ModelConfig.DROPOUT),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(ModelConfig.DROPOUT),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.layers(x)
        return x.squeeze(-1)


class PhishingLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=ModelConfig.VOCAB_SIZE,
            embedding_dim=ModelConfig.EMBEDDING_DIM,
            padding_idx=ModelConfig.PAD_IDX,
        )

        self.lstm = nn.LSTM(
            input_size=ModelConfig.EMBEDDING_DIM,
            hidden_size=ModelConfig.HIDDEN_DIM,
            num_layers=ModelConfig.NUM_LAYERS,
            batch_first=True,
            dropout=ModelConfig.DROPOUT if ModelConfig.NUM_LAYERS > 1 else 0,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(ModelConfig.DROPOUT),
            nn.Linear(ModelConfig.HIDDEN_DIM * 2, 128),
            nn.ReLU(),
            nn.Dropout(ModelConfig.DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(ModelConfig.DROPOUT),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)

        # Берём последнее состояние из двух направлений рекуррентного слоя.
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)

        output = self.classifier(combined)
        return output.squeeze(-1)


def get_model(model_type: str = "lstm"):
    if model_type.lower() == "lstm":
        return PhishingLSTM()
    if model_type.lower() == "mlp":
        return PhishingMLP()
    raise ValueError(f"Неизвестный тип модели: {model_type}")


def count_parameters(model) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
