"""Настройки проекта."""

import os
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DB_PATH = BASE_DIR / "data" / "bot_history.db"

for folder in (DATA_DIR, MODELS_DIR, LOGS_DIR):
    folder.mkdir(exist_ok=True)


class ModelConfig:
    VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!'()*+,;=%"
    VOCAB_SIZE = len(VOCAB) + 2

    PAD_IDX = 0
    UNK_IDX = 1

    MAX_URL_LENGTH = 200
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3

    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    EPOCHS = 15
    TRAIN_SPLIT = 0.8


BOT_TOKEN = os.getenv("BOT_TOKEN")
