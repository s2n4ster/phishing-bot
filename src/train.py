"""
Простое обучение модели.
Запуск:
python train.py --model lstm --dataset ../data/malicious_phish.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, random_split

from config import DATA_DIR, MODELS_DIR, ModelConfig
from models import count_parameters, get_model
from preprocessing import PhishingDataset, load_dataset


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for urls, labels in dataloader:
        urls = urls.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(urls)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for urls, labels in dataloader:
            urls = urls.to(device)
            labels = labels.to(device)

            outputs = model(urls)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
    return metrics, all_labels, all_preds


def save_confusion_matrix(labels, preds, save_path: Path):
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Безопасный", "Опасный"])
    ax.set_yticklabels(["Безопасный", "Опасный"])
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Истинное значение")
    ax.set_title("Матрица ошибок")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_training_plot(history, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], marker="o", label="Train")
    axes[0].plot(history["val_loss"], marker="o", label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Эпоха")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], marker="o", label="Train")
    axes[1].plot(history["val_acc"], marker="o", label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Эпоха")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    urls, labels = load_dataset(args.dataset)
    dataset = PhishingDataset(urls, labels)

    train_size = int(len(dataset) * ModelConfig.TRAIN_SPLIT)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=ModelConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=ModelConfig.BATCH_SIZE)

    model = get_model(args.model).to(device)
    print(f"Модель: {args.model.upper()}")
    print(f"Параметров: {count_parameters(model):,}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=ModelConfig.LEARNING_RATE)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_f1 = 0.0
    best_path = MODELS_DIR / f"{args.model}_best.pt"

    for epoch in range(1, ModelConfig.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_labels, val_preds = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"Эпоха {epoch}/{ModelConfig.EPOCHS} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.4f}, f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_type": args.model,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
            print(f"Новая лучшая модель: F1={best_f1:.4f}")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics, final_labels, final_preds = evaluate(model, val_loader, criterion, device)

    print("\nИтоговые метрики:")
    print(f"Accuracy : {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall   : {final_metrics['recall']:.4f}")
    print(f"F1       : {final_metrics['f1']:.4f}")

    plots_dir = MODELS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    save_confusion_matrix(final_labels, final_preds, plots_dir / f"{args.model}_confusion_matrix.png")
    save_training_plot(history, plots_dir / f"{args.model}_training_history.png")

    print(f"\nМодель сохранена: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение модели")
    parser.add_argument("--model", default="lstm", choices=["mlp", "lstm"], help="Тип модели")
    parser.add_argument(
        "--dataset",
        default=str(DATA_DIR / "malicious_phish.csv"),
        help="Путь к датасету",
    )

    args = parser.parse_args()
    main(args)
