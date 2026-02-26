"""
Простое сравнение моделей MLP и LSTM.
Запуск:
python compare_models.py
"""

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, random_split

from config import DATA_DIR, MODELS_DIR, ModelConfig
from models import get_model
from preprocessing import PhishingDataset, load_dataset


def evaluate_model(model_type: str, loader: DataLoader, device):
    model_path = MODELS_DIR / f"{model_type}_best.pt"
    if not model_path.exists():
        print(f"Модель не найдена: {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = get_model(model_type).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for urls, labels in loader:
            urls = urls.to(device)
            labels = labels.to(device)

            outputs = model(urls)
            preds = (outputs > 0.5).float()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def save_bar_chart(results: dict):
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = range(len(metrics))

    mlp_vals = [results["mlp"][m] for m in metrics]
    lstm_vals = [results["lstm"][m] for m in metrics]

    plt.figure(figsize=(9, 5))
    plt.bar([i - 0.2 for i in x], mlp_vals, width=0.4, label="MLP")
    plt.bar([i + 0.2 for i in x], lstm_vals, width=0.4, label="LSTM")

    plt.xticks(list(x), [m.upper() for m in metrics])
    plt.ylim(0, 1)
    plt.ylabel("Значение")
    plt.title("Сравнение моделей")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    out_dir = MODELS_DIR / "comparison"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "metrics_comparison.png"

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"График сохранён: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    dataset_path = DATA_DIR / "malicious_phish.csv"
    urls, labels = load_dataset(str(dataset_path))
    dataset = PhishingDataset(urls, labels)

    train_size = int(len(dataset) * ModelConfig.TRAIN_SPLIT)
    val_size = len(dataset) - train_size

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    val_loader = DataLoader(val_dataset, batch_size=ModelConfig.BATCH_SIZE)

    results = {}
    for model_type in ["mlp", "lstm"]:
        print(f"\nПроверяю {model_type.upper()}...")
        metrics = evaluate_model(model_type, val_loader, device)
        if metrics is not None:
            results[model_type] = metrics
            print(
                f"accuracy={metrics['accuracy']:.4f}, "
                f"precision={metrics['precision']:.4f}, "
                f"recall={metrics['recall']:.4f}, "
                f"f1={metrics['f1']:.4f}"
            )

    if "mlp" in results and "lstm" in results:
        save_bar_chart(results)
    else:
        print("Не удалось сравнить обе модели. Сначала обучите обе.")


if __name__ == "__main__":
    main()
