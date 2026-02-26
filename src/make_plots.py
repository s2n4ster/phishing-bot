"""
Простые графики для проекта.
Запуск:
python make_plots.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "malicious_phish.csv"
PLOTS_DIR = BASE_DIR / "models" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_dataset_distribution(df: pd.DataFrame):
    counts = df["type"].value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)
    plt.title("Распределение классов в датасете")
    plt.xlabel("Класс")
    plt.ylabel("Количество URL")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "dataset_distribution_simple.png", dpi=150)
    plt.close()



def plot_binary_distribution(df: pd.DataFrame):
    binary = df["type"].apply(lambda x: "benign" if x == "benign" else "dangerous")
    counts = binary.value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("Бинарная разметка: безопасные/опасные")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "binary_distribution_simple.png", dpi=150)
    plt.close()



def plot_architecture_text():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    text = (
        "Архитектура модели (упрощённо):\n\n"
        "URL -> Embedding -> BiLSTM -> Dense -> Sigmoid\n\n"
        "Выход: вероятность, что ссылка опасная"
    )

    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "architecture_simple.png", dpi=150)
    plt.close()



def main():
    if not DATA_PATH.exists():
        print(f"Файл не найден: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    if "url" not in df.columns or "type" not in df.columns:
        print("Ожидались колонки: url,type")
        return

    plot_dataset_distribution(df)
    plot_binary_distribution(df)
    plot_architecture_text()

    print(f"Графики сохранены в: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
