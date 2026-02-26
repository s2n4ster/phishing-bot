# Phishing Bot (10 класс)

Телеграм-бот на Python для проверки ссылок на фишинг.

## Состав проекта

- `src/bot.py` — запуск бота
- `src/inference.py` — анализ ссылки моделью
- `src/train.py` — обучение модели
- `src/db.py` — история проверок (SQLite)
- `models/lstm_best.pt` — обученная модель
- `railway.json` — конфиг деплоя Railway
- `requirements.txt` — зависимости для Railway (runtime)
- `requirements-train.txt` — зависимости для локального обучения

## Локальный запуск бота

```bash
pip install -r requirements.txt
python src/bot.py
```

Создай `.env` в корне проекта:

```env
BOT_TOKEN=your_telegram_bot_token
```

## Локальное обучение модели

```bash
pip install -r requirements-train.txt
python src/train.py --model lstm --dataset ../data/malicious_phish.csv
```

## Railway

1. Подключи GitHub-репозиторий в Railway.
2. Добавь переменную окружения `BOT_TOKEN`.
3. Нажми Deploy.

Команда запуска в Railway:

```bash
python src/bot.py
```

## Важно

Бот — учебный проект и не дает 100% гарантии безопасности.
