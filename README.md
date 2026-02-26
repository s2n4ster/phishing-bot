# Phishing Bot (10 класс)

Телеграм-бот на Python для проверки ссылок на фишинг.

## Состав проекта

- `src/bot.py` — запуск бота
- `src/inference.py` — анализ ссылки моделью
- `src/train.py` — обучение модели
- `src/db.py` — история проверок (SQLite)
- `models/lstm_best.pt` — обученная модель
- `models/plots/` — графики для защиты
- `logs/` — логи обучения
- `requirements.txt` — зависимости
- `railway.json` — конфиг деплоя Railway

## Локальный запуск

```bash
pip install -r requirements.txt
python src/bot.py
```

Создай `.env` в корне проекта:

```env
BOT_TOKEN=your_telegram_bot_token
```

## Railway

1. Подключи GitHub-репозиторий в Railway.
2. Добавь переменную окружения `BOT_TOKEN`.
3. Нажми Deploy.

Команда запуска в Railway:

```bash
python src/bot.py
```

## Примеры для проверки

Безопасные:
- `https://google.com`
- `https://github.com`

Подозрительные:
- `http://secure-login-google.example`
- `http://192.168.10.15/login`

## Важно

Бот — учебный проект и не дает 100% гарантии безопасности.
