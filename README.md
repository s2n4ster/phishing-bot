# Phishing Bot (10 класс)

Телеграм-бот на Python, который проверяет ссылки и показывает признаки фишинга.

## Что в репозитории

- `src/` — код бота и модели
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

Нужно задать переменную окружения:

```env
BOT_TOKEN=ваш_токен_бота
```

## Railway

1. Создать новый проект в Railway и подключить этот GitHub-репозиторий.
2. В `Variables` добавить:
   - `BOT_TOKEN` = токен вашего Telegram-бота.
3. Нажать `Deploy`.

Railway запускает команду из `railway.json`:

```bash
python src/bot.py
```

## Важно

Бот — учебный проект. Он помогает оценить ссылку, но не заменяет полную защиту.
