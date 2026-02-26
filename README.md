# Phishing Bot (Индивидуальный проект, 10 класс)


## Что есть в проекте

- `src/bot.py` — запуск Telegram-бота
- `src/inference.py` — проверка ссылки моделью
- `src/train.py` — обучение модели
- `src/db.py` — сохранение истории проверок в SQLite
- `src/preprocessing.py` — подготовка данных
- `src/models.py` — архитектуры моделей (MLP и LSTM)
- `data/malicious_phish.csv` — датасет
- `models/lstm_best.pt` — обученная модель
- `models/plots/` — графики для отчёта
- `logs/` — логи обучения


## Что показывает бот

- результат: `БЕЗОПАСНО` или `ОПАСНО`
- уверенность модели
- уровень риска
- найденные подозрительные признаки (если есть)

## Проверка для защиты

Примеры безопасных ссылок:
- `https://google.com`
- `https://github.com`

Примеры подозрительных:
- `http://secure-login-google.example`
- `http://192.168.10.15/login`
