"""Телеграм-бот для проверки ссылок."""

import asyncio
import logging
import os
import re

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

from config import BOT_TOKEN
from db import init_db, save_check
from inference import URLChecker

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = Router()
checker = None


def find_urls(text: str) -> list[str]:
    urls = re.findall(r"https?://\S+", text)
    if urls:
        return urls

    domains = re.findall(r"\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\S*", text)
    return [f"http://{d}" for d in domains]


def format_result(data: dict) -> str:
    label = data.get("result_label")
    if not label:
        label = "ОПАСНО" if data["is_dangerous"] else "БЕЗОПАСНО"

    text = (
        f"Проверка ссылки:\n"
        f"{data['url']}\n\n"
        f"Результат: {label}\n"
        f"Уверенность: {data['confidence']:.1%}\n"
        f"Уровень риска: {data['risk_level']}"
    )

    if data["suspicious_signs"]:
        text += "\n\nПодозрительные признаки:"
        for sign in data["suspicious_signs"]:
            text += f"\n- {sign}"

    return text


@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Я бот для проверки ссылок.\n"
        "Отправь ссылку или напиши:\n"
        "/check https://example.com\n\n"
        "Команды: /help, /about"
    )


@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "Как пользоваться:\n"
        "1) Отправь ссылку в сообщении\n"
        "2) Или команда: /check ссылка\n\n"
        "Важно: бот помогает, но не дает 100% гарантию."
    )


@router.message(Command("about"))
async def cmd_about(message: Message):
    await message.answer(
        "Проект: Telegram-бот для проверки ссылок с помощью нейросети.\n"
        "Автор: Марченко Марк, 10 класс."
    )


@router.message(Command("check"))
async def cmd_check(message: Message):
    text = message.text.replace("/check", "", 1).strip()

    if not text:
        await message.answer("После /check укажи ссылку. Пример: /check https://example.com")
        return

    urls = find_urls(text)
    if not urls:
        await message.answer("Не нашёл ссылку в тексте.")
        return

    await check_and_send(message, urls[0])


@router.message(F.text)
async def handle_text(message: Message):
    urls = find_urls(message.text)
    if not urls:
        await message.answer("Я не увидел ссылку. Отправь URL в сообщении.")
        return

    await check_and_send(message, urls[0])


async def check_and_send(message: Message, url: str):
    global checker

    if checker is None:
        await message.answer("Модель ещё загружается. Попробуйте через пару секунд.")
        return

    try:
        data = checker.analyze(url)
        user_id = message.from_user.id if message.from_user else 0
        save_check(user_id, data["url"], data["is_dangerous"], data["confidence"])
        await message.answer(format_result(data))
    except Exception as exc:
        logger.error("Ошибка при проверке: %s", exc)
        await message.answer("Произошла ошибка при проверке ссылки.")


async def main():
    global checker

    token = os.getenv("BOT_TOKEN", BOT_TOKEN)
    if not token:
        logger.error("BOT_TOKEN не найден. Добавьте его в .env")
        return

    bot = Bot(token=token)
    dp = Dispatcher()
    dp.include_router(router)
    init_db()

    try:
        checker = URLChecker(model_type="lstm")
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.error("Сначала обучите модель: python train.py")
        return

    logger.info("Бот запущен")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
