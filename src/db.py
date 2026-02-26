"""Простая база SQLite для истории проверок."""

import sqlite3
from datetime import datetime

from config import DB_PATH


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            url TEXT,
            is_dangerous INTEGER,
            confidence REAL,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_check(user_id: int, url: str, is_dangerous: bool, confidence: float):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO checks (user_id, url, is_dangerous, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, url, int(is_dangerous), float(confidence), datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()
