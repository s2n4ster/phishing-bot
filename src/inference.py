"""Проверка ссылок через обученную модель."""

import re

import torch

from config import MODELS_DIR
from models import get_model
from preprocessing import URLTokenizer


SAFE_DOMAINS = {
    "google.com",
    "youtube.com",
    "github.com",
    "wikipedia.org",
    "vk.com",
    "t.me",
    "telegram.org",
    "yandex.ru",
    "mail.ru",
    "sberbank.ru",
    "gosuslugi.ru",
    "ozon.ru",
    "max.ru",
    "wildberries.ru",
}


class URLChecker:
    def __init__(self, model_type: str = "lstm"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model_path = MODELS_DIR / f"{model_type}_best.pt"

        self.tokenizer = URLTokenizer()
        self.model = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Модель не найдена: {self.model_path}\n"
                "Сначала обучите модель: python train.py"
            )

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model_type = checkpoint.get("model_type", self.model_type)

        self.model = get_model(model_type).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"Модель загружена: {self.model_path}")
        print(f"Устройство: {self.device}")

    def _get_domain(self, url: str) -> str:
        value = url.lower().strip()

        for prefix in ("https://", "http://", "www."):
            if value.startswith(prefix):
                value = value[len(prefix):]

        domain = value.split("/")[0]
        domain = domain.split(":")[0]
        return domain

    def check(self, url: str):
        if not isinstance(url, str) or not url.strip():
            raise ValueError("URL должен быть непустой строкой")
        analysis_result = self.analyze(url)
        return analysis_result["is_dangerous"], analysis_result["confidence"]

    def analyze(self, url: str) -> dict:
        url_lower = url.lower()
        domain = self._get_domain(url)
        suspicious_signs = []

        if any(word in url_lower for word in ["login", "verify", "secure", "account", "update", "confirm"]):
            suspicious_signs.append("Подозрительные слова в адресе")

        if url_lower.count(".") > 4:
            suspicious_signs.append("Слишком много поддоменов")

        if len(url_lower) > 75:
            suspicious_signs.append("Очень длинная ссылка")

        if re.search(r"\d{1,3}(?:\.\d{1,3}){3}", url_lower):
            suspicious_signs.append("Вместо домена используется IP-адрес")

        if any(short in url_lower for short in ["bit.ly", "tinyurl", "t.co", "clck.ru"]):
            suspicious_signs.append("Используется сокращатель ссылок")

        if domain in SAFE_DOMAINS:
            confidence = 0.99 if not suspicious_signs else 0.85
            return {
                "url": url,
                "is_dangerous": False,
                "result_label": "БЕЗОПАСНО",
                "confidence": round(confidence, 3),
                "risk_level": "БЕЗОПАСНО",
                "suspicious_signs": suspicious_signs,
            }

        encoded = self.tokenizer.encode(url)
        input_tensor = torch.tensor([encoded], dtype=torch.long).to(self.device)

        with torch.no_grad():
            raw_probability = self.model(input_tensor).item()

        if raw_probability > 0.8 and len(url_lower) < 20 and not suspicious_signs:
            probability = 0.55 + (raw_probability * 0.1)
        elif raw_probability < 0.5 and suspicious_signs:
            probability = raw_probability + (len(suspicious_signs) * 0.15)
            probability = min(0.99, probability)
        else:
            probability = raw_probability

        if probability >= 0.55:
            is_dangerous = True
            confidence = probability
            result_label = "ОПАСНО"
            if confidence > 0.9:
                risk_level = "ВЫСОКИЙ РИСК"
            elif confidence > 0.7:
                risk_level = "СРЕДНИЙ РИСК"
            else:
                risk_level = "НИЗКИЙ РИСК"
        elif probability <= 0.45:
            is_dangerous = False
            confidence = 1 - probability
            result_label = "БЕЗОПАСНО"
            if confidence > 0.9:
                risk_level = "БЕЗОПАСНО"
            elif confidence > 0.7:
                risk_level = "ВЕРОЯТНО БЕЗОПАСНО"
            else:
                risk_level = "ТРЕБУЕТ ПРОВЕРКИ"
        else:
            is_dangerous = False
            confidence = abs(probability - 0.5) * 2
            result_label = "ТРЕБУЕТ ПРОВЕРКИ"
            risk_level = "ТРЕБУЕТ ПРОВЕРКИ"

        return {
            "url": url,
            "is_dangerous": is_dangerous,
            "result_label": result_label,
            "confidence": round(confidence, 3),
            "risk_level": risk_level,
            "suspicious_signs": suspicious_signs,
        }


_checker = None


def get_checker(model_type: str = "lstm") -> URLChecker:
    global _checker
    if _checker is None:
        _checker = URLChecker(model_type=model_type)
    return _checker
