"""Общие утилиты: конфигурация из .env, HTTP-клиент OpenAI-совместимого API, дисковый кэш.

Ключ читается только из окружения/.env внутри процесса и нигде не печатается:
в сообщениях об ошибках он подменяется на ***.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
USER_AGENT = "rag-eval-client/0.1"  # без своего User-Agent Cloudflare блокирует запросы (код 1010)


def load_env(path: Path | None = None) -> None:
    """Минимальный разбор .env (без внешних зависимостей). Уже заданные переменные не трогаем."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env()


def load_eval_set(include_excluded: bool = False) -> list[dict]:
    """Оценочный набор из data/eval_set.csv. По умолчанию без вопросов, исключённых ручной проверкой (excluded = True)."""
    import csv

    with open(DATA_DIR / "eval_set.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["subset"] = r.get("subset") or "formal"
        r["gold_article_id_2"] = r.get("gold_article_id_2") or ""
        r["excluded"] = r.get("excluded") or "False"
    return rows if include_excluded else [r for r in rows if r["excluded"] != "True"]


def cfg(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None or value == "":
        if default is None:
            raise RuntimeError(f"Не задана переменная {name} (см. .env.example)")
        return default
    return value


def _redact(text: str) -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    return text.replace(key, "***") if key else text


def _hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _post(path: str, body: dict) -> dict:
    base = cfg("OPENAI_BASE_URL").rstrip("/")
    timeout = float(cfg("REQUEST_TIMEOUT", "60"))
    retries = int(cfg("MAX_RETRIES", "5"))
    headers = {
        "Authorization": f"Bearer {cfg('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    last = ""
    for attempt in range(retries):
        try:
            resp = requests.post(base + path, json=body, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            last = f"HTTP {resp.status_code}: {_redact(resp.text[:300])}"
            if resp.status_code not in (408, 409, 425, 429, 500, 502, 503, 504):
                break
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}: {_redact(str(exc))[:200]}"
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Запрос {path} не удался: {last}")


def cached_post(path: str, body: dict, tag: str | None = None) -> dict:
    """POST с дисковым кэшем по хэшу (путь, тело, метка). Ключ API в хэш не входит.

    Метка tag нужна для независимых повторных прогонов с одинаковым телом запроса (например, run-1, run-2):
    в запрос она не отправляется, а только отличает записи кэша. Без метки хэш совпадает со старым форматом.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    key = {"path": path, "body": body}
    if tag:
        key["tag"] = tag
    file = CACHE_DIR / f"{_hash(key)}.json"
    if file.exists():
        return json.loads(file.read_text(encoding="utf-8"))
    data = _post(path, body)
    file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def chat(model: str, messages: list[dict], max_tokens: int = 4000, temperature: float = 0.0, tag: str | None = None) -> dict:
    """Запрос к чат-модели. Рассуждения (reasoning) отделяются от итогового ответа."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    data = cached_post("/chat/completions", body, tag=tag)
    message = data["choices"][0]["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if "<think>" in content:
        reasoning = reasoning or "".join(THINK_RE.findall(content))
        content = THINK_RE.sub("", content)
    if "</think>" in content:  # открывающий тег мог быть срезан сервером
        reasoning = reasoning or content.split("</think>")[0]
        content = content.split("</think>")[-1]
    return {
        "content": content.strip(),
        "reasoning": reasoning.strip(),
        "finish_reason": data["choices"][0].get("finish_reason"),
        "usage": data.get("usage", {}),
    }


def embed_remote(model: str, texts: list[str], input_type: str = "passage", batch: int = 32, workers: int = 3) -> np.ndarray:
    """Эмбеддинги через POST /embeddings. Результат L2-нормирован.

    Модели Nemotron асимметричны: документы кодируются с input_type="passage", запросы с "query"
    (проверено на пробном запросе: векторы для разных input_type различаются).
    """
    batches = [texts[i : i + batch] for i in range(0, len(texts), batch)]

    def run(chunk: list[str]) -> list[list[float]]:
        body = {"model": model, "input": chunk, "input_type": input_type, "truncate": "END"}
        data = cached_post("/embeddings", body)
        items = sorted(data["data"], key=lambda x: x["index"])
        return [x["embedding"] for x in items]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(run, batches))
    mat = np.array([v for part in parts for v in part], dtype=np.float32)
    return mat / np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-9)


def rerank_remote(model: str, query: str, docs: list[str]) -> list[float]:
    """Оценки релевантности документов запросу, в порядке исходного списка."""
    body = {"model": model, "query": query, "documents": docs}
    data = cached_post("/rerank", body)
    scores = [0.0] * len(docs)
    for item in data["results"]:
        scores[item["index"]] = item["relevance_score"]
    return scores


def parallel_map(fn, items, workers: int = 3):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))
