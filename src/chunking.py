"""Варианты разбиения статей на чанки.

Каждый чанк начинается с заголовка «ЗоЗПП, ст. 18. Название статьи», чтобы фрагмент оставался
понятным вне контекста. article_id сохраняется для оценки на уровне статьи.
"""
from __future__ import annotations

import json
import re

from common import DATA_DIR

VARIANTS = ("article", "part", "win500", "win1000")


def load_corpus() -> list[dict]:
    with open(DATA_DIR / "corpus.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def header(art: dict, part: str | None = None) -> str:
    base = f"{art['doc_short']}, ст. {art['article_no']}. {art['title']}"
    return f"{base} (п. {part})" if part else base


def split_window(text: str, size: int, overlap: int) -> list[str]:
    """Окна фиксированной длины по символам; границы сдвигаются к ближайшему пробелу."""
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            end = space if space > start else end
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


def make_chunks(corpus: list[dict], variant: str) -> list[dict]:
    chunks = []
    for art in corpus:
        if variant == "article":
            texts = [(None, art["text"])]
        elif variant == "part":
            # короткие пункты склеиваем с соседними, чтобы не получать чанки из одной строки
            texts, buf, num = [], "", None
            for part in art["parts"]:
                if num is None:
                    num = part["num"]
                buf = (buf + "\n" + part["text"]).strip()
                if len(buf) >= 250:
                    texts.append((num, buf))
                    buf, num = "", None
            if buf:
                texts.append((num, buf))
        elif variant in ("win500", "win1000"):
            size = int(variant[3:])
            texts = [(None, t) for t in split_window(art["text"], size, size // 5)]
        else:
            raise ValueError(variant)
        for i, (part, text) in enumerate(texts):
            chunks.append(
                {
                    "chunk_id": f"{art['article_id']}#{variant}#{i}",
                    "article_id": art["article_id"],
                    "variant": variant,
                    "text": f"{header(art, part if variant == 'part' else None)}\n{text}",
                }
            )
    return chunks


TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.I)


def tokenize(text: str, prefix: int | None = None) -> list[str]:
    """Токены нижнего регистра; prefix=5 даёт грубую «основу» слова (обрезка до 5 букв)."""
    tokens = [t.lower().replace("ё", "е") for t in TOKEN_RE.findall(text)]
    return [t[:prefix] if prefix else t for t in tokens]
