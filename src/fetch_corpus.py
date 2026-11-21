"""Сбор корпуса: Закон РФ «О защите прав потребителей» и ГК РФ ст. 492-505 из Викитеки.

Результат: data/corpus.jsonl (одна строка = одна статья) и data/texts/*.txt для чтения глазами.
Запуск: python src/fetch_corpus.py
"""
from __future__ import annotations

import json
import re
import time
from datetime import date

import requests
from bs4 import BeautifulSoup

from common import DATA_DIR, USER_AGENT

API = "https://ru.wikisource.org/w/api.php"
SOURCES = {
    "zpp": {
        "page": "Закон РФ от 07.02.1992 № 2300-I",
        "title": "Закон РФ «О защите прав потребителей» от 07.02.1992 № 2300-1",
        "short": "ЗоЗПП",
        "articles": None,  # все статьи
    },
    "gk": {
        "page": "Гражданский кодекс РФ/Глава 30",
        "title": "Гражданский кодекс РФ (часть вторая), глава 30, § 2 «Розничная купля-продажа»",
        "short": "ГК РФ",
        "articles": range(492, 506),  # статьи 492-505
    },
}
ARTICLE_RE = re.compile(r"^Статья\s+(\d+(?:\.\d+)?)\.\s*(.*)$")
PART_RE = re.compile(r"^(\d+(?:\.\d+)?)\.\s+")


def clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\[\s*править\s*\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_page(page: str) -> tuple[str, int]:
    params = {"action": "parse", "page": page, "prop": "text|revid", "format": "json", "formatversion": 2}
    for attempt in range(5):
        try:
            resp = requests.get(API, params=params, headers={"User-Agent": USER_AGENT}, timeout=120)
            resp.raise_for_status()
            data = resp.json()["parse"]
            return data["text"], data["revid"]
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Не удалось загрузить {page}")


def revision_timestamp(revid: int) -> str:
    params = {"action": "query", "prop": "revisions", "revids": revid, "rvprop": "timestamp", "format": "json"}
    data = requests.get(API, params=params, headers={"User-Agent": USER_AGENT}, timeout=60).json()
    page = next(iter(data["query"]["pages"].values()))
    return page["revisions"][0]["timestamp"]


def parse_articles(html: str, doc: str, wanted) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    body = max(soup.find("div", class_="mw-parser-output").find_all("div", class_="text"), key=lambda x: len(x.get_text()))
    articles: list[dict] = []
    chapter = ""
    current = None
    for el in body.children:
        if not getattr(el, "name", None):
            continue
        classes = el.get("class", [])
        if el.name == "div" and "mw-heading" in classes:
            text = clean(el.get_text(" "))
            match = ARTICLE_RE.match(text)
            if match:
                current = {"doc": doc, "article_no": match.group(1), "title": match.group(2), "chapter": chapter, "parts": []}
                articles.append(current)
            else:
                chapter = text
                current = None
        elif el.name == "p" and current is not None:
            for sup in el.find_all("sup"):
                sup.decompose()
            text = clean(el.get_text(" "))
            if text:
                current["parts"].append(text)
    result = []
    for art in articles:
        number = art["article_no"]
        if wanted is not None and not (number.isdigit() and int(number) in wanted):
            continue
        text = "\n".join(art["parts"])
        if len(text) < 60 or text.lower().startswith("утратила силу"):
            continue  # утратившие силу статьи без содержания
        art["article_id"] = f"{doc}:{number}"
        art["text"] = text
        art["parts"] = split_parts(art["parts"])
        result.append(art)
    return result


def split_parts(paragraphs: list[str]) -> list[dict]:
    """Группируем абзацы по пунктам статьи: новый пункт начинается с «N. »."""
    parts: list[dict] = []
    for par in paragraphs:
        match = PART_RE.match(par)
        if match or not parts:
            parts.append({"num": match.group(1) if match else "0", "text": par})
        else:
            parts[-1]["text"] += "\n" + par
    return parts


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "texts").mkdir(exist_ok=True)
    corpus: list[dict] = []
    meta: dict = {"retrieved": date.today().isoformat(), "sources": {}}
    for doc, src in SOURCES.items():
        html, revid = fetch_page(src["page"])
        arts = parse_articles(html, doc, src["articles"])
        for art in arts:
            art["doc_title"] = src["title"]
            art["doc_short"] = src["short"]
        corpus += arts
        meta["sources"][doc] = {"wikisource_page": src["page"], "revid": revid, "revision_timestamp": revision_timestamp(revid), "articles": len(arts)}
        lines = [f"{src['title']}\n"]
        for art in arts:
            lines.append(f"Статья {art['article_no']}. {art['title']}\n{art['text']}\n")
        (DATA_DIR / "texts" / f"{doc}.txt").write_text("\n".join(lines), encoding="utf-8")
    with open(DATA_DIR / "corpus.jsonl", "w", encoding="utf-8") as fh:
        for art in corpus:
            fh.write(json.dumps(art, ensure_ascii=False) + "\n")
    meta["total_articles"] = len(corpus)
    meta["total_chars"] = sum(len(a["text"]) for a in corpus)
    (DATA_DIR / "corpus_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
