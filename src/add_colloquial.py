"""Добавляет разговорное подмножество в data/eval_set.csv (колонка subset = colloquial).

Существующие строки (subset = formal) не меняются и не переупорядочиваются, чтобы кэш результатов оставался валиден.
Скрипт идемпотентен: ранее добавленные разговорные строки удаляются и дописываются заново.
Проверка: каждая цитата из evidence должна дословно входить в текст эталонных статей.
Запуск: python src/add_colloquial.py
"""
from __future__ import annotations

import csv
import re

from chunking import load_corpus
from colloquial_questions import QUESTIONS
from common import DATA_DIR

FIELDS = ["qid", "question", "gold_answer", "gold_article_id", "gold_article_id_2", "evidence", "qtype", "answerable", "source",
          "subset", "verified_by_human", "excluded", "exclusion_reason"]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ").replace("«", '"').replace("»", '"')).strip().lower()


def main() -> None:
    corpus = {a["article_id"]: a for a in load_corpus()}
    with open(DATA_DIR / "eval_set.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)]
    for r in rows:
        r.setdefault("subset", "formal")
        r["subset"] = r.get("subset") or "formal"
        r.setdefault("gold_article_id_2", "")
        r["gold_article_id_2"] = r.get("gold_article_id_2") or ""
    rows = [r for r in rows if r["subset"] != "colloquial"]
    by_qid = {r["qid"]: r for r in rows}
    next_id = max(int(r["qid"][1:]) for r in rows) + 1
    problems = []
    for item in QUESTIONS:
        evidence = item.get("evidence") or by_qid[item["evidence_from"]]["evidence"].split(" || ")
        text = norm(" ".join(corpus[a]["text"] for a in (item["gold"], item["gold2"]) if a))
        for quote in evidence:
            if norm(quote).strip(" .;:") not in text:
                problems.append((item["question"][:50], quote[:60]))
        rows.append({"qid": f"q{next_id:02d}", "question": item["question"], "gold_answer": item["answer"], "gold_article_id": item["gold"],
                     "gold_article_id_2": item["gold2"], "evidence": " || ".join(evidence), "qtype": item["qtype"], "answerable": "True",
                     "source": "llm_written_colloquial_assistant", "subset": "colloquial", "verified_by_human": "False",
                     "excluded": "False", "exclusion_reason": ""})
        next_id += 1
    if problems:
        raise SystemExit(f"Цитаты не найдены в статьях: {problems}")
    with open(DATA_DIR / "eval_set.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
    colloq = [r for r in rows if r["subset"] == "colloquial"]
    two = sum(1 for r in colloq if r["gold_article_id_2"])
    print(f"всего вопросов {len(rows)}: formal {len(rows) - len(colloq)}, colloquial {len(colloq)} (из них с двумя эталонными статьями: {two})")


if __name__ == "__main__":
    main()
