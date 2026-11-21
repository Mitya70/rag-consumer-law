"""Выборка из 20 вопросов для ручной проверки: review/review_sample.md.

Проверяющий смотрит, верны ли вопрос и эталонный ответ относительно приведённого текста статьи, и отмечает
результат. После проверки значения verified_by_human в data/eval_set.csv нужно обновить вручную.
Запуск: python src/make_review_sample.py
"""
from __future__ import annotations

import csv
import random

from chunking import load_corpus
from common import DATA_DIR, ROOT

random.seed(11)


def main() -> None:
    if (ROOT / "review" / "human_verdicts.csv").exists():
        raise SystemExit("review/human_verdicts.csv уже есть: выборка проверена человеком, новая случайная выборка не создаётся, чтобы не затереть вердикты.")
    corpus = {a["article_id"]: a for a in load_corpus()}
    with open(DATA_DIR / "eval_set.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["subset"] = r.get("subset") or "formal"
    answerable = [r for r in rows if r["answerable"] == "True" and r["subset"] == "formal"]
    colloquial = [r for r in rows if r["subset"] == "colloquial"]
    oos = [r for r in rows if r["answerable"] != "True"]
    by_type: dict[str, list[dict]] = {}
    for r in answerable:
        by_type.setdefault(r["qtype"], []).append(r)
    picked: list[dict] = []
    # по одному вопросу каждого типа, затем добираем случайные до 12
    for qtype in sorted(by_type):
        picked.append(random.choice(by_type[qtype]))
    rest = [r for r in answerable if r not in picked]
    random.shuffle(rest)
    picked += rest[: 12 - len(picked)]
    picked.sort(key=lambda r: r["qid"])
    picked += random.sample(oos, 3)
    picked += sorted(random.sample(colloquial, 5), key=lambda r: r["qid"])

    out = ["# Выборка для ручной проверки оценочного набора", "",
           "Для каждого вопроса проверьте по тексту статьи: (1) вопрос понятен и по смыслу отвечается этой статьёй; "
           "(2) эталонный ответ верен и не содержит лишнего; (3) для вопросов «вне корпуса» в тексте статей действительно нет ответа. "
           "Отметьте результат и впишите правку, если нужна. Затем поставьте `verified_by_human = True` для проверенных вопросов "
           "в `data/eval_set.csv`.", "",
           "Состав: 12 формальных вопросов с ответом, 3 вопроса вне корпуса и 5 разговорных. Все разговорные вопросы написаны ассистентом (LLM), не автором проекта.",
           "Статус: **не проверено человеком** (0 из 20).", ""]
    for n, r in enumerate(picked, 1):
        out.append(f"## {n}. {r['qid']} ({r['subset']}, {r['qtype']})")
        out.append("")
        out.append(f"**Вопрос:** {r['question']}")
        out.append("")
        if r["answerable"] == "True":
            gold_ids = [a for a in (r["gold_article_id"], r.get("gold_article_id_2", "")) if a]
            arts = [corpus[a] for a in gold_ids]
            art = arts[0]
            out.append(f"**Эталонный ответ:** {r['gold_answer']}")
            out.append("")
            out.append("**Эталонные статьи:** " + "; ".join(f"{a['doc_short']}, ст. {a['article_no']}. {a['title']}" for a in arts))
            out.append("")
            out.append("**Цитаты, на которых основан эталон:**")
            for quote in r["evidence"].split(" || "):
                out.append(f"> {quote}")
            out.append("")
            snippet = art["text"][:900] + (" ..." if len(art["text"]) > 900 else "")
            out.append("<details><summary>Начало текста статьи</summary>")
            out.append("")
            out.append(snippet.replace("\n", "\n\n"))
            out.append("")
            out.append("</details>")
        else:
            out.append("**Ожидаемое поведение:** система должна отказаться отвечать (в корпусе нет ответа).")
        out.append("")
        out.append("- [ ] вопрос и эталон верны  - [ ] нужна правка: ____________  - [ ] исключить вопрос")
        out.append("")
    review_dir = ROOT / "review"
    review_dir.mkdir(exist_ok=True)
    (review_dir / "review_sample.md").write_text("\n".join(out), encoding="utf-8")
    print("вопросов в выборке:", len(picked), [r["qid"] for r in picked])


if __name__ == "__main__":
    main()
