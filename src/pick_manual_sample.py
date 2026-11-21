"""Выбор 10 ответов основной сетки (k = 3) для ручной разметки и вывод их в читаемом виде.

Выборка воспроизводима (seed = 5): по 2 ответа из групп «формальный с ответом», «разговорный», «вне корпуса»,
«контроль без поиска» и «другая конфигурация поиска». Результат печатается, чтобы разметчик сверил ответ с эталоном и текстом контекста.
Запуск: python src/pick_manual_sample.py [путь_для_сохранения.txt]
"""
from __future__ import annotations

import json
import random
import sys

from common import RESULTS_DIR

random.seed(5)


def main() -> None:
    records = [json.loads(line) for line in open(RESULTS_DIR / "generation_outputs.jsonl", encoding="utf-8")]
    grid = [r for r in records if r["experiment"] == "grid"]
    best = grid and next(r for r in grid if r["prompt"] != "no_context")
    best_cfg = (best["chunking"], best["retriever"])
    # только проверенные человеком вопросы: эталон надёжен, и ответы покрыты кросс-проверкой j3
    on_best = [r for r in grid if (r["chunking"], r["retriever"]) == best_cfg and r["verified_by_human"]]
    pools = {
        "formal": [r for r in on_best if r["subset"] == "formal" and r["answerable"]],
        "colloquial": [r for r in on_best if r["subset"] == "colloquial"],
        "oos": [r for r in on_best if not r["answerable"]],
        "no_context": [r for r in grid if r["prompt"] == "no_context" and r["verified_by_human"] and r["answerable"]],
        "naive_prompt": [r for r in on_best if r["prompt"] == "naive"],
    }
    picked = []
    for name, pool in pools.items():
        for r in random.sample(pool, 2):
            picked.append((name, r))
    lines = []
    for n, (name, r) in enumerate(picked, 1):
        lines.append(f"### {n}. [{name}] qid={r['qid']} cfg={r['chunking']}/{r['retriever']} prompt={r['prompt']} subset={r['subset']}")
        lines.append(f"ВОПРОС: {r['question']}")
        lines.append(f"ЭТАЛОН: {r['gold_answer'] or '(нет ответа в корпусе: ожидается отказ)'}  [{r['gold_article_id']} {r['gold_article_id_2']}]")
        lines.append(f"ПОЛУЧЕННЫЕ СТАТЬИ: {r['retrieved_articles']} | эталон найден: {r['gold_retrieved']}")
        lines.append(f"ОТВЕТ: {r['answer']}")
        lines.append(f"ОЦЕНКИ j1: corr={r['j1_correctness']} faith={r['j1_faithful']} abst={r['j1_abstained']} | j2: corr={r['j2_correctness']} faith={r['j2_faithful']} abst={r['j2_abstained']}")
        lines.append("КОНТЕКСТ:")
        lines.append(r["contexts"][:1500])
        lines.append("")
    text = "\n".join(lines)
    if len(sys.argv) > 1:
        open(sys.argv[1], "w", encoding="utf-8").write(text)
        print("записано в", sys.argv[1], "| ключи:", [(r["qid"], r["chunking"], r["retriever"], r["prompt"]) for _, r in picked])
    else:
        print(text)


if __name__ == "__main__":
    main()
