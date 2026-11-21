"""Проба: влияет ли параметр temperature на ответы генератора (модель с рассуждениями).

Пять вопросов с эталонной статьёй в контексте, по два независимых вызова при T=0.0 и T=0.7 (без кэша).
Сравниваются совпадения ответов между повторами и число токенов. Запуск: python src/temperature_probe.py
"""
from __future__ import annotations

import csv
import statistics

from chunking import load_corpus
from common import RESULTS_DIR, _post, cfg
from prompts import build_messages
from run_generation import load_all_questions

N_QUESTIONS, REPEATS = 5, 2


def main() -> None:
    corpus = {a["article_id"]: a for a in load_corpus()}
    qs = [q for q in load_all_questions() if q["answerable"] == "True" and q.get("subset", "formal") == "formal"][:N_QUESTIONS]
    llm = cfg("LLM_MODEL")
    rows = []
    for q in qs:
        art = corpus[q["gold_article_id"]]
        contexts = f"[1] {art['doc_short']}, ст. {art['article_no']}. {art['title']}\n{art['text'][:3000]}"
        messages = build_messages("strict", contexts, q["question"])
        for temp in (0.0, 0.7):
            for rep in range(REPEATS):
                data = _post("/chat/completions", {"model": llm, "messages": messages, "max_tokens": 3000, "temperature": temp})
                usage = data.get("usage", {})
                rows.append({"qid": q["qid"], "temperature": temp, "repeat": rep, "answer": data["choices"][0]["message"].get("content", "").strip(),
                             "completion_tokens": usage.get("completion_tokens"),
                             "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")})
    with open(RESULTS_DIR / "temperature_probe.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for temp in (0.0, 0.7):
        ident, diffs = 0, []
        for q in qs:
            pair = [r for r in rows if r["qid"] == q["qid"] and r["temperature"] == temp]
            ident += pair[0]["answer"] == pair[1]["answer"]
            diffs.append(abs(pair[0]["completion_tokens"] - pair[1]["completion_tokens"]))
        print(f"T={temp}: одинаковых ответов между повторами {ident} из {len(qs)}; "
              f"медианная разница completion_tokens между повторами {statistics.median(diffs)}")


if __name__ == "__main__":
    main()
