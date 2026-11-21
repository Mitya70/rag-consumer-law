"""Эксперимент по поиску: чанкинг × поисковик (× реранк). Метрики на уровне статьи.

hit@k: среди первых k чанков есть чанк из эталонной статьи; MRR считается по первому такому чанку.
Для каждой метрики считается 95%-й бутстрэп-интервал по вопросам (набор маленький, интервалы широкие).
Запуск: python src/run_retrieval.py
"""
from __future__ import annotations

import csv
import time
import uuid

import numpy as np

from chunking import VARIANTS, load_corpus, make_chunks
from common import DATA_DIR, RESULTS_DIR, cfg, load_eval_set, cached_post, embed_remote, parallel_map, rerank_remote, _post
from retrievers import BM25, Dense, TfidfChar, rerank_top, rrf

RERANK_CHARS = 2500  # длинные чанки обрезаются перед реранком
RNG = np.random.default_rng(42)


def load_questions() -> list[dict]:
    """Вопросы с ответом без исключённых ручной проверкой."""
    return [r for r in load_eval_set() if r["answerable"] == "True"]


def first_gold_rank(order: np.ndarray, chunks: list[dict], gold: set[str]) -> int:
    """Позиция (с 1) первого чанка из любой эталонной статьи; 999, если ни одной нет в выдаче."""
    for pos, idx in enumerate(order, 1):
        if chunks[idx]["article_id"] in gold:
            return pos
    return 999


def both_gold_in_top(order: np.ndarray, chunks: list[dict], gold: set[str], k: int = 5) -> bool | None:
    """Для вопросов с двумя эталонными статьями: обе ли статьи есть среди первых k чанков."""
    if len(gold) < 2:
        return None
    return gold <= {chunks[i]["article_id"] for i in order[:k]}


def bootstrap_ci(values: np.ndarray, n: int = 2000) -> tuple[float, float]:
    means = [RNG.choice(values, size=len(values), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(ranks: np.ndarray) -> dict:
    out = {}
    for k in (1, 3, 5):
        hits = (ranks <= k).astype(float)
        lo, hi = bootstrap_ci(hits)
        out[f"hit@{k}"], out[f"hit@{k}_lo"], out[f"hit@{k}_hi"] = hits.mean(), lo, hi
    rr = np.where(ranks <= 20, 1.0 / ranks, 0.0)
    lo, hi = bootstrap_ci(rr)
    out["mrr"], out["mrr_lo"], out["mrr_hi"] = rr.mean(), lo, hi
    return out


def latency_benchmark() -> list[dict]:
    """Реальная задержка API: уникальные запросы, чтобы не попасть в кэш."""
    rows = []
    for name, model in (("embed_small", cfg("EMBED_REMOTE_SMALL")), ("embed_large", cfg("EMBED_REMOTE_LARGE"))):
        times = []
        for _ in range(5):
            t = time.time()
            _post("/embeddings", {"model": model, "input": [f"пробный запрос {uuid.uuid4()}"], "input_type": "query", "truncate": "END"})
            times.append(time.time() - t)
        rows.append({"step": name, "model": model, "median_s": float(np.median(times)), "n": 5})
    times = []
    for _ in range(5):
        t = time.time()
        _post("/rerank", {"model": cfg("RERANK_MODEL"), "query": f"пробный запрос {uuid.uuid4()}", "documents": ["документ " * 50] * 20})
        times.append(time.time() - t)
    rows.append({"step": "rerank_top20", "model": cfg("RERANK_MODEL"), "median_s": float(np.median(times)), "n": 5})
    return rows


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    corpus = load_corpus()
    questions = load_questions()
    queries = [q["question"] for q in questions]
    gold = [{a for a in (q["gold_article_id"], q.get("gold_article_id_2", "")) if a} for q in questions]
    subsets = np.array([q.get("subset") or "formal" for q in questions])
    print(f"вопросов с ответом: {len(questions)} (formal {int((subsets == 'formal').sum())}, colloquial {int((subsets == 'colloquial').sum())}); "
          f"статей в корпусе: {len(corpus)}")

    small, large = cfg("EMBED_REMOTE_SMALL"), cfg("EMBED_REMOTE_LARGE")
    summary, per_question = [], []

    for variant in VARIANTS:
        chunks = make_chunks(corpus, variant)
        texts = [c["text"] for c in chunks]
        avg_chars = float(np.mean([len(t) for t in texts]))
        print(f"\n== {variant}: {len(chunks)} чанков, средняя длина {avg_chars:.0f} симв.")
        bm25, tfidf = BM25(texts), TfidfChar(texts)
        dense = {"dense_small": Dense(texts, small), "dense_large": Dense(texts, large)}

        scores = {
            "bm25": np.array([bm25.scores(q) for q in queries]),
            "tfidf_char": np.array([tfidf.scores(q) for q in queries]),
        }
        for name, model in dense.items():
            scores[name] = model.scores_many(queries)
        scores["hybrid_small"] = np.array([rrf([scores["bm25"][i], scores["dense_small"][i]]) for i in range(len(queries))])
        scores["hybrid_large"] = np.array([rrf([scores["bm25"][i], scores["dense_large"][i]]) for i in range(len(queries))])

        orders = {name: np.argsort(-s, axis=1) for name, s in scores.items()}

        # реранк: два базовых кандидата
        rerank_texts = [t[:RERANK_CHARS] for t in texts]
        for base in ("bm25", "dense_large"):
            def run(i, base=base):
                return rerank_top(queries[i], rerank_texts, orders[base][i], top_n=20)

            orders[f"{base}+rerank"] = np.array(parallel_map(run, range(len(queries)), workers=3))

        for name, order in orders.items():
            ranks = np.array([first_gold_rank(order[i], chunks, gold[i]) for i in range(len(queries))])
            top5_chars = float(np.mean([sum(len(texts[j]) for j in order[i][:5]) for i in range(len(queries))]))
            for subset in ("formal", "colloquial", "all"):
                mask = np.ones(len(queries), dtype=bool) if subset == "all" else subsets == subset
                if not mask.any():
                    continue
                summary.append({"subset": subset, "chunking": variant, "retriever": name, "n_chunks": len(chunks), "avg_chunk_chars": round(avg_chars),
                                "top5_context_chars": round(top5_chars), "n_questions": int(mask.sum()), **summarize(ranks[mask])})
            line = {s: summary[-3 + i] for i, s in enumerate(("formal", "colloquial", "all")) if len(summary) >= 3}
            for i, q in enumerate(questions):
                per_question.append({"subset": subsets[i], "chunking": variant, "retriever": name, "qid": q["qid"],
                                     "gold_article_id": "|".join(sorted(gold[i])), "first_gold_rank": int(ranks[i]),
                                     "both_gold_in_top5": both_gold_in_top(order[i], chunks, gold[i]),
                                     "top5_articles": "|".join(chunks[j]["article_id"] for j in order[i][:5])})
            f, c = line["formal"], line["colloquial"]
            print(f"  {name:22s} formal hit@3={f['hit@3']:.2f} mrr={f['mrr']:.3f} | colloquial hit@1={c['hit@1']:.2f} hit@3={c['hit@3']:.2f} mrr={c['mrr']:.3f}")

    with open(RESULTS_DIR / "retrieval_summary.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()} for row in summary)
    with open(RESULTS_DIR / "retrieval_per_question.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_question[0].keys()))
        writer.writeheader()
        writer.writerows(per_question)

    lat = latency_benchmark()
    with open(RESULTS_DIR / "api_latency.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(lat[0].keys()))
        writer.writeheader()
        writer.writerows({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in lat)
    print("\nзадержки API:", lat)


if __name__ == "__main__":
    main()
