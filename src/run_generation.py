"""Эксперимент с генерацией.

Основная сетка (по умолчанию k = 3 контекста, температура 0.0): лучшие конфигурации поиска × три варианта промпта
+ контроль без поиска. Затем две абляции на лучшей конфигурации поиска с лучшим вариантом промпта:
  * число контекстов k в {1, 3, 5};
  * сравнение с платным генератором LLM_MODEL_2 (та же конфигурация, k = 3, T = 0.0, и контроль без поиска);
  * температура генератора {0.0, 0.7}, по 3 независимых прогона (метка кэша run-N, чтобы повторы не отдавались из кэша).
Ответы генерирует LLM_MODEL; оценивают две модели (JUDGE_MODEL и JUDGE_MODEL_2) по одной рубрике (prompts.py) при
температуре 0.0. Метрики считаются отдельно для подмножеств formal, colloquial и для всех вопросов.
Повторные запуски дёшевы: запросы кэшируются в .cache/.

Запуск: python src/run_generation.py [--top-k 3] [--skip-k-ablation] [--with-generator2] [--with-temperature]
По умолчанию выполняется фаза 1 (основная сетка и абляция по k); сравнение генераторов и абляция по температуре (фаза 2) включаются флагами.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time

import numpy as np

from chunking import load_corpus, make_chunks
from common import DATA_DIR, RESULTS_DIR, _post, cfg, chat, load_eval_set, parallel_map
from prompts import GOLD_NONE, JUDGE_NOCONTEXT_NOTE, JUDGE_SYSTEM, JUDGE_TEMPLATE, NO_ANSWER_PHRASE, PROMPT_VARIANTS, build_messages
from retrievers import BM25, Dense, TfidfChar, rerank_top, rrf

DEFAULT_TOP_K = 3
ABLATION_KS = (1, 3, 5)
TEMPERATURES = (0.0, 0.7)
TEMPERATURE_RUNS = 3
SUBSETS = ("formal", "colloquial", "all")
CONTEXT_CHARS = 6000
WORKERS = 4
RNG = np.random.default_rng(7)


def load_all_questions() -> list[dict]:
    """Все вопросы итогового набора (без исключённых ручной проверкой)."""
    return load_eval_set()


def gold_set(q: dict) -> set[str]:
    return {a for a in (q["gold_article_id"], q.get("gold_article_id_2", "")) if a}


def pick_configs(summary_path=RESULTS_DIR / "retrieval_summary.csv") -> list[tuple[str, str]]:
    """Лучшая конфигурация по hit@3 (при равенстве по MRR) на ВСЕХ вопросах и лучшая бесплатная (без API)."""
    with open(summary_path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("subset", "all") == "all"]
    key = lambda r: (float(r["hit@3"]), float(r["mrr"]))
    best = max(rows, key=key)
    free = max((r for r in rows if r["retriever"] in ("bm25", "tfidf_char")), key=key)
    configs = [(best["chunking"], best["retriever"])]
    if (free["chunking"], free["retriever"]) not in configs:
        configs.append((free["chunking"], free["retriever"]))
    return configs


def retrieve(chunking: str, retriever: str, queries: list[str], corpus: list[dict], depth: int = max(ABLATION_KS)):
    """Ранжированные индексы чанков (первые depth) для каждого запроса."""
    chunks = make_chunks(corpus, chunking)
    texts = [c["text"] for c in chunks]
    base, _, rerank = retriever.partition("+")
    if base == "bm25":
        model = BM25(texts)
        scores = np.array([model.scores(q) for q in queries])
    elif base == "tfidf_char":
        model = TfidfChar(texts)
        scores = np.array([model.scores(q) for q in queries])
    elif base in ("dense_small", "dense_large"):
        scores = Dense(texts, cfg("EMBED_REMOTE_SMALL" if base == "dense_small" else "EMBED_REMOTE_LARGE")).scores_many(queries)
    elif base in ("hybrid_small", "hybrid_large"):
        bm = BM25(texts)
        dense = Dense(texts, cfg("EMBED_REMOTE_SMALL" if base == "hybrid_small" else "EMBED_REMOTE_LARGE")).scores_many(queries)
        scores = np.array([rrf([bm.scores(q), dense[i]]) for i, q in enumerate(queries)])
    else:
        raise ValueError(retriever)
    orders = list(np.argsort(-scores, axis=1))
    if rerank:
        cut = [t[:2500] for t in texts]
        orders = parallel_map(lambda i: rerank_top(queries[i], cut, orders[i], 20), range(len(queries)), workers=WORKERS)
    return chunks, [o[:depth] for o in orders]


def format_contexts(chunks: list[dict], idx: np.ndarray) -> str:
    return "\n\n".join(f"[{n}] {chunks[j]['text'][:CONTEXT_CHARS]}" for n, j in enumerate(idx, 1))


def parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def cites_article(answer: str, article_ids: set[str]) -> bool:
    """Упомянута ли в ответе хотя бы одна из эталонных статей (формат «ст. 18», «статья 18»)."""
    for article_id in article_ids:
        no = article_id.split(":")[1]
        if re.search(rf"ст(?:\.|атья|атьи|атье)?\s*{re.escape(no)}(?!\d)(?!\.\d)", answer, re.I):
            return True
    return False


def safe_chat(*args, **kwargs) -> dict:
    """chat() без падения при сетевом сбое: пустой ответ с finish_reason = error (не кэшируется, повторный запуск довычислит)."""
    try:
        return chat(*args, **kwargs)
    except RuntimeError as exc:
        print(f"  сбой генерации: {str(exc)[:120]}", flush=True)
        return {"content": "", "reasoning": "", "finish_reason": "error", "usage": {}}


def bootstrap_ci(values: np.ndarray, n: int = 2000) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    means = [RNG.choice(values, size=len(values), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def base_record(q: dict, chunking: str, retriever: str, prompt: str, k: int | None, experiment: str, temperature: float = 0.0, run: int = 0,
                generator: str = "g1") -> dict:
    return {"experiment": experiment, "generator": generator, "chunking": chunking, "retriever": retriever, "prompt": prompt, "top_k": k, "temperature": temperature, "run": run,
            "qid": q["qid"], "subset": q["subset"], "question": q["question"], "answerable": q["answerable"] == "True",
            "gold_article_id": q["gold_article_id"], "gold_article_id_2": q["gold_article_id_2"], "gold_answer": q["gold_answer"],
            "verified_by_human": q.get("verified_by_human") == "True"}


def generate(chunking, retriever, prompts, k, questions, chunks, tops, llm, experiment="grid", temperature=0.0, run=0, generator="g1") -> list[dict]:
    """Ответы LLM_MODEL для всех вопросов и заданных вариантов промпта при k контекстах и температуре."""
    contexts = [format_contexts(chunks, t[:k]) for t in tops]
    retrieved = ["|".join(dict.fromkeys(chunks[j]["article_id"] for j in t[:k])) for t in tops]
    jobs = [(i, v) for i in range(len(questions)) for v in prompts]
    tag = f"run-{run}" if run else None  # независимые повторы: тело запроса одно и то же, различается только метка кэша

    def gen(job):
        i, variant = job
        return safe_chat(llm, build_messages(variant, contexts[i], questions[i]["question"]), max_tokens=3000, temperature=temperature, tag=tag)

    records = []
    for (i, variant), out in zip(jobs, parallel_map(gen, jobs, workers=WORKERS)):
        rec = base_record(questions[i], chunking, retriever, variant, k, experiment, temperature, run, generator)
        rec.update({"retrieved_articles": retrieved[i], "gold_retrieved": bool(gold_set(questions[i]) & set(retrieved[i].split("|"))),
                    "answer": out["content"], "finish_reason": out["finish_reason"], "contexts": contexts[i],
                    "context_chars": len(contexts[i]), "prompt_tokens": out["usage"].get("prompt_tokens"),
                    "completion_tokens": out["usage"].get("completion_tokens"),
                    "reasoning_tokens": (out["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens"), "model": llm})
        records.append(rec)
    return records


def generate_control(questions, llm, generator="g1", experiment="grid") -> list[dict]:
    """Контроль без поиска: тот же LLM_MODEL, только вопрос и нейтральный промпт."""
    def gen(i):
        return safe_chat(llm, build_messages("no_context", "", questions[i]["question"]), max_tokens=3000)

    records = []
    for i, out in enumerate(parallel_map(gen, range(len(questions)), workers=WORKERS)):
        rec = base_record(questions[i], "-", "none", "no_context", None, experiment, generator=generator)
        rec.update({"retrieved_articles": "", "gold_retrieved": False, "answer": out["content"], "finish_reason": out["finish_reason"],
                    "contexts": "(фрагменты документов не предоставлялись)", "context_chars": 0,
                    "prompt_tokens": out["usage"].get("prompt_tokens"), "completion_tokens": out["usage"].get("completion_tokens"),
                    "reasoning_tokens": (out["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens"), "model": llm})
        records.append(rec)
    return records


def judge_records(records: list[dict], judges: dict[str, str], subset_judges: dict | None = None) -> None:
    """Оценка ответов судьями j1 и j2 (температура 0.0). subset_judges: {ключ: (модель, предикат)}: судья оценивает
    только записи, для которых предикат истинен (кросс-проверка j3 на проверенных человеком вопросах); у остальных поля None."""
    def judge_one(job):
        rec, model = job
        gold = rec["gold_answer"] if rec["answerable"] else GOLD_NONE
        prompt = JUDGE_TEMPLATE.format(question=rec["question"], gold=gold, contexts=rec["contexts"], answer=rec["answer"] or "(пустой ответ)")
        if rec["prompt"] == "no_context":
            prompt += JUDGE_NOCONTEXT_NOTE
        try:
            out = chat(model, [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}], max_tokens=4000)
        except RuntimeError as exc:  # сетевой сбой: не прерываем прогон; запись не кэшируется, повторный запуск довычислит пропуск
            print(f"  сбой вызова оценщика {model}: {str(exc)[:120]}", flush=True)
            return {}
        return parse_json(out["content"]) or {}

    for key, model in judges.items():
        verdicts = parallel_map(judge_one, [(rec, model) for rec in records], workers=WORKERS)
        for rec, v in zip(records, verdicts):
            rec[f"{key}_correctness"] = v.get("correctness")
            rec[f"{key}_faithful"] = None if rec["prompt"] == "no_context" else v.get("faithful")
            rec[f"{key}_abstained"] = v.get("abstained")
            rec[f"{key}_comment"] = v.get("comment", "")
        print(f"  оценка {key} ({model}): {len(records)} ответов, JSON разобран в {sum(1 for v in verdicts if v)}", flush=True)
    for key, (model, predicate) in (subset_judges or {}).items():
        target = [rec for rec in records if predicate(rec)]
        verdicts = iter(parallel_map(judge_one, [(rec, model) for rec in target], workers=WORKERS))
        parsed = 0
        for rec in records:
            if rec in target:
                v = next(verdicts)
                parsed += bool(v)
                rec[f"{key}_correctness"] = v.get("correctness")
                rec[f"{key}_faithful"] = None if rec["prompt"] == "no_context" else v.get("faithful")
                rec[f"{key}_abstained"] = v.get("abstained")
                rec[f"{key}_comment"] = v.get("comment", "")
            else:
                for field in ("correctness", "faithful", "abstained", "comment"):
                    rec[f"{key}_{field}"] = None
        print(f"  кросс-проверка {key} ({model}): {len(target)} из {len(records)} ответов, JSON разобран в {parsed}", flush=True)
    for rec in records:
        rec["cites_gold"] = bool(rec["answerable"] and cites_article(rec["answer"], gold_set(rec)))
        rec["phrase_abstain"] = NO_ANSWER_PHRASE.lower() in rec["answer"].lower()


def metrics(rs: list[dict], jkey: str) -> dict:
    """Метрики по списку ответов для одного оценщика."""
    ans = [r for r in rs if r["answerable"]]
    oos = [r for r in rs if not r["answerable"]]
    full = np.array([1.0 if r[f"{jkey}_correctness"] == 2 else 0.0 for r in ans])
    ok = [r for r in ans if r[f"{jkey}_correctness"] is not None]
    lo, hi = bootstrap_ci(full)
    no_ctx = bool(rs) and rs[0]["prompt"] == "no_context"
    rate = lambda xs: round(float(np.mean(xs)), 3) if len(xs) else "n/a"
    return {
        "n_answerable": len(ans), "n_unanswerable": len(oos),
        "correct_full_rate": rate(full), "correct_full_lo": round(lo, 3) if len(full) else "n/a", "correct_full_hi": round(hi, 3) if len(full) else "n/a",
        "correct_mean_0_2": rate([r[f"{jkey}_correctness"] for r in ok]),
        "faithful_rate": "n/a" if no_ctx else rate([r[f"{jkey}_faithful"] == 1 for r in rs]),
        "false_abstain_rate": rate([r[f"{jkey}_abstained"] == 1 for r in ans]),
        "abstain_correct_rate": rate([r[f"{jkey}_abstained"] == 1 for r in oos]),
        "cites_gold_rate": rate([r["cites_gold"] for r in ans]),
        "gold_in_context_rate": "n/a" if no_ctx else rate([r["gold_retrieved"] for r in ans]),
        "avg_context_chars": round(float(np.mean([r["context_chars"] for r in rs]))) if rs else "n/a",
        "avg_prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in rs if r["prompt_tokens"]]))) if any(r["prompt_tokens"] for r in rs) else None,
        "avg_answer_chars": round(float(np.mean([len(r["answer"]) for r in rs]))) if rs else "n/a",
        "avg_completion_tokens": round(float(np.mean([r["completion_tokens"] for r in rs if r.get("completion_tokens")]))) if any(r.get("completion_tokens") for r in rs) else None,
        "avg_reasoning_tokens": round(float(np.mean([r["reasoning_tokens"] for r in rs if r.get("reasoning_tokens") is not None]))) if any(r.get("reasoning_tokens") is not None for r in rs) else None,
        "empty_or_truncated": int(sum(1 for r in rs if not r["answer"] or r["finish_reason"] != "stop")),
    }


def select(records: list[dict], subset: str, **fields) -> list[dict]:
    return [r for r in records if (subset == "all" or r["subset"] == subset) and all(r[k] == v for k, v in fields.items())]


def summarize_grid(records: list[dict], judges: dict[str, str], groups: list[tuple], k: int) -> list[dict]:
    rows = []
    for jkey, model in judges.items():
        for subset in SUBSETS:
            for chunking, retriever, prompt in groups:
                kk = None if prompt == "no_context" else k
                rs = select(records, subset, experiment="grid", chunking=chunking, retriever=retriever, prompt=prompt)
                if rs:
                    rows.append({"judge": jkey, "judge_model": model, "subset": subset, "chunking": chunking, "retriever": retriever,
                                 "prompt": prompt, "top_k": kk if kk else "-", **metrics(rs, jkey)})
    return rows


def best_prompt(rows: list[dict], config: tuple[str, str]) -> str:
    """Лучший вариант промпта на лучшей конфигурации поиска: по доле полностью верных ответов на всех вопросах
    (среднее по двум оценщикам), при равенстве по верности и затем по верному отказу."""
    scores = {}
    for row in rows:
        if row["subset"] == "all" and (row["chunking"], row["retriever"]) == config and row["prompt"] in PROMPT_VARIANTS:
            scores.setdefault(row["prompt"], []).append((row["correct_full_rate"], row["faithful_rate"], row["abstain_correct_rate"]))
    avg = {p: tuple(float(np.mean([s[i] for s in v])) for i in range(3)) for p, v in scores.items()}
    return max(avg, key=avg.get)


def latency_benchmark(prompt, ks, questions, chunks, tops, llm, n_questions: int = 6) -> dict[int, tuple[float, int]]:
    """Реальная задержка генерации по k: прямые вызовы без кэша на одних и тех же вопросах."""
    idx = [i for i, q in enumerate(questions) if q["answerable"] == "True"][:n_questions]
    result = {}
    for k in ks:
        times = []
        for i in idx:
            messages = build_messages(prompt, format_contexts(chunks, tops[i][:k]), questions[i]["question"])
            t = time.time()
            _post("/chat/completions", {"model": llm, "messages": messages, "max_tokens": 3000, "temperature": 0.0})
            times.append(time.time() - t)
        result[k] = (round(statistics.median(times), 2), len(times))
    return result


def latency_generators(models: dict[str, str], prompt: str, questions, chunks, tops, n_formal: int = 5, n_colloquial: int = 3) -> dict:
    """Задержка на один ответ: прямые вызовы без кэша на одних и тех же вопросах, T = 0.0."""
    idx = [i for i, q in enumerate(questions) if q["answerable"] == "True" and q["subset"] == "formal"][:n_formal]
    idx += [i for i, q in enumerate(questions) if q["subset"] == "colloquial"][:n_colloquial]
    result = {}
    for gkey, model in models.items():
        for setup in ("rag_best", "no_context"):
            times = []
            for i in idx:
                if setup == "rag_best":
                    messages = build_messages(prompt, format_contexts(chunks, tops[i][:DEFAULT_TOP_K]), questions[i]["question"])
                else:
                    messages = build_messages("no_context", "", questions[i]["question"])
                t0 = time.time()
                _post("/chat/completions", {"model": model, "messages": messages, "max_tokens": 3000, "temperature": 0.0})
                times.append(time.time() - t0)
            result[(gkey, setup)] = {"latency_mean_s": round(float(np.mean(times)), 2), "latency_median_s": round(float(np.median(times)), 2),
                                     "latency_n": len(times)}
            print(f"  задержка {gkey} {setup}: среднее {np.mean(times):.1f} с (n={len(times)})", flush=True)
    return result


def write_csv(path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def dump_outputs(records: list[dict]) -> None:
    """Промежуточная и итоговая запись всех ответов с оценками (после каждого этапа)."""
    with open(RESULTS_DIR / "generation_outputs.jsonl", "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + chr(10))


def temperature_summary(records: list[dict], judges: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """Метрики по прогонам и сводка по температурам: среднее, std между прогонами, доля вопросов с изменившейся оценкой."""
    runs, agg = [], []
    for jkey, model in judges.items():
        for temp in TEMPERATURES:
            per_run = []
            for run in range(1, TEMPERATURE_RUNS + 1):
                rs = [r for r in records if r["experiment"] == "temperature" and r["temperature"] == temp and r["run"] == run]
                m = metrics(rs, jkey)
                per_run.append(m)
                runs.append({"judge": jkey, "temperature": temp, "run": run, **{k: m[k] for k in
                             ("correct_mean_0_2", "correct_full_rate", "faithful_rate", "abstain_correct_rate", "cites_gold_rate", "avg_answer_chars")}})
            row = {"judge": jkey, "judge_model": model, "temperature": temp, "runs": TEMPERATURE_RUNS}
            for name in ("correct_mean_0_2", "correct_full_rate", "faithful_rate", "abstain_correct_rate"):
                vals = [float(m[name]) for m in per_run]
                row[f"{name}_mean"] = round(float(np.mean(vals)), 3)
                row[f"{name}_std"] = round(float(np.std(vals, ddof=1)), 3)
            # устойчивость: доля вопросов, у которых оценка менялась между прогонами (по correctness для вопросов с ответом,
            # по отказу для вопросов вне корпуса)
            changed_ans, changed_oos = [], []
            for qid in sorted({r["qid"] for r in records if r["experiment"] == "temperature"}):
                rs = [r for r in records if r["experiment"] == "temperature" and r["temperature"] == temp and r["qid"] == qid]
                if rs[0]["answerable"]:
                    changed_ans.append(len({r[f"{jkey}_correctness"] for r in rs}) > 1)
                else:
                    changed_oos.append(len({r[f"{jkey}_abstained"] for r in rs}) > 1)
            row["unstable_share_answerable"] = round(float(np.mean(changed_ans)), 3)
            row["unstable_share_unanswerable"] = round(float(np.mean(changed_oos)), 3) if changed_oos else "n/a"
            agg.append(row)
    return runs, agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="число контекстов в основной сетке (по умолчанию 3)")
    parser.add_argument("--with-generator2", action="store_true", help="фаза 2: сравнение с платным генератором LLM_MODEL_2")
    parser.add_argument("--skip-k-ablation", action="store_true")
    parser.add_argument("--with-temperature", action="store_true", help="фаза 2: абляция по температуре (3 прогона на каждую)")
    args = parser.parse_args()
    top_k = args.top_k

    corpus = load_corpus()
    questions = load_all_questions()
    queries = [q["question"] for q in questions]
    configs = pick_configs()
    llm = cfg("LLM_MODEL")
    judges = {"j1": cfg("JUDGE_MODEL"), "j2": cfg("JUDGE_MODEL_2")}  # две модели-оценщика, одна рубрика, температура 0.0
    print(f"k = {top_k}; вопросов {len(questions)}; конфигурации поиска: {configs}", flush=True)

    records: list[dict] = []
    retrieved = {}
    for chunking, retriever in configs:
        chunks, tops = retrieve(chunking, retriever, queries, corpus)
        retrieved[(chunking, retriever)] = (chunks, tops)
        records += generate(chunking, retriever, PROMPT_VARIANTS, top_k, questions, chunks, tops, llm)
        print(f"  сгенерировано для {chunking}/{retriever} (k={top_k})", flush=True)
    records += generate_control(questions, llm)
    print("  контроль без поиска сгенерирован", flush=True)
    # кросс-проверка j3 (JUDGE_MODEL_3, медленная большая модель) только на проверенных человеком вопросах основной сетки лучшей конфигурации
    best_cfg_main = configs[0]
    only_verified = lambda r: r["verified_by_human"] and (r["prompt"] == "no_context" or (r["chunking"], r["retriever"]) == best_cfg_main)
    judge_records(records, judges, {"j3": (cfg("JUDGE_MODEL_3"), only_verified)})

    groups = [(c, r, p) for c, r in configs for p in PROMPT_VARIANTS] + [("-", "none", "no_context")]
    grid = summarize_grid(records, judges, groups, top_k)
    RESULTS_DIR.mkdir(exist_ok=True)
    write_csv(RESULTS_DIR / "generation_summary.csv", grid)
    dump_outputs(records)

    best_cfg = configs[0]
    prompt = best_prompt(grid, best_cfg)
    chunks, tops = retrieved[best_cfg]
    print(f"лучшая конфигурация {best_cfg}, лучший промпт: {prompt}", flush=True)

    if args.with_generator2:
        llm2 = cfg("LLM_MODEL_2")  # платный генератор для сравнения с бесплатным LLM_MODEL
        extra = generate(best_cfg[0], best_cfg[1], [prompt], top_k, questions, chunks, tops, llm2, experiment="generator2", generator="g2")
        extra += generate_control(questions, llm2, generator="g2", experiment="generator2")
        judge_records(extra, judges)
        records += extra
        lat = latency_generators({"g1": llm, "g2": llm2}, prompt, questions, chunks, tops)
        pool = [r for r in records if r["experiment"] in ("grid", "generator2")]
        rows = []
        for jkey in judges:
            for subset in SUBSETS:
                for gkey, gmodel in (("g1", llm), ("g2", llm2)):
                    for setup in ("rag_best", "no_context"):
                        if setup == "rag_best":
                            rs = select([r for r in pool if r["generator"] == gkey], subset, chunking=best_cfg[0], retriever=best_cfg[1], prompt=prompt)
                        else:
                            rs = select([r for r in pool if r["generator"] == gkey], subset, prompt="no_context")
                        rows.append({"judge": jkey, "subset": subset, "generator": gkey, "model": gmodel, "setup": setup,
                                     "prompt": prompt if setup == "rag_best" else "no_context", **metrics(rs, jkey), **lat[(gkey, setup)]})
        write_csv(RESULTS_DIR / "generator_comparison.csv", rows)
        dump_outputs(records)
        print("сравнение генераторов записано", flush=True)

    if not args.skip_k_ablation:
        extra = []
        for k in ABLATION_KS:
            if k != top_k:
                extra += generate(best_cfg[0], best_cfg[1], [prompt], k, questions, chunks, tops, llm, experiment="k_ablation")
        judge_records(extra, judges)
        records += extra
        base_ablation = [dict(r, experiment="k_ablation") for r in records if r["experiment"] == "grid" and (r["chunking"], r["retriever"], r["prompt"]) == (*best_cfg, prompt)]
        pool = base_ablation + extra
        lat = latency_benchmark(prompt, ABLATION_KS, questions, chunks, tops, llm)
        rows = []
        for jkey, model in judges.items():
            for subset in SUBSETS:
                for k in ABLATION_KS:
                    rs = select(pool, subset, top_k=k)
                    rows.append({"judge": jkey, "subset": subset, "top_k": k, **metrics(rs, jkey), "latency_median_s": lat[k][0], "latency_n": lat[k][1]})
        write_csv(RESULTS_DIR / "k_ablation.csv", rows)
        dump_outputs(records)
        print("k-абляция записана", flush=True)

    if args.with_temperature:
        extra = []
        for temp in TEMPERATURES:
            for run in range(1, TEMPERATURE_RUNS + 1):
                extra += generate(best_cfg[0], best_cfg[1], [prompt], top_k, questions, chunks, tops, llm, experiment="temperature", temperature=temp, run=run)
                print(f"  температура {temp}, прогон {run} сгенерирован", flush=True)
        judge_records(extra, judges)
        records += extra
        runs, agg = temperature_summary(records, judges)
        write_csv(RESULTS_DIR / "temperature_runs.csv", runs)
        write_csv(RESULTS_DIR / "temperature_ablation.csv", agg)
        print("температурная абляция записана", flush=True)

    dump_outputs(records)
    for row in grid:
        if row["subset"] == "all":
            print(row)


if __name__ == "__main__":
    main()
