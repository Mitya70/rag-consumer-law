"""Собирает и исполняет notebooks/results.ipynb: таблицы и графики по файлам из results/.

Запуск: python src/make_notebook.py  (нужны результаты run_retrieval.py, run_generation.py и judge_agreement.py)
"""
from __future__ import annotations

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

from common import ROOT

CELLS = [
    ("md", """# Результаты: RAG-ассистент по законодательству о защите прав потребителей

Ноутбук только читает готовые результаты из `../results/` и `../data/` (запросы к API здесь не выполняются).
Эксперименты запускаются скриптами `src/run_retrieval.py` и `src/run_generation.py`, подробности в README.

Это учебный проект, а не юридическая консультация. Оценочный набор составлен моделями (формальные вопросы по тексту статей,
разговорные написаны ассистентом-LLM) и **не проверен человеком** (`verified_by_human = False`). Цифры показывают сравнительное
качество вариантов, а не абсолютную точность системы."""),
    ("code", """from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd() if (Path.cwd() / "results").exists() else Path.cwd().parent
R, D = ROOT / "results", ROOT / "data"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 110)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})

meta = json.loads((D / "corpus_meta.json").read_text(encoding="utf-8"))
ev = pd.read_csv(D / "eval_set.csv")
print(f"Корпус: {meta['total_articles']} статей, {meta['total_chars']:,} символов; загружен {meta['retrieved']}")
for k, v in meta["sources"].items():
    print(f"  {k}: версия страницы Викитеки от {v['revision_timestamp']}, статей {v['articles']}")
print(f"Оценочный набор: {len(ev)} вопросов; проверено человеком: {int(ev.verified_by_human.sum())}")
display(ev.groupby(["subset", "answerable"]).size().rename("вопросов").to_frame())
display(ev[ev.subset == "colloquial"][["qid", "question", "gold_article_id", "gold_article_id_2"]])"""),
    ("md", "## 1. Поиск: чанкинг × поисковик, по подмножествам\n\nhit@k: среди первых k чанков есть чанк эталонной статьи (для вопросов с двумя эталонными статьями засчитывается любая); MRR по первому такому чанку. Подмножества: `formal` (вопросы по тексту статей, 44) и `colloquial` (разговорные, 18)."),
    ("code", """s = pd.read_csv(R / "retrieval_summary.csv")
order = ["bm25", "tfidf_char", "dense_small", "dense_large", "hybrid_small", "hybrid_large", "bm25+rerank", "dense_large+rerank"]
chunk_order = ["article", "part", "win500", "win1000"]
display(s.drop_duplicates("chunking").set_index("chunking")[["n_chunks", "avg_chunk_chars"]].loc[chunk_order])
for subset in ("formal", "colloquial", "all"):
    print(f"\\nMRR, подмножество {subset}")
    display(s[s.subset == subset].pivot(index="retriever", columns="chunking", values="mrr").loc[order, chunk_order].round(3))"""),
    ("code", """fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
for ax, subset in zip(axes, ("formal", "colloquial", "all")):
    mat = s[s.subset == subset].pivot(index="retriever", columns="chunking", values="mrr").loc[order, chunk_order]
    im = ax.imshow(mat.values, cmap="viridis", vmin=0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(chunk_order)), chunk_order)
    ax.set_yticks(range(len(order)), order)
    ax.set_title(f"MRR: {subset}")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", color="white" if mat.values[i, j] < 0.6 else "black", fontsize=8)
    ax.grid(False)
fig.colorbar(im, ax=axes, shrink=0.8)
plt.show()"""),
    ("md", """### Где методы расходятся: разговорные вопросы

На формальных вопросах почти все плотные конфигурации упираются в потолок (hit@3 около 1.0). Разговорные вопросы почти не пересекаются по лексике со статьями, и методы разделяются. Интервалы бутстрэпа (95%, по 18 вопросам) широкие: при отсутствии промахов интервал вырождается в точку, это не доказательство безошибочности."""),
    ("code", """cols = ["chunking", "retriever", "hit@1", "hit@1_lo", "hit@1_hi", "hit@3", "hit@3_lo", "hit@3_hi", "mrr", "mrr_lo", "mrr_hi"]
col = s[(s.subset == "colloquial") & (s.chunking == "part")].set_index("retriever").loc[order].reset_index()
display(col[["retriever"] + cols[2:]].round(3))"""),
    ("code", """sel = s[(s.subset == "colloquial") & s.retriever.isin(["bm25", "tfidf_char", "dense_small", "dense_large", "hybrid_large", "bm25+rerank", "dense_large+rerank"])].copy()
sel["label"] = sel.chunking + " / " + sel.retriever
sel = sel.sort_values("mrr")
fig, ax = plt.subplots(figsize=(8, 9))
ax.errorbar(sel.mrr, range(len(sel)), xerr=[sel.mrr - sel.mrr_lo, sel.mrr_hi - sel.mrr], fmt="o", capsize=3)
ax.set_yticks(range(len(sel)), sel.label)
ax.set_xlabel("MRR на разговорных вопросах, 95% бутстрэп-интервал")
plt.show()"""),
    ("code", """pq = pd.read_csv(R / "retrieval_per_question.csv")
two = pq[(pq.chunking == "part") & pq.both_gold_in_top5.notna()]
print("Вопросы с двумя эталонными статьями (разговорные): обе статьи среди первых 5 чанков, чанкинг part")
display(two.groupby("retriever").both_gold_in_top5.mean().round(2).rename("доля").to_frame().loc[order])"""),
    ("md", "### Задержка API (медиана из 5 вызовов с уникальными запросами)"),
    ("code", """pd.read_csv(R / "api_latency.csv")"""),
    ("md", "### Разбор ошибок поиска (чанкинг part)"),
    ("code", """evq = ev.set_index("qid")
def fails(retriever, subset, min_rank, chunking="part"):
    sub = pq[(pq.chunking == chunking) & (pq.retriever == retriever) & (pq.subset == subset) & (pq.first_gold_rank > min_rank)].copy()
    sub["вопрос"] = sub.qid.map(evq.question)
    return sub[["qid", "gold_article_id", "first_gold_rank", "top5_articles", "вопрос"]]
print("dense_large, формальные, эталон не на 1-м месте:")
display(fails("dense_large", "formal", 1))
print("dense_large, разговорные, эталон не на 1-м месте:")
display(fails("dense_large", "colloquial", 1))
print("bm25, разговорные, эталон ниже 3-го места (первые 8):")
display(fails("bm25", "colloquial", 3).head(8))"""),
    ("md", """## 2. Генерация: промпты, контроль без поиска, k = 3

Генератор `LLM_MODEL` (бесплатная модель среднего размера с рассуждениями), температура 0.0, 3 контекста (k = 3). Ответы оценивают **две** модели одной рубрикой при T = 0:
`j1` = `ag/gemini-3-flash` (основной, выбран за скорость около 2 с на вызов), `j2` = `am/laguna-xs-2.1` (другое семейство, без рассуждений).
Кросс-проверка `j3` = `am/nemotron-3-ultra-550b-a55b` (медленная большая модель того же семейства, что и генератор) выполнена только на ответах на проверенные человеком вопросы лучшей конфигурации.
Рубрика: correctness 0-2 относительно эталона, faithful (все утверждения подтверждены фрагментами), abstained (отказ). Для контроля без поиска верность не определена (n/a).
`cites_gold` считается программно (упомянута ли в ответе эталонная статья)."""),
    ("code", """g = pd.read_csv(R / "generation_summary.csv")
g["config"] = g.chunking + " / " + g.retriever
headline = (R / "headline_judge.txt").read_text().strip() if (R / "headline_judge.txt").exists() else "j1"
print("Основной оценщик (лучше согласуется с ручными метками):", headline)
show = ["config", "prompt", "n_answerable", "correct_full_rate", "correct_full_lo", "correct_full_hi", "correct_mean_0_2", "faithful_rate",
        "false_abstain_rate", "abstain_correct_rate", "cites_gold_rate", "gold_in_context_rate", "avg_context_chars", "avg_answer_chars"]
for j in ("j1", "j2"):
    for subset in ("all", "formal", "colloquial"):
        print(f"\\nОценщик {j}: {g[g.judge == j].judge_model.iloc[0]}; подмножество {subset}" + ("  <- основной" if j == headline else ""))
        display(g[(g.judge == j) & (g.subset == subset)][show].reset_index(drop=True))"""),
    ("code", """fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
for ax, subset in zip(axes, ("formal", "colloquial", "all")):
    d = g[(g.judge == headline) & (g.subset == subset)].reset_index(drop=True)
    x = np.arange(len(d))
    cf = pd.to_numeric(d.correct_full_rate)
    ax.bar(x, cf, yerr=[cf - pd.to_numeric(d.correct_full_lo), pd.to_numeric(d.correct_full_hi) - cf], capsize=3, color="#4c72b0")
    ax.set_xticks(x, (d.config.str.replace("dense_large+rerank", "dense+rerank") + "\\n" + d.prompt).tolist(), rotation=60, ha="right", fontsize=7)
    ax.set_title(f"Полностью верные ответы: {subset}")
    ax.set_ylim(0, 1.05)
plt.suptitle(f"Основной оценщик: {headline}", fontsize=9)
plt.tight_layout()
plt.show()"""),
    ("md", """## 3. Абляция по числу контекстов k (лучшая конфигурация поиска, лучший промпт)

Задержка измерена прямыми вызовами без кэша на одних и тех же 6 вопросах (медиана), поэтому она грубая."""),
    ("code", """ka = pd.read_csv(R / "k_ablation.csv")
cols = ["top_k", "correct_full_rate", "correct_mean_0_2", "faithful_rate", "abstain_correct_rate", "false_abstain_rate", "gold_in_context_rate",
        "avg_context_chars", "avg_prompt_tokens", "latency_median_s"]
for j in ("j1", "j2"):
    for subset in ("all", "colloquial"):
        print(f"\\nОценщик {j}, подмножество {subset}")
        display(ka[(ka.judge == j) & (ka.subset == subset)][cols].reset_index(drop=True))"""),
    ("md", """## 4. Абляции фазы 2: температура генератора и платный генератор

Эти эксперименты (`--with-temperature`, `--with-generator2`) подготовлены в `src/run_generation.py`, но на момент сборки ноутбука могли не выполняться: если файлов результатов нет, ячейка об этом сообщает. Проба влияния температуры на ответы (без кэша) выполнена отдельно."""),
    ("code", """probe = pd.read_csv(R / "temperature_probe.csv")
print("Проба: два независимых вызова на каждый из 5 вопросов, без кэша")
display(probe.groupby("temperature").agg(completion_tokens_mean=("completion_tokens", "mean"), reasoning_tokens_mean=("reasoning_tokens", "mean")).round(0))
for name in ("temperature_ablation", "generator_comparison"):
    path = R / f"{name}.csv"
    if path.exists():
        print(name)
        display(pd.read_csv(path))
    else:
        print(f"{name}: НЕ ВЫПОЛНЕНО (результатов нет)")"""),
    ("md", """## 5. Оценщики: согласие между собой и с ручными метками

Согласие считается по всем сгенерированным ответам. Каппа Коэна поправляет долю совпадений на случайное совпадение. Ручные метки поставлены ассистентом, писавшим код (не автором проекта), на выборке из 10 ответов на проверенные вопросы (для 2 вопросов вне корпуса correctness не определён, поэтому n = 8): выборка мала, выводы по ней ориентировочные."""),
    ("code", """display(pd.read_csv(R / "judge_pairwise.csv"))
path = R / "judge_vs_manual_summary.csv"
if path.exists():
    print("Каждый оценщик против ручных меток (n = 10):")
    display(pd.read_csv(path))
    display(pd.read_csv(R / "judge_agreement.csv"))"""),
    ("md", "## 6. Разбор ошибок генерации\n\nПримеры ответов, которые основной оценщик счёл неверными или неподтверждёнными (из `generation_outputs.jsonl`, основная сетка)."),
    ("code", """out = pd.read_json(R / "generation_outputs.jsonl", lines=True)
grid = out[out.experiment == "grid"]
corr, comm, abst = f"{headline}_correctness", f"{headline}_comment", f"{headline}_abstained"
bad = grid[(grid.answerable) & (grid[corr].fillna(2) < 2)]
print("Не полностью верные ответы по вариантам и подмножествам:")
display(bad.groupby(["retriever", "prompt", "subset"]).size().rename("ответов").to_frame())
bad[["subset", "retriever", "prompt", "qid", "question", "gold_answer", "answer", comm]].head(8)"""),
    ("code", """oos = grid[~grid.answerable]
nonab = oos[oos[abst] != 1]
print("Вопросы вне корпуса: ответы без отказа")
display(nonab.groupby(["retriever", "prompt"]).size().rename("ответов без отказа").to_frame())
nonab[["retriever", "prompt", "qid", "question", "answer", comm]].head(6)"""),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out_dir = ROOT / "notebooks"
    out_dir.mkdir(exist_ok=True)
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out_dir)}})
    nbf.write(nb, out_dir / "results.ipynb")
    print("notebooks/results.ipynb собран и исполнен")


if __name__ == "__main__":
    main()
