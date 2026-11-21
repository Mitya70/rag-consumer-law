"""Согласие моделей-оценщиков друг с другом и с ручными метками.

Оценщики: j1 (основной, JUDGE_MODEL), j2 (JUDGE_MODEL_2) и j3 (JUDGE_MODEL_3, кросс-проверка только на ответах
на проверенные человеком вопросы лучшей конфигурации). Ручные метки (results/manual_labels.csv) поставлены на
выборке из 10 ответов ассистентом, который писал код, а не автором проекта. Выборка маленькая, поэтому сравнение
оценщиков с ручными метками грубое.
Запуск: python src/judge_agreement.py
"""
from __future__ import annotations

import csv
import json

import numpy as np

from common import RESULTS_DIR


def kappa(a: list, b: list) -> float:
    """Каппа Коэна для двух разметок одних и тех же объектов (категориальные значения)."""
    a, b = np.array(a, dtype=object), np.array(b, dtype=object)
    cats = sorted(set(a) | set(b), key=str)
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in cats)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def pair_stats(name: str, a: list, b: list, extra: dict | None = None) -> dict:
    keep = [i for i, (x, y) in enumerate(zip(a, b)) if x is not None and y is not None and x == x and y == y]
    a, b = [a[i] for i in keep], [b[i] for i in keep]
    return {**(extra or {}), "metric": name, "n": len(a),
            "agreement": round(float(np.mean(np.array(a, dtype=object) == np.array(b, dtype=object))), 3) if a else None,
            "kappa": round(kappa(a, b), 3) if a else None}


def full(x):
    return None if x is None else int(x == 2)


def fields(records: list[dict], judge: str) -> dict:
    return {
        "correctness (0/1/2)": [r.get(f"{judge}_correctness") for r in records],
        "полностью верно (correctness = 2)": [full(r.get(f"{judge}_correctness")) for r in records],
        "faithful (без контроля без поиска)": [None if r["prompt"] == "no_context" else r.get(f"{judge}_faithful") for r in records],
        "abstained": [r.get(f"{judge}_abstained") for r in records],
    }


def main() -> None:
    all_records = [json.loads(line) for line in open(RESULTS_DIR / "generation_outputs.jsonl", encoding="utf-8")]
    grid = [r for r in all_records if r["experiment"] == "grid"]  # ручные метки ставились на ответах основной сетки (k = 3)
    rows = []
    # 1. пары оценщиков на всех ответах (j1-j2) и на подмножестве j3 (проверенные вопросы лучшей конфигурации)
    pairs = [("j1", "j2", all_records, "все ответы"), ("j1", "j3", [r for r in all_records if r.get("j3_abstained") is not None or r.get("j3_correctness") is not None], "ответы на проверенные вопросы"),
             ("j2", "j3", [r for r in all_records if r.get("j3_abstained") is not None or r.get("j3_correctness") is not None], "ответы на проверенные вопросы")]
    for a, b, recs, scope in pairs:
        fa, fb = fields(recs, a), fields(recs, b)
        for name in fa:
            rows.append(pair_stats(name, fa[name], fb[name], {"pair": f"{a} vs {b}", "scope": scope}))
    with open(RESULTS_DIR / "judge_pairwise.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["pair", "scope", "metric", "n", "agreement", "kappa"])
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)

    # 2. каждый оценщик против ручных меток
    path = RESULTS_DIR / "manual_labels.csv"
    if not path.exists():
        print("manual_labels.csv не найден: сравнение с ручными метками пропущено")
        return
    manual = {(m["qid"], m["chunking"], m["retriever"], m["prompt"]): m for m in csv.DictReader(open(path, encoding="utf-8"))}
    index = {(r["qid"], r["chunking"], r["retriever"], r["prompt"]): r for r in grid}

    def norm(x):
        return None if x in ("", "n/a", None) else int(float(x))

    detail, summary = [], []
    for key, m in manual.items():
        r = index[key]
        detail.append({"qid": key[0], "chunking": key[1], "retriever": key[2], "prompt": key[3], "subset": r["subset"],
                       "manual_correctness": m["manual_correctness"], "j1_correctness": r.get("j1_correctness"), "j2_correctness": r.get("j2_correctness"),
                       "j3_correctness": r.get("j3_correctness"), "manual_abstained": m["manual_abstained"], "j1_abstained": r.get("j1_abstained"),
                       "j2_abstained": r.get("j2_abstained"), "j3_abstained": r.get("j3_abstained"), "manual_faithful": m["manual_faithful"],
                       "j1_faithful": r.get("j1_faithful"), "j2_faithful": r.get("j2_faithful"), "j3_faithful": r.get("j3_faithful"), "note": m["note"]})
    for judge in ("j1", "j2", "j3"):
        for field in ("correctness", "abstained", "faithful"):
            man = [norm(d[f"manual_{field}"]) for d in detail]
            jud = [norm(d[f"{judge}_{field}"]) for d in detail]
            summary.append(pair_stats(field, man, jud, {"judge": judge}))
    with open(RESULTS_DIR / "judge_agreement.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(detail[0].keys()))
        writer.writeheader()
        writer.writerows(detail)
    with open(RESULTS_DIR / "judge_vs_manual_summary.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["judge", "metric", "n", "agreement", "kappa"])
        writer.writeheader()
        writer.writerows(summary)
    for s in summary:
        print("с ручными метками:", s)
    # основной оценщик: j1 по заданию координатора; для сравнения считается средняя доля совпадений с ручными метками по correctness и abstained
    score = {j: float(np.mean([s["agreement"] for s in summary if s["judge"] == j and s["metric"] in ("correctness", "abstained") and s["agreement"] is not None]))
             for j in ("j1", "j2", "j3")}
    print("средняя доля совпадений с ручными метками (correctness + abstained):", {k: round(v, 3) for k, v in score.items()})
    (RESULTS_DIR / "headline_judge.txt").write_text("j1", encoding="utf-8")


if __name__ == "__main__":
    main()
