"""Применяет вердикты ручной проверки (review/review_sample.md) к data/eval_set.csv.

Проверка выполнена пользователем (автором проекта) по 20 вопросам выборки:
  * ok: 15 вопросов приняты как есть;
  * edit-confirmed: q68 и q71 исправлены (новые эталонные ответы и статьи), правка подтверждена пользователем;
  * exclude: q27, q34 (юридизированные или нереалистичные формулировки) и q70 (дубликат q40) исключены из метрик.
Исключённые вопросы тоже проверены человеком (verified_by_human = True), строки не удаляются: excluded = True и причина в exclusion_reason.
Цитаты-основания для правленых вопросов извлекаются из текста корпуса программно (предложения целиком), поэтому гарантированно дословны.
Скрипт идемпотентен. Запуск: python src/apply_review.py
"""
from __future__ import annotations

import csv
import re

from chunking import load_corpus
from common import DATA_DIR, ROOT

FIELDS = ["qid", "question", "gold_answer", "gold_article_id", "gold_article_id_2", "evidence", "qtype", "answerable", "source",
          "subset", "verified_by_human", "excluded", "exclusion_reason"]

# (номер в review_sample.md, qid, вердикт, примечание)
VERDICTS = [
    (1, "q09", "ok", ""), (2, "q11", "ok", ""), (3, "q15", "ok", ""), (4, "q16", "ok", ""), (5, "q18", "ok", ""), (6, "q26", "ok", ""),
    (7, "q27", "exclude", "исключён ревьюером; причина в явном виде не указана (в итоговой сводке отнесён к юридизированным/нереалистичным вопросам)"),
    (8, "q28", "ok", ""),
    (9, "q34", "exclude", "юридизированный вопрос, который реальный потребитель не задал бы"),
    (10, "q35", "ok", ""), (11, "q40", "ok", ""), (12, "q44", "ok", ""), (13, "q52", "ok", ""), (14, "q45", "ok", ""), (15, "q46", "ok", ""),
    (16, "q63", "ok", ""),
    (17, "q68", "edit-confirmed", "эталонный ответ и статьи переписаны по правке ревьюера: ст. 14 + ст. 15 (возмещение вреда здоровью и компенсация морального вреда)"),
    (18, "q69", "ok", ""),
    (19, "q70", "exclude", "дубликат q40: оба вопроса про автомат, который забрал деньги и ничего не выдал"),
    (20, "q71", "edit-confirmed", "эталонный ответ и статьи переписаны по правке ревьюера: ст. 18 (проверка качества, экспертиза, бремя доказывания) + ст. 22 (срок возврата денег)"),
]

EDITS = {
    "q68": {
        "gold_answer": "С владельца кафе (исполнителя услуги / продавца) можно требовать возмещения вреда здоровью в полном объёме (ст. 14 ЗоЗПП); "
        "это право есть у любого потерпевшего независимо от договорных отношений. Также можно требовать компенсации морального вреда (ст. 15).",
        "gold_article_id": "zpp:14", "gold_article_id_2": "zpp:15",
        # (статья, регулярное выражение для поиска предложения-цитаты)
        "quotes": [("zpp:14", r"признаётся за любым потерпевшим независимо"), ("zpp:14", r"подлежит возмещению в полном объёме"),
                   ("zpp:15", r"^Моральный вред")],
    },
    "q71": {
        "gold_answer": "Магазин обязан принять товар и при необходимости провести проверку качества; потребитель вправе участвовать. Если спор о причинах недостатка, "
        "продавец обязан провести экспертизу за свой счёт, потребитель вправе присутствовать и оспорить заключение в суде (ст. 18, п. 5). "
        "Если на товар установлен гарантийный срок, продавец отвечает за недостаток, пока не докажет, что он возник после передачи товара из-за нарушения "
        "потребителем правил использования, хранения или транспортировки (ст. 18, п. 6): доказывать должен продавец. Если экспертиза покажет, что продавец "
        "не отвечает, потребитель возместит расходы на неё (ст. 18, п. 5). Требование о возврате денег удовлетворяется в течение десяти дней со дня "
        "предъявления (ст. 22).",
        "gold_article_id": "zpp:18", "gold_article_id_2": "zpp:22",
        "quotes": [("zpp:18", r"обязаны принять товар ненадлежащего качества"), ("zpp:18", r"^Потребитель вправе участвовать в проверке качества"),
                   ("zpp:18", r"обязаны провести экспертизу товара за свой счёт"), ("zpp:18", r"^Потребитель вправе присутствовать при проведении экспертизы"),
                   ("zpp:18", r"отвечает за недостатки товара, если не докажет"), ("zpp:18", r"потребитель обязан возместить продавцу"),
                   ("zpp:22", r"десяти дней со дня предъявления")],
    },
}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;:])\s+", text.replace("\n", " ")) if s.strip()]


def extract_quotes(corpus: dict, quotes: list[tuple[str, str]]) -> list[str]:
    result = []
    for article_id, pattern in quotes:
        found = [s for s in split_sentences(corpus[article_id]["text"]) if re.search(pattern, s)]
        if not found:
            raise SystemExit(f"Не найдено предложение для {article_id}: {pattern}")
        result.append(found[0].rstrip(" ;:."))
    return result


def main() -> None:
    corpus = {a["article_id"]: a for a in load_corpus()}
    path = DATA_DIR / "eval_set.csv"
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_qid = {r["qid"]: r for r in rows}
    for r in rows:
        r.setdefault("excluded", "")
        r["excluded"] = r.get("excluded") or "False"
        r["exclusion_reason"] = r.get("exclusion_reason") or ""

    for _, qid, verdict, note in VERDICTS:
        r = by_qid[qid]
        r["verified_by_human"] = "True"  # все 20 вопросов выборки просмотрены человеком
        if verdict == "exclude":
            r["excluded"], r["exclusion_reason"] = "True", note
        elif verdict == "edit-confirmed":
            edit = EDITS[qid]
            r["gold_answer"], r["gold_article_id"], r["gold_article_id_2"] = edit["gold_answer"], edit["gold_article_id"], edit["gold_article_id_2"]
            r["evidence"] = " || ".join(extract_quotes(corpus, edit["quotes"]))
            if not r["source"].endswith("+human_edit"):
                r["source"] += "+human_edit"

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)

    review_dir = ROOT / "review"
    review_dir.mkdir(exist_ok=True)
    with open(review_dir / "human_verdicts.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["review_no", "qid", "verdict", "note"])
        writer.writerows(VERDICTS)

    kept = [r for r in rows if r["excluded"] != "True"]
    verified = [r for r in kept if r["verified_by_human"] == "True"]
    print(f"всего {len(rows)}; исключено {len(rows) - len(kept)}; в итоговом наборе {len(kept)}; проверено человеком в итоговом наборе {len(verified)}; "
          f"черновиков без ручной проверки {len(kept) - len(verified)}")


STATUS_LINE = (
    "Статус: **проверено человеком: 20 из 72 вопросов.** Принято 17 (15 как есть, 2 после правки пользователя), исключено 3 "
    "(q27 и q34 как юридизированные или нереалистичные, q70 как дубликат q40). Итоговый набор: 69 вопросов, из них 17 проверены человеком, "
    "52 являются черновиками LLM без ручной проверки. Ревьюер отклонил 3 из 20 черновиков (15%), поэтому непроверенная часть, вероятно, "
    "содержит похожие недочёты. Вердикты: `human_verdicts.csv`."
)


def update_review_sample() -> None:
    """Обновляет статус и вердикты в review/review_sample.md, не пересоздавая выборку (решения ревьюера сохраняются)."""
    path = ROOT / "review" / "review_sample.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"Статус: \*\*.*", STATUS_LINE, text, count=1)
    text = re.sub(r"Отметьте результат и впишите правку, если нужна\. Затем поставьте `verified_by_human = True`.*?в `data/eval_set\.csv`\.",
                  "Вердикты ревьюера внесены в `data/eval_set.csv` (колонки `verified_by_human`, `excluded`, `exclusion_reason`) и в `human_verdicts.csv`.",
                  text, flags=re.S)
    verdicts = {qid: (verdict, note) for _, qid, verdict, note in VERDICTS}
    edits = {qid: e["gold_answer"] for qid, e in EDITS.items()}
    parts = re.split(r"(?m)^(?=## \d+\. )", text)
    out = [parts[0]]
    for part in parts[1:]:
        qid = re.match(r"## \d+\. (q\d+)", part).group(1)
        verdict, note = verdicts[qid]
        label = {"ok": "принят как есть", "exclude": "исключён из набора", "edit-confirmed": "эталон исправлен ревьюером, правка подтверждена"}[verdict]
        line = f"**Вердикт ревьюера:** {label}." + (f" {note}." if note and verdict == "exclude" else "")
        if qid in edits:
            line += f"\n\n**Новый эталонный ответ (после правки):** {edits[qid]}"
        part = re.sub(r"(?:\n\n)*\*\*Новый эталонный ответ \(после правки\):\*\*[^\n]*", "", part)  # повторный запуск: убираем прежнюю вставку
        part = re.sub(r"(?m)^- \[ \] вопрос и эталон верны.*$", lambda m: line, part)
        part = re.sub(r"(?m)^\*\*Вердикт ревьюера:\*\*[^\n]*$", lambda m: line, part)
        out.append(part)
    path.write_text("".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
    update_review_sample()
