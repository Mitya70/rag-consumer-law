"""Построение оценочного набора.

1. Черновики вопросов с эталонными ответами и цитатами-доказательствами генерирует JUDGE_MODEL по тексту статьи.
2. Программный фильтр: цитата должна дословно входить в статью, вопрос не должен ссылаться на номер статьи,
   дубликаты по лексике отбрасываются.
3. Вопросы без ответа в корпусе (проверка отказа от ответа) написаны вручную, проверяется отсутствие ключевых слов.

Колонка verified_by_human по умолчанию False: набор проверен только программно.
Запуск: python src/build_eval_set.py
"""
from __future__ import annotations

import csv
import json
import re

from chunking import load_corpus, tokenize
from common import RESULTS_DIR, DATA_DIR, cfg, chat, parallel_map

CANDIDATES = [
    "zpp:4", "zpp:5", "zpp:6", "zpp:7", "zpp:8", "zpp:10", "zpp:12", "zpp:13", "zpp:14", "zpp:15", "zpp:16", "zpp:17",
    "zpp:18", "zpp:19", "zpp:20", "zpp:21", "zpp:22", "zpp:23", "zpp:23.1", "zpp:24", "zpp:25", "zpp:26.1", "zpp:26.2",
    "zpp:27", "zpp:28", "zpp:29", "zpp:30", "zpp:31", "zpp:32", "zpp:33", "zpp:34", "zpp:35", "zpp:36", "zpp:39.1",
    "gk:492", "gk:493", "gk:494", "gk:495", "gk:497", "gk:498", "gk:499", "gk:500", "gk:502", "gk:503",
]
TYPES = ["срок", "право потребителя", "исключение или ограничение", "ответственность или санкция", "расчёт или размер"]

# (вопрос, ключевые слова, которых не должно быть в корпусе: иначе у вопроса мог бы найтись ответ)
UNANSWERABLE = [
    ("Какой штраф по КоАП РФ грозит продавцу за продажу товара без кассового чека?", ["коап"]),
    ("Как рассчитывается НДС при возврате товара покупателем?", ["ндс"]),
    ("В каком размере страховая компания выплачивает возмещение по ОСАГО при ДТП?", ["осаго"]),
    ("Какие требования предъявляются к маркировке алкогольной продукции?", ["акциз", "егаис"]),
    ("Какой срок годности установлен для питьевого молока техническим регламентом?", ["молок"]),
    ("Какие налоги платит самозанятый и в каком размере?", ["самозанят"]),
    ("Как вернуть авиабилет и какая часть стоимости будет возвращена?", ["авиа", "воздушн"]),
    ("Как получить налоговый вычет за лечение?", ["вычет"]),
    ("Что такое ипотечные каникулы и на какой срок они предоставляются?", ["ипотек", "каникул"]),
    ("Какой максимальный размер процентной ставки по потребительскому кредиту установил Банк России?", ["банк россии", "процентн"]),
]

# модель иногда возвращает собственные метки типа (в том числе на английском): приводим к единому набору
TYPE_MAP = {
    "срок": "срок", "сроки выполнения": "срок",
    "право потребителя": "право потребителя", "consumer_right": "право потребителя", "consumer right": "право потребителя",
    "termination": "право потребителя", "качество": "право потребителя", "условие": "право потребителя",
    "ограничение": "исключение или ограничение", "исключение": "исключение или ограничение",
    "исключение или ограничение": "исключение или ограничение",
    "responsibility": "ответственность или санкция", "ответственность": "ответственность или санкция",
    "санкция": "ответственность или санкция", "ответственность или санкция": "ответственность или санкция",
    "penalty for delay": "ответственность или санкция",
    "размер": "расчёт или размер", "расчёт": "расчёт или размер", "расчёт или размер": "расчёт или размер",
    "regulation": "порядок и определения", "procedure": "порядок и определения", "definition": "порядок и определения",
}

PROMPT = """Ты помогаешь составить оценочный набор для системы вопросов и ответов по законодательству о защите прав потребителей.

Ниже текст одной статьи. Составь ОДИН вопрос и эталонный ответ строго по этому тексту.

Требования:
- вопрос звучит как обычный вопрос потребителя, понятен без самой статьи и не содержит слов «статья», «ст.», «закон», «кодекс», номеров статей;
- желательный тип вопроса: {qtype} (если по тексту такой вопрос не составить, выбери ближайший подходящий и укажи его в поле type);
- ответ в одном-двух предложениях и содержит конкретику из текста (сроки, числа, условия);
- evidence: одна-две цитаты из статьи, ДОСЛОВНО скопированные, на которых основан ответ;
- ничего не выдумывай и не добавляй знания вне текста.

Ответь только JSON без пояснений и без markdown:
{{"question": "...", "answer": "...", "evidence": ["...", "..."], "type": "..."}}

{doc}, статья {no}. {title}

{text}"""


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ").replace("«", '"').replace("»", '"')).strip().lower()


def parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a, 5)), set(tokenize(b, 5))
    return len(sa & sb) / max(1, len(sa | sb))


def main() -> None:
    corpus = {a["article_id"]: a for a in load_corpus()}
    model = cfg("JUDGE_MODEL")

    def gen(item):
        i, aid = item
        art = corpus[aid]
        prompt = PROMPT.format(qtype=TYPES[i % len(TYPES)], doc=art["doc_short"], no=art["article_no"], title=art["title"], text=art["text"][:6000])
        out = chat(model, [{"role": "user", "content": prompt}], max_tokens=4000)
        return aid, out["content"]

    drafts = parallel_map(gen, list(enumerate(CANDIDATES)), workers=3)
    rows, rejected = [], []
    for aid, raw in drafts:
        data = parse_json(raw)
        art = corpus[aid]
        if not data or not all(k in data for k in ("question", "answer", "evidence")):
            rejected.append((aid, "не разобран JSON"))
            continue
        q, ans, ev = data["question"].strip(), data["answer"].strip(), data["evidence"]
        if re.search(r"стать[яиеь]|\bст\.|закон|кодекс|\d+\.\d+", q, re.I):
            rejected.append((aid, "вопрос ссылается на статью/закон"))
            continue
        article_text = norm(art["text"])
        good = [e for e in ev if norm(e).strip(" .;:") and norm(e).strip(" .;:") in article_text]
        if not good:
            rejected.append((aid, "цитата не найдена в статье"))
            continue
        if any(jaccard(q, r["question"]) > 0.6 for r in rows):
            rejected.append((aid, "дубликат по лексике"))
            continue
        rows.append(
            {"question": q, "gold_answer": ans, "gold_article_id": aid, "evidence": " || ".join(good),
             "qtype": TYPE_MAP.get(str(data.get("type", "")).strip().lower(), "порядок и определения"), "answerable": True, "source": "llm_draft_judge_model"}
        )

    corpus_text = norm(" ".join(a["text"] for a in corpus.values()))
    kept_oos = []
    for q, words in UNANSWERABLE:
        hits = [w for w in words if w in corpus_text]
        if hits:
            rejected.append((q, f"в корпусе встречаются слова {hits}: ответ мог бы найтись, вопрос оставлен для ручной проверки"))
            continue
        kept_oos.append(q)
    for q in kept_oos:
        rows.append({"question": q, "gold_answer": "", "gold_article_id": "", "evidence": "", "qtype": "вне корпуса",
                     "answerable": False, "source": "manual_out_of_scope"})

    for i, r in enumerate(rows, 1):
        r["qid"] = f"q{i:02d}"
        r["verified_by_human"] = False
    RESULTS_DIR.mkdir(exist_ok=True)
    fields = ["qid", "question", "gold_answer", "gold_article_id", "evidence", "qtype", "answerable", "source", "verified_by_human"]
    with open(DATA_DIR / "eval_set.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)
    ans = sum(r["answerable"] for r in rows)
    print(f"формальных вопросов: {len(rows)} (с ответом {ans}, без ответа {len(rows) - ans}); отброшено: {len(rejected)}")
    for r in rejected:
        print("  отброшено:", r)
    # разговорное подмножество дописывается отдельным скриптом (формальные строки не меняются)
    from add_colloquial import main as add_colloquial
    from apply_review import main as apply_review
    add_colloquial()
    apply_review()  # вердикты ручной проверки (review/human_verdicts.csv)


if __name__ == "__main__":
    main()
