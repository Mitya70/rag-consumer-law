"""Собирает docs/prompts.md из src/prompts.py, чтобы документация точно совпадала с используемыми промптами."""
from __future__ import annotations

import prompts as P
from common import ROOT


def block(title: str, text: str) -> str:
    return f"### {title}\n\n```text\n{text}\n```\n"


def main() -> None:
    parts = ["# Промпты эксперимента", "",
             "Тексты ниже выгружены из `src/prompts.py` скриптом `src/make_prompts_doc.py`. Фрагменты документов подставляются вместо `{contexts}`: "
             "пять лучших чанков, пронумерованных `[1]`...`[5]`, перед каждым стоит заголовок «ЗоЗПП, ст. N. Название».", "",
             "## Генерация ответа", ""]
    parts.append(block("no_context (контроль без поиска): system", P.NOCONTEXT_SYSTEM))
    parts.append(block("no_context: user", P.NOCONTEXT_USER))
    parts.append(block("naive: system", P.NAIVE_SYSTEM))
    parts.append(block("strict и strict_fewshot: system", P.STRICT_SYSTEM))
    parts.append(block("user-сообщение (naive, strict, strict_fewshot)", P.USER_TEMPLATE))
    for i, ex in enumerate(P.FEWSHOT_EXAMPLES, 1):
        parts.append(block(f"strict_fewshot: пример {i} (user)", P.USER_TEMPLATE.format(contexts=ex["contexts"], question=ex["question"])))
        parts.append(block(f"strict_fewshot: пример {i} (assistant)", ex["answer"]))
    parts += ["## Оценка ответов моделью-оценщиком", "",
              block("system", P.JUDGE_SYSTEM), block("user", P.JUDGE_TEMPLATE),
              block("добавка для контроля без поиска", P.JUDGE_NOCONTEXT_NOTE.strip()),
              block("эталон для вопросов вне корпуса", P.GOLD_NONE)]
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "prompts.md").write_text("\n".join(parts), encoding="utf-8")
    print("docs/prompts.md записан")


if __name__ == "__main__":
    main()
