"""Поисковики: BM25, TF-IDF по символьным n-граммам, плотные эмбеддинги (удалённые), гибрид и реранк.

Все поисковики возвращают ранжированный список индексов чанков.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from chunking import tokenize
from common import cfg, embed_remote, rerank_remote


class BM25:
    """Классический Okapi BM25 (k1=1.5, b=0.75) на токенах-основах."""

    def __init__(self, texts: list[str], prefix: int | None = 5, k1: float = 1.5, b: float = 0.75):
        self.prefix, self.k1, self.b = prefix, k1, b
        self.docs = [Counter(tokenize(t, prefix)) for t in texts]
        self.len = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg = self.len.mean()
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(texts)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query, self.prefix)
        out = np.zeros(len(self.docs))
        for i, d in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            for w in q:
                if w in d:
                    out[i] += self.idf[w] * d[w] * (self.k1 + 1) / (d[w] + norm)
        return out


class TfidfChar:
    """TF-IDF по символьным n-граммам (3-5) внутри слов: устойчив к падежам и окончаниям."""

    def __init__(self, texts: list[str]):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True)
        self.mat = self.vec.fit_transform(texts)

    def scores(self, query: str) -> np.ndarray:
        return (self.mat @ self.vec.transform([query]).T).toarray().ravel()


class Dense:
    """Плотный поиск: эмбеддинги чанков считаются один раз (кэшируются на диске)."""

    def __init__(self, texts: list[str], model: str):
        self.model = model
        self.mat = embed_remote(model, texts, input_type="passage")

    def scores_many(self, queries: list[str]) -> np.ndarray:
        q = embed_remote(self.model, queries, input_type="query")
        return q @ self.mat.T


def rrf(rank_lists: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion: объединение нескольких ранжирований в одну оценку."""
    total = np.zeros(len(rank_lists[0]))
    for scores in rank_lists:
        order = np.argsort(-scores)
        ranks = np.empty(len(scores))
        ranks[order] = np.arange(1, len(scores) + 1)
        total += 1.0 / (k + ranks)
    return total


def rerank_top(query: str, texts: list[str], order: np.ndarray, top_n: int = 20) -> np.ndarray:
    """Переранжирует первые top_n кандидатов моделью-реранкером, хвост оставляет как есть."""
    head = order[:top_n]
    scores = rerank_remote(cfg("RERANK_MODEL"), query, [texts[i] for i in head])
    new_head = head[np.argsort(-np.array(scores), kind="stable")]
    return np.concatenate([new_head, order[top_n:]])
