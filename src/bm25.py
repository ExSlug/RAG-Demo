"""Sparse retrieval: BM25 over chunk text, using the rank_bm25 package.

BM25 scores a chunk by how often the query's words appear in it, with a boost for
rare words and a penalty for very long chunks. It matches exact words, not meaning.
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word tokens. LaTeX commands like \\binom become 'binom'."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def topk(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Return (row_index, bm25_score) for the k best chunks, best first."""
        scores = np.asarray(self.bm25.get_scores(tokenize(query)))
        order = np.argsort(-scores)[: min(k, len(scores))]
        return [(int(i), float(scores[i])) for i in order]
