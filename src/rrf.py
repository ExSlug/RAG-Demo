"""Reciprocal rank fusion (RRF).

RRF merges several rankings of the same chunks into one. Each ranking gives a chunk
a score of 1 / (k + rank), where rank is its position in that ranking counting from 1.
A chunk's final score is the sum over every ranking that contains it:

    RRF(d) = sum over rankings r of  1 / (k + rank_r(d)),     k = 60 by default

A chunk ranked high in several lists beats a chunk ranked first in only one.
tests/test_rrf.py has worked examples.
"""


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse rankings into one list of (row_index, rrf_score), best first.

    Each ranking is a list of row indices, best first. A row missing from a ranking
    gets nothing from that ranking. Return every row that appears in any ranking.
    """
    scores = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):        
            scores[row] = scores.get(row, 0) + 1 / (k + rank)

    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    return [(key, value) for key, value in sorted_scores.items()]
    