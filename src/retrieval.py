"""Dense retrieval with NumPy.

Cosine similarity is a dot product after each vector is scaled to length 1:

    cos(q, d) = q . d / (|q| |d|)

tests/test_retrieval.py describes what each function must do.
"""

import numpy as np


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to length 1. A row of all zeros stays all zeros.

    Input shape (n, dim). Output shape (n, dim).
    """
    normalizer = np.sqrt(np.sum(matrix**2, axis=1))
    normalizer[normalizer == 0] = 1.0
    return (matrix.T / normalizer).T


def dense_topk(query: np.ndarray, matrix: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    """Return (row_index, cosine_score) for the k rows most similar to query, best first.

    Normalize the query and the matrix first (use normalize_rows).
    If k is bigger than the number of rows, return every row.
    """
    n_query = query / np.linalg.norm(query)
    n_matrix = normalize_rows(matrix)
    cosine_scores = n_matrix @ n_query

    order = np.argsort(-cosine_scores)
    sliced_order = order[:min(k, len(order))]
    list = [(int(i), float(cosine_scores[i])) for i in sliced_order]

    return list
