import numpy as np

from src.retrieval import dense_topk, normalize_rows


def test_rows_have_length_one():
    m = np.array([[3.0, 4.0], [0.0, 2.0]])
    out = normalize_rows(m)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_zero_row_stays_zero():
    m = np.array([[0.0, 0.0], [1.0, 0.0]])
    out = normalize_rows(m)
    assert np.allclose(out[0], [0.0, 0.0])


def test_topk_finds_most_similar_row_first():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    query = np.array([1.0, 0.1])
    result = dense_topk(query, matrix, k=3)
    assert result[0][0] == 0


def test_scores_are_sorted_best_first():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    scores = [s for _, s in dense_topk(np.array([1.0, 0.1]), matrix, k=3)]
    assert scores == sorted(scores, reverse=True)


def test_score_is_the_cosine():
    matrix = np.array([[1.0, 0.0], [1.0, 1.0]])
    query = np.array([1.0, 0.0])
    scores = dict(dense_topk(query, matrix, k=2))
    # cos between [1,0] and [1,1] is 1 / sqrt(2)
    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], 1 / np.sqrt(2))


def test_k_is_capped_at_number_of_rows():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert len(dense_topk(np.array([1.0, 0.0]), matrix, k=5)) == 2


def test_k_limits_the_result():
    matrix = np.eye(4)
    assert len(dense_topk(np.array([1.0, 0.0, 0.0, 0.0]), matrix, k=2)) == 2
