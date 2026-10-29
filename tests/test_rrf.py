import pytest

from src.rrf import rrf_fuse


def test_single_ranking_keeps_its_order_and_scores_by_rank():
    result = rrf_fuse([[5, 7]], k=0)
    assert result == [(5, 1.0), (7, 0.5)]


def test_default_k_is_60():
    result = dict(rrf_fuse([[3]]))
    assert result[3] == pytest.approx(1 / 61)


def test_a_chunk_in_two_rankings_adds_both_scores():
    result = dict(rrf_fuse([[0, 1], [1, 0]], k=0))
    assert result[0] == pytest.approx(1 / 1 + 1 / 2)
    assert result[1] == pytest.approx(1 / 2 + 1 / 1)


def test_agreement_beats_a_single_high_rank():
    # row 0 is 1st and 2nd, row 2 is 3rd and 1st, row 1 is 2nd and 3rd
    result = rrf_fuse([[0, 1, 2], [2, 0, 1]])
    assert [row for row, _ in result] == [0, 2, 1]


def test_a_chunk_missing_from_one_ranking_still_appears():
    result = dict(rrf_fuse([[0, 1], [0]]))
    assert set(result) == {0, 1}
    assert result[1] == pytest.approx(1 / 62)


def test_result_is_sorted_best_first():
    scores = [s for _, s in rrf_fuse([[4, 2, 9], [9, 4, 2]])]
    assert scores == sorted(scores, reverse=True)
