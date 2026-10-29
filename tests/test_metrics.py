import pytest

from src.metrics import covered_pages, mean, recall_at_k, reciprocal_rank

F = "FYP_notes.pdf"


def chunk(pages):
    return {"file": F, "pages": pages}


def gold(*pages):
    return [{"file": F, "page": p} for p in pages]


# --- covered_pages ---

def test_covered_pages_takes_only_the_top_k():
    ranked = [chunk([1, 2]), chunk([2, 3])]
    assert covered_pages(ranked, 1) == {(F, 1), (F, 2)}
    assert covered_pages(ranked, 2) == {(F, 1), (F, 2), (F, 3)}


def test_covered_pages_k_larger_than_list():
    assert covered_pages([chunk([4])], 10) == {(F, 4)}


# --- recall@k ---

def test_recall_is_one_when_gold_page_is_covered():
    assert recall_at_k(gold(14), [chunk([14, 15])], k=1) == 1.0


def test_recall_is_zero_when_gold_page_is_not_covered():
    assert recall_at_k(gold(14), [chunk([1, 2])], k=1) == 0.0


def test_recall_counts_the_fraction_of_gold_pages():
    ranked = [chunk([2]), chunk([5])]
    assert recall_at_k(gold(2, 5), ranked, k=1) == 0.5
    assert recall_at_k(gold(2, 5), ranked, k=2) == 1.0


def test_recall_ignores_pages_of_a_different_file():
    other = [{"file": "other.pdf", "pages": [14]}]
    assert recall_at_k(gold(14), other, k=1) == 0.0


# --- reciprocal rank ---

def test_reciprocal_rank_first_hit_at_rank_one():
    assert reciprocal_rank(gold(3), [chunk([3]), chunk([9])]) == 1.0


def test_reciprocal_rank_first_hit_at_rank_three():
    ranked = [chunk([1]), chunk([2]), chunk([3])]
    assert reciprocal_rank(gold(3), ranked) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_the_first_hit_only():
    ranked = [chunk([1]), chunk([3]), chunk([3])]
    assert reciprocal_rank(gold(3), ranked) == 0.5


def test_reciprocal_rank_is_zero_when_nothing_hits():
    assert reciprocal_rank(gold(3), [chunk([1]), chunk([2])]) == 0.0


# --- mean ---

def test_mean():
    assert mean([1.0, 0.0, 0.5]) == pytest.approx(0.5)


def test_mean_of_empty_list_is_zero():
    assert mean([]) == 0.0
