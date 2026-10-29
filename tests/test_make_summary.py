from src.make_summary import best_label, table


def result(mrr, r3, math=1.0, not_math=0.5):
    return {
        "overall": {"recall@1": 0.5, "recall@3": r3, "recall@5": 0.9, "mrr": mrr},
        "math": {"mrr": math},
        "not_math": {"mrr": not_math},
    }


def test_best_has_the_highest_mrr():
    rows = [("a", result(0.5, 0.9)), ("b", result(0.7, 0.4))]
    assert best_label(rows) == "b"


def test_mrr_tie_goes_to_recall_at_3():
    rows = [("a", result(0.7, 0.5)), ("b", result(0.7, 0.8))]
    assert best_label(rows) == "b"


def test_best_of_nothing_is_none():
    assert best_label([]) is None


def test_missing_result_is_marked_not_run():
    out = table([("hybrid", None)])
    assert "hybrid | not run" in out


def test_table_formats_numbers_from_the_result():
    out = table([("dense", result(0.77, 0.7, math=1.0, not_math=0.41))])
    assert "0.77" in out and "0.70" in out and "0.41" in out


def test_near_tie_lists_close_runners_up():
    from src.make_summary import near_ties

    rows = [("a", result(0.78, 0.7)), ("b", result(0.77, 0.7)), ("c", result(0.5, 0.7))]
    assert near_ties(rows) == ["b"]
