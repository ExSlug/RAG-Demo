from src.validate_questions import validate

PAGES = {"FYP_notes.pdf": 26}


def q(qid, qtype="single", math=False, gold=None):
    if gold is None:
        gold = [{"file": "FYP_notes.pdf", "page": 1}] if qtype == "single" else []
    return {"id": qid, "question": "?", "type": qtype, "math": math, "gold": gold}


def full_set():
    """A valid 20-question set: 14 single, 4 multi, 2 unanswerable, 6 math."""
    two = [{"file": "FYP_notes.pdf", "page": 1}, {"file": "FYP_notes.pdf", "page": 2}]
    qs = [q(f"s{i}", "single", math=i < 6) for i in range(14)]
    qs += [q(f"m{i}", "multi", gold=two) for i in range(4)]
    qs += [q(f"u{i}", "unanswerable") for i in range(2)]
    return qs


def test_valid_set_has_no_problems():
    assert validate(full_set(), PAGES) == []


def test_page_out_of_range_is_reported():
    qs = full_set()
    qs[0]["gold"] = [{"file": "FYP_notes.pdf", "page": 27}]
    assert any("page 27" in p for p in validate(qs, PAGES))


def test_unknown_file_is_reported():
    qs = full_set()
    qs[0]["gold"] = [{"file": "other.pdf", "page": 1}]
    assert any("other.pdf" in p for p in validate(qs, PAGES))


def test_wrong_mix_is_reported():
    qs = full_set()[:-1]  # drops one unanswerable question
    assert any("unanswerable" in p for p in validate(qs, PAGES))


def test_too_few_math_questions_is_reported():
    qs = full_set()
    for item in qs:
        item["math"] = False
    assert any("math" in p for p in validate(qs, PAGES))


def test_duplicate_id_is_reported():
    qs = full_set()
    qs[1]["id"] = qs[0]["id"]
    assert any("duplicate" in p for p in validate(qs, PAGES))


def test_unanswerable_with_gold_is_reported():
    qs = full_set()
    qs[-1]["gold"] = [{"file": "FYP_notes.pdf", "page": 3}]
    assert any("unanswerable" in p and "empty gold" in p for p in validate(qs, PAGES))
