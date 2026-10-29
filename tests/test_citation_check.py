from src.citation_check import ABSTAIN, check_citations, extract_citations

F = "FYP_notes.pdf"


def chunk(*pages):
    return {"file": F, "pages": list(pages), "text": "..."}


RETRIEVED = [chunk(15), chunk(16)]


# --- extract_citations ---

def test_extract_one_tag():
    assert extract_citations("K_r is defined so. [FYP_notes.pdf p.15]") == {(F, 15)}


def test_extract_several_tags_and_remove_duplicates():
    answer = "A [FYP_notes.pdf p.15]. B [FYP_notes.pdf p.16]. C [FYP_notes.pdf p.15]."
    assert extract_citations(answer) == {(F, 15), (F, 16)}


def test_extract_no_tags():
    assert extract_citations("No sources here.") == set()


def test_extract_ignores_a_tag_without_a_file_name():
    assert extract_citations("See [p.4] for details.") == set()


def test_extract_two_digit_pages():
    assert extract_citations("[FYP_notes.pdf p.22]") == {(F, 22)}


# --- check_citations: answerable questions ---

def test_valid_citation_passes():
    result = check_citations("Defined by a sum. [FYP_notes.pdf p.15]", RETRIEVED, answerable=True)
    assert result == {"ok": True, "failures": [], "cited": [(F, 15)]}


def test_answer_without_citation_is_missing_citation():
    result = check_citations("It is defined by a sum.", RETRIEVED, answerable=True)
    assert result["ok"] is False
    assert result["failures"] == ["missing_citation"]


def test_citing_a_page_that_was_not_retrieved_fails():
    result = check_citations("A sum. [FYP_notes.pdf p.14]", RETRIEVED, answerable=True)
    assert result["failures"] == ["unretrieved_citation"]


def test_one_good_and_one_bad_citation_still_fails_once():
    answer = "A [FYP_notes.pdf p.15]. B [FYP_notes.pdf p.3]. C [FYP_notes.pdf p.9]."
    result = check_citations(answer, RETRIEVED, answerable=True)
    assert result["failures"] == ["unretrieved_citation"]


def test_citation_to_a_page_inside_a_multi_page_chunk_counts_as_retrieved():
    result = check_citations("X [FYP_notes.pdf p.17]", [chunk(16, 17, 18)], answerable=True)
    assert result["ok"] is True


def test_abstaining_on_an_answerable_question_passes_the_citation_check():
    assert check_citations(ABSTAIN, RETRIEVED, answerable=True)["ok"] is True


# --- check_citations: unanswerable questions ---

def test_abstaining_on_an_unanswerable_question_is_correct():
    result = check_citations(ABSTAIN, RETRIEVED, answerable=False)
    assert result == {"ok": True, "failures": [], "cited": []}


def test_abstention_ignores_spaces_and_newlines_at_the_ends():
    assert check_citations("  " + ABSTAIN + "\n", RETRIEVED, answerable=False)["ok"] is True


def test_answering_an_unanswerable_question_fails():
    result = check_citations("Volatility is modelled by... [FYP_notes.pdf p.15]", RETRIEVED, answerable=False)
    assert result["ok"] is False
    assert result["failures"] == ["answered_when_unanswerable"]


def test_a_sentence_that_contains_the_phrase_is_not_an_abstention():
    answer = "That is " + ABSTAIN + " But I think it is a sum."
    result = check_citations(answer, RETRIEVED, answerable=False)
    assert result["failures"] == ["answered_when_unanswerable"]
