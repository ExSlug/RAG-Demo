from src.chunking import (
    chunk_by_heading,
    chunk_fixed,
    chunk_per_page,
    count_tokens,
    safe_cut_points,
)


def words(n, tag="w"):
    """A string of n plain words, so token counts are exact."""
    return " ".join(f"{tag}{i}" for i in range(n))


def inline_math_count_is_even(text):
    return text.replace("$$", "").count("$") % 2 == 0


def display_count_is_even(text):
    return text.count("$$") % 2 == 0


# --- helpers ---

def test_count_tokens_counts_words():
    assert count_tokens("one two  three\nfour") == 4


def test_safe_cut_points_skip_inside_math():
    text = "a $x + y$ b"
    cuts = safe_cut_points(text)
    # positions strictly inside the span: after the opening $ up to and including the position before the closing $
    inside = range(text.index("$") + 1, text.index("$", text.index("$") + 1) + 1)
    assert all(c not in inside for c in cuts)
    assert 0 in cuts and len(text) in cuts


# --- strategy 1: per_page ---

def test_per_page_one_chunk_per_page():
    pages = [(1, "first page text"), (2, "second page text")]
    chunks = chunk_per_page(pages, file="FYP_notes.pdf", method="vision")

    assert len(chunks) == 2
    assert chunks[0]["pages"] == [1]
    assert chunks[1]["text"] == "second page text"
    assert all(c["strategy"] == "per_page" for c in chunks)
    assert all(c["method"] == "vision" for c in chunks)


def test_per_page_ids_are_unique():
    pages = [(1, "a"), (2, "b"), (3, "c")]
    chunks = chunk_per_page(pages, file="FYP_notes.pdf", method="raw")
    assert len({c["id"] for c in chunks}) == 3


# --- strategy 2: fixed ---

def test_fixed_chunks_are_about_500_tokens():
    pages = [(1, words(1200))]
    chunks = chunk_fixed(pages, file="FYP_notes.pdf", method="vision")

    assert len(chunks) >= 2
    for c in chunks[:-1]:  # the last chunk can be shorter
        assert 450 <= count_tokens(c["text"]) <= 550


def test_fixed_chunks_overlap_by_80_tokens():
    pages = [(1, words(1200))]
    chunks = chunk_fixed(pages, file="FYP_notes.pdf", method="vision")

    first, second = chunks[0]["text"].split(), chunks[1]["text"].split()
    assert first[-80:] == second[:80]


def test_fixed_chunk_tracks_pages_it_spans():
    pages = [(13, words(300, "a")), (14, words(300, "b"))]
    chunks = chunk_fixed(pages, file="FYP_notes.pdf", method="vision")

    spanning = [c for c in chunks if c["pages"] == [13, 14]]
    assert len(spanning) >= 1


# --- strategy 3: heading ---

def test_heading_split_records_heading_text():
    text = "## Intro\nSome intro words.\n\n## Method\nSome method words."
    chunks = chunk_by_heading([(1, text)], file="FYP_notes.pdf", method="vision")

    assert [c["heading"] for c in chunks] == ["Intro", "Method"]
    assert all(c["strategy"] == "heading" for c in chunks)


def test_long_heading_section_falls_back_to_fixed():
    text = "## Long section\n" + words(1500)
    chunks = chunk_by_heading([(1, text)], file="FYP_notes.pdf", method="vision")

    assert len(chunks) >= 2
    assert all(c["heading"] == "Long section" for c in chunks)


# --- the math rule (the project spec, section 8) ---

def display_block_straddling_cut(filler_tokens):
    """Text where a long $$...$$ block sits across the 500-token cut point."""
    return (
        words(filler_tokens)
        + "\n\n$$"
        + words(120, "m")
        + "$$\n\n"
        + words(600, "after")
    )


def test_fixed_never_splits_display_math():
    text = display_block_straddling_cut(440)
    chunks = chunk_fixed([(1, text)], file="FYP_notes.pdf", method="vision")

    assert len(chunks) >= 2
    for c in chunks:
        assert display_count_is_even(c["text"])
        assert inline_math_count_is_even(c["text"])


def test_fixed_never_splits_inline_math():
    text = words(495) + " $a_1 + a_2 + a_3$ " + words(400, "tail")
    chunks = chunk_fixed([(1, text)], file="FYP_notes.pdf", method="vision")

    for c in chunks:
        assert inline_math_count_is_even(c["text"])


def test_heading_fallback_never_splits_display_math():
    text = "## Section\n" + display_block_straddling_cut(440)
    chunks = chunk_by_heading([(1, text)], file="FYP_notes.pdf", method="vision")

    for c in chunks:
        assert display_count_is_even(c["text"])
        assert inline_math_count_is_even(c["text"])
