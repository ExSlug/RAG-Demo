from src.citation_check import ABSTAIN
from src.generate import build_prompt, label

CHUNKS = [
    {"file": "FYP_notes.pdf", "pages": [4], "text": "Hamming distance counts differing coordinates."},
    {"file": "FYP_notes.pdf", "pages": [15, 16], "text": "The Krawtchouk polynomial is defined by a sum."},
]


def test_label_has_one_tag_per_page():
    assert label(CHUNKS[0]) == "[FYP_notes.pdf p.4]"
    assert label(CHUNKS[1]) == "[FYP_notes.pdf p.15] [FYP_notes.pdf p.16]"


def test_prompt_shows_every_chunk_under_its_label():
    prompt = build_prompt("What is it?", CHUNKS)
    assert "[FYP_notes.pdf p.4]\nHamming distance counts differing coordinates." in prompt
    assert "[FYP_notes.pdf p.15] [FYP_notes.pdf p.16]\nThe Krawtchouk polynomial" in prompt


def test_prompt_uses_the_abstain_phrase_from_the_check():
    assert ABSTAIN in build_prompt("What is it?", CHUNKS)


def test_prompt_ends_with_the_question():
    prompt = build_prompt("What is regularity?", CHUNKS)
    assert prompt.rstrip().endswith("Question: What is regularity?\nAnswer:")
