from src.bm25 import BM25Index, tokenize


def test_tokenize_lowercases_and_splits_latex():
    assert tokenize("The $\\binom{n}{k}$ term") == ["the", "binom", "n", "k", "term"]


def test_exact_word_match_ranks_first():
    chunks = [
        {"text": "the hamming distance counts differing coordinates"},
        {"text": "characters are eigenvectors of every distance matrix"},
        {"text": "portfolio losses are exchangeable"},
    ]
    top = BM25Index(chunks).topk("hamming distance", k=3)
    assert top[0][0] == 0


def test_k_is_capped_at_number_of_chunks():
    chunks = [{"text": "alpha beta"}, {"text": "gamma delta"}]
    assert len(BM25Index(chunks).topk("alpha", k=10)) == 2
