"""Retrieval metrics for the eval set.

A chunk "hits" a gold page if the chunk's `pages` list contains it.
Gold is a list of {"file": ..., "page": ...} dicts, exactly as in eval/questions.jsonl.
A ranked list of chunks is best first. Each chunk has "file" and "pages" (list of ints).

For one question q, with gold page set G and R_k the pages covered by the top-k chunks:

    recall@k(q) = |G intersect R_k| / |G|
    rr(q)       = 1 / rank of the first chunk that hits any gold page, or 0 if none does

Over all answerable questions, report the mean of each. tests/test_metrics.py has examples.
"""


def covered_pages(ranked_chunks: list[dict], k: int) -> set[tuple[str, int]]:
    """Return {(file, page)} for every page that the top-k chunks cover."""
    relevant_chunks = ranked_chunks[: min(k, len(ranked_chunks))]
    list = []
    for chunk in relevant_chunks:
        file = chunk["file"]
        for page in chunk["pages"]:
            list.append((file, page))
    
    return set(list)


def recall_at_k(gold: list[dict], ranked_chunks: list[dict], k: int) -> float:
    """Fraction of the gold pages that appear among the pages the top-k chunks cover."""
    covered = covered_pages(ranked_chunks, k)
    set_gold = set([(g["file"], g["page"]) for g in gold])
    return len(set_gold & covered) / len(set_gold)
    

def reciprocal_rank(gold: list[dict], ranked_chunks: list[dict]) -> float:
    """1 / rank of the first chunk that hits a gold page (rank 1 is the best chunk). 0.0 if none."""
    for idx, chunk in enumerate(ranked_chunks, start=1):
        covered = covered_pages([chunk], 1)
        set_gold = set([(g["file"], g["page"]) for g in gold])
        if covered & set_gold:
            return 1 / idx        
    return 0.0
    

def mean(values: list[float]) -> float:
    """Average of the values. 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0
