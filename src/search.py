"""One search call that also measures how long each stage took.

Used by the MCP server and by generate.py, so both log the same fields.
Paths are anchored to the project folder, so it works no matter which folder
the caller (for example Claude Code) starts the process from.
"""

import time
from functools import lru_cache
from pathlib import Path

from src.embed import embed_query, load_index
from src.retrieval import dense_topk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_ROOT = PROJECT_ROOT / "cache" / "index"
DEFAULT_INDEX = "per_page_vision"  # best config from E1 to E3 (see results/summary.md)


@lru_cache(maxsize=4)
def _index(name: str):
    """Read the saved index from disk once, then reuse it."""
    return load_index(name, root=INDEX_ROOT)


def search(query: str, index: str = DEFAULT_INDEX, k: int = 5) -> dict:
    """Return {"hits": [{"chunk", "score"}, ...], "latency_ms": {"embed", "retrieve"}}."""
    t0 = time.perf_counter()
    query_vector = embed_query(query)
    t1 = time.perf_counter()
    chunks, matrix = _index(index)
    ranked = dense_topk(query_vector, matrix, k=k)
    t2 = time.perf_counter()
    return {
        "hits": [{"chunk": chunks[row], "score": score} for row, score in ranked],
        "latency_ms": {"embed": round((t1 - t0) * 1000, 1), "retrieve": round((t2 - t1) * 1000, 1)},
    }
