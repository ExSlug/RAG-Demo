"""Turn text into vectors with a local sentence-transformers model. No API key needed.

The model name comes from EMBED_MODEL in .env (the project spec, section 9).
An index is saved to cache/index/<name>/: chunks.jsonl (the chunk records) and
embeddings.npy (one row per chunk, in the same order).
"""

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

INDEX_ROOT = Path("cache/index")

# BGE models were trained with this prefix on search queries (not on the passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def load_model():
    """Load the embedding model once, the first time it is needed."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return one vector per text, as a float32 array of shape (n, dim)."""
    vectors = load_model().encode(texts, batch_size=32, show_progress_bar=True)
    return np.asarray(vectors, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Return the vector for one search query (with the BGE query prefix)."""
    return embed_texts([QUERY_PREFIX + query])[0]


def save_index(chunks: list[dict], name: str, root: Path = INDEX_ROOT) -> Path:
    """Embed every chunk's text and save chunks.jsonl and embeddings.npy. Return the folder."""
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = embed_texts([c["text"] for c in chunks])
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    np.save(out_dir / "embeddings.npy", matrix)
    return out_dir


def load_index(name: str, root: Path = INDEX_ROOT) -> tuple[list[dict], np.ndarray]:
    """Load a saved index. Return (chunks, embeddings) in the same row order."""
    out_dir = root / name
    with open(out_dir / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    matrix = np.load(out_dir / "embeddings.npy")
    return chunks, matrix
