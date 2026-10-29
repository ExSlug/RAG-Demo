"""Interactive search: type a question, see the top chunks with their scores.

Usage:
    python -m src.search_cli --index fixed_vision --k 5

Type a question and press Enter. Type an empty line to quit.
"""

import argparse

from src.embed import embed_query, load_index
from src.retrieval import dense_topk


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the notes with dense retrieval.")
    parser.add_argument("--index", default="fixed_vision", help="index name in cache/index/")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    chunks, matrix = load_index(args.index)
    print(f"loaded {len(chunks)} chunks from {args.index}. Empty line to quit.")

    while True:
        query = input("\nquery> ").strip()
        if not query:
            break
        hits = dense_topk(embed_query(query), matrix, k=args.k)
        for rank, (row, score) in enumerate(hits, start=1):
            chunk = chunks[row]
            pages = ", ".join(str(p) for p in chunk["pages"])
            preview = " ".join(chunk["text"].split())[:150]
            print(f"{rank}. [{chunk['file']} p.{pages}] score={score:.3f}  {preview}...")


if __name__ == "__main__":
    main()
