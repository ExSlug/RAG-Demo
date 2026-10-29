"""Run the retrieval eval and save results/<name>.json. Never write results by hand.

Usage:
    python -m src.eval_run --index per_page_vision --name e1_vision
    python -m src.eval_run --index fixed_vision --retrieval bm25 --name e3_bm25

For every answerable question: retrieve the top `depth` chunks, and score
recall@1/3/5 and MRR with src/metrics.py. Unanswerable questions are skipped here
(abstention is measured in src/eval_generate.py). Results are reported overall and split by math true/false.

Retrieval modes: dense (embeddings), bm25 (exact words), hybrid (both, fused with RRF).
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.bm25 import BM25Index
from src.embed import embed_query, load_index
from src.metrics import mean, recall_at_k, reciprocal_rank
from src.retrieval import dense_topk

QUESTIONS = Path("eval/questions.jsonl")
RESULTS_DIR = Path("results")
KS = (1, 3, 5)


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def retrieve(mode: str, query: str, matrix, bm25: BM25Index, depth: int) -> list[tuple[int, float]]:
    """Return (row, score) pairs, best first, using the chosen retrieval mode."""
    if mode == "dense":
        return dense_topk(embed_query(query), matrix, k=depth)
    if mode == "bm25":
        return bm25.topk(query, k=depth)
    if mode == "hybrid":
        from src.rrf import rrf_fuse  # imported here so dense and bm25 runs never need it

        n = len(matrix)  # fuse the full rankings, then cut to depth
        dense = [row for row, _ in dense_topk(embed_query(query), matrix, k=n)]
        sparse = [row for row, _ in bm25.topk(query, k=n)]
        return rrf_fuse([dense, sparse])[:depth]
    raise ValueError(f"unknown retrieval mode: {mode}")


def score_question(question: dict, mode: str, chunks: list[dict], matrix, bm25: BM25Index, depth: int) -> dict:
    """Retrieve for one question and return its scores and what was retrieved."""
    hits = retrieve(mode, question["question"], matrix, bm25, depth)
    ranked = [chunks[row] for row, _ in hits]
    row = {f"recall@{k}": recall_at_k(question["gold"], ranked, k) for k in KS}
    row["mrr"] = reciprocal_rank(question["gold"], ranked)
    row.update({
        "id": question["id"],
        "type": question["type"],
        "math": question["math"],
        "retrieved": [{"id": c["id"], "pages": c["pages"], "score": round(s, 4)}
                      for c, (_, s) in zip(ranked, hits)],
    })
    return row


def summarise(rows: list[dict]) -> dict:
    """Mean of every metric over the given per-question rows."""
    metrics = [f"recall@{k}" for k in KS] + ["mrr"]
    return {"n": len(rows), **{m: round(mean([r[m] for r in rows]), 4) for m in metrics}}


def run_eval(index_name: str, mode: str = "dense", depth: int = 10) -> dict:
    chunks, matrix = load_index(index_name)
    bm25 = BM25Index(chunks)
    answerable = [q for q in load_questions() if q["type"] != "unanswerable"]
    rows = [score_question(q, mode, chunks, matrix, bm25, depth) for q in answerable]
    return {
        "index": index_name,
        "retrieval": mode,
        "depth": depth,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall": summarise(rows),
        "math": summarise([r for r in rows if r["math"]]),
        "not_math": summarise([r for r in rows if not r["math"]]),
        "per_question": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval eval.")
    parser.add_argument("--index", required=True, help="index name in cache/index/")
    parser.add_argument("--retrieval", default="dense", choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--name", required=True, help="results file name, without .json")
    parser.add_argument("--depth", type=int, default=10, help="how many chunks to retrieve")
    args = parser.parse_args()

    result = run_eval(args.index, args.retrieval, args.depth)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.name}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    for part in ("overall", "math", "not_math"):
        r = result[part]
        print(f"{part:9} n={r['n']:2}  recall@1={r['recall@1']:.2f}  recall@3={r['recall@3']:.2f}"
              f"  recall@5={r['recall@5']:.2f}  MRR={r['mrr']:.2f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
