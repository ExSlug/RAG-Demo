"""Run all eval questions through retrieval, generation and the citation check.

Usage:
    python -m src.eval_generate --name generation_per_page_dense

Reports, and saves to results/<name>.json:
  - citation pass rate over answerable questions that the model answered
  - false abstentions: answerable questions where the model said Not in the notes.
  - abstention rate on unanswerable questions (q19, q20)
  - failure counts by type
"""

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.citation_check import ABSTAIN, check_citations
from src.eval_run import load_questions
from src.generate import DEFAULT_INDEX, generate_answer, retrieve_chunks

RESULTS_DIR = Path("results")
PAUSE_S = 1.0


def run(index: str, k: int) -> dict:
    rows = []
    for q in load_questions():
        answerable = q["type"] != "unanswerable"
        chunks = retrieve_chunks(q["question"], index, k)
        answer = generate_answer(q["question"], chunks)
        check = check_citations(answer["text"], chunks, answerable)
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "type": q["type"],
            "answerable": answerable,
            "answer": answer["text"],
            "abstained": answer["text"].strip() == ABSTAIN,
            "retrieved_pages": [c["pages"] for c in chunks],
            "input_tokens": answer["input_tokens"],
            "output_tokens": answer["output_tokens"],
            **check,
        })
        time.sleep(PAUSE_S)

    answered = [r for r in rows if r["answerable"] and not r["abstained"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    return {
        "index": index,
        "k": k,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "citation_pass_rate": round(sum(r["ok"] for r in answered) / len(answered), 4) if answered else 0.0,
        "n_answered": len(answered),
        "false_abstentions": [r["id"] for r in rows if r["answerable"] and r["abstained"]],
        "abstention_rate_on_unanswerable": round(sum(r["abstained"] for r in unanswerable) / len(unanswerable), 4)
        if unanswerable else 0.0,
        "failure_counts": dict(Counter(f for r in rows for f in r["failures"])),
        "per_question": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generation and citations.")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    result = run(args.index, args.k)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.name}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"citation pass rate: {result['citation_pass_rate']:.2f} over {result['n_answered']} answered questions")
    print(f"false abstentions: {result['false_abstentions'] or 'none'}")
    print(f"abstention rate on unanswerable: {result['abstention_rate_on_unanswerable']:.2f}")
    print(f"failures: {result['failure_counts'] or 'none'}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
