"""Summarise logs/queries.jsonl: p50 and p95 latency per stage, and total tokens.

Usage:
    python -m src.log_report
"""

import json
from pathlib import Path

import numpy as np

from src.log import LOG_PATH


def read_log(path: Path = LOG_PATH) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_report(records: list[dict]) -> dict:
    """Return {"n_queries", "latency_ms": {stage: {"p50", "p95"}}, "input_tokens", "output_tokens"}."""
    by_stage: dict[str, list[float]] = {}
    for record in records:
        for stage, ms in record["latency_ms"].items():
            if ms is not None:
                by_stage.setdefault(stage, []).append(ms)
    return {
        "n_queries": len(records),
        "latency_ms": {
            stage: {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95))}
            for stage, v in by_stage.items()
        },
        "input_tokens": sum(r["input_tokens"] or 0 for r in records),
        "output_tokens": sum(r["output_tokens"] or 0 for r in records),
    }


def main() -> None:
    records = read_log()
    if not records:
        print(f"no queries logged yet ({LOG_PATH})")
        return
    report = compute_report(records)
    print(f"queries: {report['n_queries']}")
    for stage, p in report["latency_ms"].items():
        print(f"{stage:>9}: p50 {p['p50']:.1f} ms   p95 {p['p95']:.1f} ms")
    print(f"tokens: in {report['input_tokens']}  out {report['output_tokens']}")


if __name__ == "__main__":
    main()
