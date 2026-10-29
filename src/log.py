"""Append one JSON line per query to logs/queries.jsonl (the project spec, section 13).

A record has: ts, query, config, latency_ms (per stage), input_tokens, output_tokens,
chunk_ids, citation_check. Stages or numbers that do not apply are None
(the MCP server never calls an LLM, so it has no generate time and no tokens).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "queries.jsonl"


def log_query(
    query: str,
    config: str,
    latency_ms: dict,
    chunk_ids: list[str],
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    citation_check: dict | None = None,
    path: Path = LOG_PATH,
) -> dict:
    """Append one record to the log file and return it."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query": query,
        "config": config,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "chunk_ids": chunk_ids,
        "citation_check": citation_check,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
