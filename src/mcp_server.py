"""MCP server: exposes the notes search as one tool. It only retrieves, it never calls an LLM.

The client (Claude Code) starts this file as a child process and talks to it over
stdio (standard input and output). Because of that, NEVER print() in this file:
stdout carries the protocol messages, and a stray print would corrupt them.
Logs go to logs/queries.jsonl instead.

Run by hand to check that it starts (it then waits for a client):
    python -m src.mcp_server
"""

import os
import threading

# The model is already downloaded. Offline mode skips the slow checks against the
# HuggingFace website, which otherwise add many seconds to the first search.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from mcp.server.mcpserver import MCPServer  # noqa: E402

from src.embed import load_model  # noqa: E402
from src.log import log_query  # noqa: E402
from src.search import DEFAULT_INDEX, search  # noqa: E402

mcp = MCPServer("notes")


@mcp.tool()
def search_notes(query: str, k: int = 5) -> list[dict]:
    """Search the user's personal lecture notes (PDF pages) and return the best-matching pages.

    Use this whenever the user asks about their notes, or about definitions, theorems or
    formulas that may be in them. Each result has: file, page (physical page number, starting
    at 1), score (cosine similarity, higher is closer), and text (the page as Markdown with
    LaTeX). Cite results as [file p.N]. If no result answers the question, say the notes do
    not cover it instead of using outside knowledge.

    Args:
        query: what to look for, in plain English or with the key terms.
        k: how many pages to return (default 5).
    """
    result = search(query, DEFAULT_INDEX, k)
    hits = [
        {
            "file": h["chunk"]["file"],
            "page": h["chunk"]["pages"][0],  # per_page chunks cover exactly one page
            "score": round(h["score"], 4),
            "text": h["chunk"]["text"],
        }
        for h in result["hits"]
    ]
    log_query(
        query=query,
        config=f"{DEFAULT_INDEX}/dense/k={k}",
        latency_ms=result["latency_ms"],
        chunk_ids=[h["chunk"]["id"] for h in result["hits"]],
    )
    return hits


if __name__ == "__main__":
    # Load the embedding model in the background so the first search is not slow,
    # while the server can already answer the client's handshake.
    threading.Thread(target=load_model, daemon=True).start()
    mcp.run(transport="stdio")
