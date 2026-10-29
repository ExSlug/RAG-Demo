"""Answer a question from the retrieved chunks, with citations.

The prompt shows each chunk under a label like [FYP_notes.pdf p.4]. The model must cite
those labels, or reply with exactly ABSTAIN when the chunks do not answer the question.
Checking the answer is src/citation_check.py's job, not this file's.

Usage:
    python -m src.generate "What is regularity?"
"""

import argparse

import time

from src.citation_check import ABSTAIN, check_citations
from src.embed import embed_query, load_index
from src.llm import generate_text
from src.log import log_query
from src.retrieval import dense_topk
from src.search import DEFAULT_INDEX, search

INSTRUCTIONS = f"""You answer questions about a set of lecture notes. Use ONLY the excerpts below.

Rules:
- After every factual statement, cite its source with the label shown above the excerpt, for example [FYP_notes.pdf p.4]. Use one tag per page.
- Write maths in LaTeX: inline as $...$, display as $$...$$.
- If the excerpts do not contain the answer, reply with exactly: {ABSTAIN}
- Do not use knowledge from outside the excerpts."""


def label(chunk: dict) -> str:
    """The citation label(s) printed above a chunk, one tag per page."""
    return " ".join(f"[{chunk['file']} p.{page}]" for page in chunk["pages"])


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Instructions, then each chunk under its label, then the question."""
    excerpts = "\n\n".join(f"{label(c)}\n{c['text']}" for c in chunks)
    return f"{INSTRUCTIONS}\n\nExcerpts:\n\n{excerpts}\n\nQuestion: {question}\nAnswer:"


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """Return {"text", "input_tokens", "output_tokens"} for the model's answer."""
    return generate_text(build_prompt(question, chunks))


def retrieve_chunks(question: str, index: str = DEFAULT_INDEX, k: int = 5) -> list[dict]:
    chunks, matrix = load_index(index)
    return [chunks[row] for row, _ in dense_topk(embed_query(question), matrix, k=k)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about the notes.")
    parser.add_argument("question")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    found = search(args.question, args.index, args.k)
    chunks = [h["chunk"] for h in found["hits"]]
    print("retrieved:", ", ".join(label(c) for c in chunks))

    t0 = time.perf_counter()
    answer = generate_answer(args.question, chunks)
    latency = {**found["latency_ms"], "generate": round((time.perf_counter() - t0) * 1000, 1)}
    print("\n" + answer["text"])
    print(f"\n[tokens in={answer['input_tokens']} out={answer['output_tokens']}]")

    check = check_citations(answer["text"], chunks, answerable=True)
    print("citation check:", check)
    log_query(
        query=args.question,
        config=f"{args.index}/dense/k={args.k}",
        latency_ms=latency,
        chunk_ids=[c["id"] for c in chunks],
        input_tokens=answer["input_tokens"],
        output_tokens=answer["output_tokens"],
        citation_check=check,
    )


if __name__ == "__main__":
    main()
