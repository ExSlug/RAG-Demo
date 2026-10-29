"""Check eval/questions.jsonl before it is used (the project spec, section 11).

Checks: valid JSON, unique ids, the fields exist, every gold file and page exists in
data/pdfs, the gold count matches the question type, and the mix is 14 single /
4 multi / 2 unanswerable with at least 6 math questions.

Usage:
    python -m src.validate_questions
"""

import json
import sys
from pathlib import Path

import pymupdf

QUESTIONS = Path("eval/questions.jsonl")
PDF_DIR = Path("data/pdfs")
TYPES = {"single", "multi", "unanswerable"}
EXPECTED_MIX = {"single": 14, "multi": 4, "unanswerable": 2}
MIN_MATH = 6


def load_questions(path: Path) -> tuple[list[dict], list[str]]:
    """Read the file. Return (questions, problems). Bad lines are reported, not raised."""
    questions, problems = [], []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            questions.append(json.loads(line))
        except json.JSONDecodeError as exc:
            problems.append(f"line {n}: not valid JSON ({exc.msg})")
    return questions, problems


def page_counts(pdf_dir: Path) -> dict[str, int]:
    """Return {pdf file name: number of pages} for every PDF in pdf_dir."""
    counts = {}
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        with pymupdf.open(pdf) as doc:
            counts[pdf.name] = doc.page_count
    return counts


def validate(questions: list[dict], pdf_pages: dict[str, int]) -> list[str]:
    """Return a list of problems. An empty list means the file is good."""
    problems = []
    seen = set()
    for q in questions:
        qid = q.get("id", "<no id>")
        for field in ("id", "question", "type", "math", "gold"):
            if field not in q:
                problems.append(f"{qid}: missing field '{field}'")
        if qid in seen:
            problems.append(f"{qid}: duplicate id")
        seen.add(qid)

        qtype, gold = q.get("type"), q.get("gold", [])
        if qtype not in TYPES:
            problems.append(f"{qid}: type must be one of {sorted(TYPES)}, got {qtype!r}")
        elif qtype == "single" and len(gold) != 1:
            problems.append(f"{qid}: a single question needs exactly 1 gold page, has {len(gold)}")
        elif qtype == "multi" and len(gold) < 2:
            problems.append(f"{qid}: a multi question needs 2 or more gold pages, has {len(gold)}")
        elif qtype == "unanswerable" and gold:
            problems.append(f"{qid}: an unanswerable question must have empty gold")

        for g in gold:
            file, page = g.get("file"), g.get("page")
            if file not in pdf_pages:
                problems.append(f"{qid}: gold file {file!r} is not in data/pdfs")
            elif not isinstance(page, int) or not 1 <= page <= pdf_pages[file]:
                problems.append(f"{qid}: page {page!r} is not in {file} (1 to {pdf_pages[file]})")

    mix = {t: sum(1 for q in questions if q.get("type") == t) for t in TYPES}
    for t, want in EXPECTED_MIX.items():
        if mix[t] != want:
            problems.append(f"mix: expected {want} {t} questions, found {mix[t]}")
    n_math = sum(1 for q in questions if q.get("math") is True)
    if n_math < MIN_MATH:
        problems.append(f"mix: need at least {MIN_MATH} math questions, found {n_math}")
    return problems


def main() -> None:
    if not QUESTIONS.exists():
        sys.exit(f"{QUESTIONS} does not exist yet. Write your questions there, one JSON per line.")
    questions, problems = load_questions(QUESTIONS)
    problems += validate(questions, page_counts(PDF_DIR))
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print(f"OK: {len(questions)} questions, mix and gold pages all valid.")


if __name__ == "__main__":
    main()
