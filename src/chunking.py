"""Split transcribed pages into chunks for search. Three strategies.

tests/test_chunking.py describes what each function must do.

Input: pages as a list of (page_no, text), read from cache/md/<pdf stem>/p-NN.md.
Output: a list of chunk records, each one a dict:

    {
        "id": "FYP_notes:fixed:0003",    # <file stem>:<strategy>:<index, 4 digits>, unique per strategy
        "file": "FYP_notes.pdf",
        "pages": [13, 14],               # every physical page the chunk touches
        "heading": "5.2 Generating function" or None,
        "text": "...",
        "method": "vision",              # which ingestion produced the text (raw or vision)
        "strategy": "fixed",             # per_page | fixed | heading
    }

Token rule (decided in DECISIONS.md): one token = one whitespace-separated word.
Sizes are in tokens, so 500 means about 500 words.

Math rule for every strategy: never cut inside a $$...$$ block or a $...$ span.
"""

import bisect
import re
from pathlib import Path

MD_ROOT = Path("cache/md")
MATH_SPAN = re.compile(r"\$\$(.*?)\$\$|\$([^$]*?)\$", re.DOTALL)


def load_pages(md_dir: Path) -> list[tuple[int, str]]:
    """Read every p-NN.md (vision) or p-NN.txt (raw) in md_dir. Return (page_no, text) pairs in page order."""
    pages = []
    for md_path in sorted(p for p in md_dir.glob("p-*") if p.suffix in (".md", ".txt")):
        page_no = int(md_path.stem.split("-")[1])
        pages.append((page_no, md_path.read_text(encoding="utf-8")))
    return pages


def count_tokens(text: str) -> int:
    """Return the token count of text (whitespace-separated words)."""
    return len(text.split())


def safe_cut_points(text: str) -> list[int]:
    """Return character offsets where the text may be cut without breaking math.

    A cut is safe only outside every $...$ and $$...$$ span.
    Also return 0 and len(text), so callers can always use them as bounds.
    """
    cut_points = [i for i in range(len(text) + 1)]

    for match in MATH_SPAN.finditer(text):
        start, end = match.span()
        for nocut in range(start + 1, end):
            cut_points.remove(nocut)
    
    return cut_points


def chunk_per_page(pages: list[tuple[int, str]], file: str, method: str) -> list[dict]:
    """Strategy 1 (baseline): one chunk per page. Pages list is [page_no] and text is the page."""
    chunks = []
    strategy = "per_page"
    for i in range(len(pages)):
        page_no, text = pages[i]
        chunks.append({
            "id": f"{Path(file).stem}:{strategy}:{i + 1:04d}",
            "file": file,
            "pages": [page_no],
            "heading": None,
            "text": text,
            "method": method,
            "strategy": strategy,
            })

    return chunks

def chunk_fixed(
    pages: list[tuple[int, str]],
    file: str,
    method: str,
    size: int = 500,
    overlap: int = 80,
) -> list[dict]:
    """Strategy 2: about `size` tokens per chunk, with `overlap` tokens shared with the next chunk.

    Track which pages each chunk touches. Cut only at safe_cut_points.
    The strategy label is "fixed".
    """
    strategy = "fixed"

    # 1. Join all pages into one string, remembering where each page starts.
    joined = "\n\n".join(text for _, text in pages)
    page_spans = []  # (start_char, end_char, page_no)
    pos = 0
    for page_no, text in pages:
        page_spans.append((pos, pos + len(text), page_no))
        pos += len(text) + 2  # the "\n\n" we put between pages

    # 2. Allowed cut points: safe for math, and just before a space (never inside a word).
    cuts = [
        c for c in safe_cut_points(joined)
        if c in (0, len(joined)) or joined[c].isspace()
    ]

    # 3. Walk through the text, one chunk at a time.
    chunks = []
    start = 0
    while start < len(joined):
        end = _furthest_cut(cuts, joined, start, size)
        if end is None:
            end = len(joined)

        text = joined[start:end].strip()
        pages_hit = sorted({p for a, b, p in page_spans if a < end and b > start})
        if text:
            chunks.append({
                "id": f"{Path(file).stem}:{strategy}:{len(chunks) + 1:04d}",
                "file": file,
                "pages": pages_hit,
                "heading": None,
                "text": text,
                "method": method,
                "strategy": strategy,
            })

        if end >= len(joined):
            break
        start = _overlap_start(cuts, joined, start, end, overlap)

    return chunks


def _furthest_cut(cuts: list[int], joined: str, start: int, budget: int) -> int | None:
    """Largest allowed cut after start with at most `budget` words in joined[start:cut].

    If even the next cut is over budget (one formula bigger than the chunk),
    return that next cut anyway, so we always make progress.
    """
    first = bisect.bisect_right(cuts, start)
    if first >= len(cuts):
        return None
    # count of words grows as the cut moves right, so bisect can find the boundary
    over = bisect.bisect_right(cuts, budget, lo=first, key=lambda c: count_tokens(joined[start:c]))
    if over - 1 < first:
        return cuts[first]
    return cuts[over - 1]


def _overlap_start(cuts: list[int], joined: str, start: int, end: int, overlap: int) -> int:
    """First allowed cut after start where joined[cut:end] has at most `overlap` words."""
    first = bisect.bisect_right(cuts, start)
    last = bisect.bisect_right(cuts, end)
    # words left in joined[cut:end] shrink as the cut moves right, so search on the negated count
    i = bisect.bisect_left(cuts, -overlap, lo=first, hi=last, key=lambda c: -count_tokens(joined[c:end]))
    if i >= last:
        return end
    return cuts[i]


def chunk_by_heading(
    pages: list[tuple[int, str]],
    file: str,
    method: str,
    max_section: int = 800,
    size: int = 500,
    overlap: int = 80,
) -> list[dict]:
    """Strategy 3: split on Markdown headings (## and deeper). Method B only.

    A section longer than max_section tokens is split with chunk_fixed rules.
    Record the heading text in each chunk. The strategy label is "heading".
    """
    strategy = "heading"

    # 1. Split the lines into sections. A "## " line starts a new section.
    #    Each section keeps its text as (page_no, text) pieces, so pages are tracked.
    sections = []  # each: {"heading": str or None, "pieces": [(page_no, text), ...]}
    for page_no, text in pages:
        for line in text.split("\n"):
            if line.startswith("## "):
                sections.append({"heading": line[3:].strip(), "pieces": []})
            if not sections:
                sections.append({"heading": None, "pieces": []})  # text before the first heading
            pieces = sections[-1]["pieces"]
            if pieces and pieces[-1][0] == page_no:
                pieces[-1] = (page_no, pieces[-1][1] + "\n" + line)  # same page: keep adding lines
            else:
                pieces.append((page_no, line))  # new page: start a new piece

    # 2. Short sections become one chunk. Long sections are split with chunk_fixed.
    chunks = []
    for section in sections:
        total = sum(count_tokens(text) for _, text in section["pieces"])
        if total <= max_section:
            chunks.append({
                "id": f"{Path(file).stem}:{strategy}:{len(chunks) + 1:04d}",
                "file": file,
                "pages": sorted({p for p, _ in section["pieces"]}),
                "heading": section["heading"],
                "text": "\n".join(text for _, text in section["pieces"]).strip(),
                "method": method,
                "strategy": strategy,
            })
        else:
            for piece in chunk_fixed(section["pieces"], file, method, size, overlap):
                chunks.append({
                    **piece,
                    "id": f"{Path(file).stem}:{strategy}:{len(chunks) + 1:04d}",
                    "heading": section["heading"],
                    "strategy": strategy,
                })

    return chunks
