"""Method B: vision transcription. Each page image goes to the vision model, and the reply is saved as Markdown.

Output: cache/md/<pdf stem>/p-NN.md (NN is the 1-based physical page index).
A page is skipped if its .md file already exists, so a crashed run can be resumed.

Usage:
    python -m src.ingest_vision data/pdfs/FYP_notes.pdf
"""

import argparse
import time
from pathlib import Path

import pymupdf

from src.llm import transcribe_image

CACHE_MD = Path("cache/md")
DPI = 150
PAUSE_S = 1.0          # short gap between pages, to stay under rate limits
MAX_RETRIES = 5        # retries for rate limits and server errors

# Verbatim from the project spec, section 7. Keep it verbatim: it is the agreed transcription prompt.
PROMPT = """Transcribe this page to Markdown. Rules:
- Write all math in LaTeX: inline as $...$, display as $$...$$.
- Keep section headings as Markdown headings.
- Copy text exactly. Do not summarize or fix anything.
- If a symbol is unclear, write [UNCLEAR] instead of guessing."""

RETRY_CODES = {429, 500, 503}


def render_png(pdf_path: Path, page_no: int, dpi: int = DPI) -> bytes:
    """Render one page to PNG bytes at the given resolution."""
    with pymupdf.open(pdf_path) as doc:
        return doc[page_no - 1].get_pixmap(dpi=dpi).tobytes("png")


def transcribe_with_retry(png_bytes: bytes) -> str:
    """Call the vision model. On a rate limit or server error, wait and try again."""
    from google.genai import errors

    for attempt in range(MAX_RETRIES):
        try:
            return transcribe_image(png_bytes, PROMPT)
        except errors.APIError as exc:
            if exc.code not in RETRY_CODES or attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt  # 1, 2, 4, 8 seconds
            print(f"  API error {exc.code}, retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def transcribe_pdf(pdf_path: Path, out_root: Path = CACHE_MD) -> Path:
    """Transcribe every page that has no .md yet. Return the output folder."""
    out_dir = out_root / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    with pymupdf.open(pdf_path) as doc:
        n_pages = doc.page_count

    for page_no in range(1, n_pages + 1):
        out_path = out_dir / f"p-{page_no:02d}.md"
        if out_path.exists():
            print(f"page {page_no:02d}: cached, skipping")
            continue
        print(f"page {page_no:02d}: transcribing")
        reply = transcribe_with_retry(render_png(pdf_path, page_no))
        out_path.write_text(reply, encoding="utf-8")
        time.sleep(PAUSE_S)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Method B: vision transcription to Markdown.")
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()
    out_dir = transcribe_pdf(args.pdf)
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
