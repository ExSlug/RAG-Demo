"""Method A: raw text extraction with PyMuPDF. No vision model, no API calls.

This is the baseline. It reads the PDF's own text layer, one page at a time.
Output: cache/raw/<pdf stem>/p-NN.txt (NN is the 1-based physical page index).

Usage:
    python -m src.ingest_raw data/pdfs/FYP_notes.pdf
    python -m src.ingest_raw data/pdfs/FYP_notes.pdf --render 14
"""

import argparse
from pathlib import Path

import pymupdf

CACHE_RAW = Path("cache/raw")
CACHE_PAGES = Path("cache/pages")


def extract_pages(pdf_path: Path) -> list[str]:
    """Return the text layer of every page, in page order."""
    with pymupdf.open(pdf_path) as doc:
        return [page.get_text() for page in doc]


def write_raw(pdf_path: Path, out_root: Path = CACHE_RAW) -> Path:
    """Write one .txt per page into out_root/<pdf stem>/. Return that folder."""
    out_dir = out_root / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for page_no, text in enumerate(extract_pages(pdf_path), start=1):
        (out_dir / f"p-{page_no:02d}.txt").write_text(text, encoding="utf-8")
    return out_dir


def render_page(pdf_path: Path, page_no: int, out_root: Path = CACHE_PAGES, dpi: int = 150) -> Path:
    """Render one page to PNG so you can compare it with the raw text by eye."""
    out_dir = out_root / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"p-{page_no:02d}.png"
    with pymupdf.open(pdf_path) as doc:
        doc[page_no - 1].get_pixmap(dpi=dpi).save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Method A: raw PyMuPDF text per page.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--render", type=int, metavar="PAGE", help="also render this page to PNG")
    args = parser.parse_args()

    out_dir = write_raw(args.pdf)
    n_pages = len(list(out_dir.glob("p-*.txt")))
    print(f"wrote {n_pages} pages to {out_dir}")

    if args.render:
        png = render_page(args.pdf, args.render)
        print(f"rendered page {args.render} to {png}")


if __name__ == "__main__":
    main()
