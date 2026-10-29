"""Build a saved index from the extracted notes.

Usage:
    python -m src.build_index --strategy fixed --method vision
    python -m src.build_index --strategy per_page --method raw

The index name is "<strategy>_<method>", for example cache/index/fixed_vision/.
The heading strategy needs Markdown headings, so it works on vision text only.
"""

import argparse
from pathlib import Path

from src.chunking import chunk_by_heading, chunk_fixed, chunk_per_page, load_pages
from src.embed import save_index

TEXT_DIRS = {"vision": Path("cache/md"), "raw": Path("cache/raw")}
STRATEGIES = {
    "per_page": lambda pages, file, method: chunk_per_page(pages, file, method),
    "fixed": lambda pages, file, method: chunk_fixed(pages, file, method),
    "heading": lambda pages, file, method: chunk_by_heading(pages, file, method),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk the notes and build an embedding index.")
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--method", default="vision", choices=TEXT_DIRS)
    parser.add_argument("--file", default="FYP_notes.pdf", help="PDF name, used in citations")
    args = parser.parse_args()

    if args.strategy == "heading" and args.method == "raw":
        raise SystemExit("The heading strategy needs Markdown headings. Use --method vision.")

    pages = load_pages(TEXT_DIRS[args.method] / Path(args.file).stem)
    chunks = STRATEGIES[args.strategy](pages, args.file, args.method)
    out_dir = save_index(chunks, f"{args.strategy}_{args.method}")
    print(f"{len(chunks)} chunks embedded -> {out_dir}")


if __name__ == "__main__":
    main()
