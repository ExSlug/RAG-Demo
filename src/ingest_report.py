"""Run latex_check on every transcribed page and write results/ingest_report.json.

Usage:
    python -m src.ingest_report cache/md/FYP_notes
"""

import argparse
import json
from pathlib import Path

from src.latex_check import check_page

RESULTS = Path("results/ingest_report.json")


def build_report(md_dir: Path) -> dict:
    """Check each p-NN.md in md_dir. Return {page: report} plus a summary."""
    pages = {}
    for md_path in sorted(md_dir.glob("p-*.md")):
        page_no = int(md_path.stem.split("-")[1])
        pages[page_no] = check_page(md_path.read_text(encoding="utf-8"))
    flagged = [p for p, r in pages.items() if not r["ok"]]
    return {
        "source": str(md_dir),
        "n_pages": len(pages),
        "n_flagged": len(flagged),
        "flagged_pages": flagged,
        "pages": pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the per-page ingest report.")
    parser.add_argument("md_dir", type=Path)
    args = parser.parse_args()

    report = build_report(args.md_dir)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{report['n_pages']} pages checked, {report['n_flagged']} flagged: {report['flagged_pages']}")
    print(f"report: {RESULTS}")


if __name__ == "__main__":
    main()
