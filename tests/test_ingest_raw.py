import pymupdf

from src.ingest_raw import extract_pages, write_raw


def make_pdf(path, pages):
    """Build a small PDF with one text string per page."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_extract_pages_returns_one_string_per_page(tmp_path):
    pdf = tmp_path / "toy.pdf"
    make_pdf(pdf, ["first page words", "second page words"])

    texts = extract_pages(pdf)

    assert len(texts) == 2
    assert "first page" in texts[0]
    assert "second page" in texts[1]


def test_write_raw_names_files_by_physical_page(tmp_path):
    pdf = tmp_path / "toy.pdf"
    make_pdf(pdf, ["alpha", "beta", "gamma"])

    out_dir = write_raw(pdf, out_root=tmp_path / "raw")

    assert out_dir == tmp_path / "raw" / "toy"
    assert (out_dir / "p-01.txt").read_text(encoding="utf-8").strip() == "alpha"
    assert (out_dir / "p-03.txt").read_text(encoding="utf-8").strip() == "gamma"
