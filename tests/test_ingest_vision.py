import pymupdf

from src import ingest_vision
from src.ingest_vision import transcribe_pdf


def make_pdf(path, n_pages):
    doc = pymupdf.open()
    for i in range(n_pages):
        doc.new_page().insert_text((72, 72), f"page {i + 1}")
    doc.save(path)
    doc.close()


def test_transcribe_pdf_writes_one_md_per_page(tmp_path, monkeypatch):
    pdf = tmp_path / "toy.pdf"
    make_pdf(pdf, 3)
    monkeypatch.setattr(ingest_vision, "PAUSE_S", 0)
    monkeypatch.setattr(ingest_vision, "transcribe_with_retry", lambda png: "fake reply")

    out_dir = transcribe_pdf(pdf, out_root=tmp_path / "md")

    assert sorted(p.name for p in out_dir.glob("*.md")) == ["p-01.md", "p-02.md", "p-03.md"]
    assert (out_dir / "p-02.md").read_text(encoding="utf-8") == "fake reply"


def test_cached_pages_are_not_sent_again(tmp_path, monkeypatch):
    pdf = tmp_path / "toy.pdf"
    make_pdf(pdf, 2)
    monkeypatch.setattr(ingest_vision, "PAUSE_S", 0)

    out_dir = tmp_path / "md" / "toy"
    out_dir.mkdir(parents=True)
    (out_dir / "p-01.md").write_text("already done", encoding="utf-8")

    calls = []

    def fake(png):
        calls.append(png)
        return "new"

    monkeypatch.setattr(ingest_vision, "transcribe_with_retry", fake)
    transcribe_pdf(pdf, out_root=tmp_path / "md")

    assert len(calls) == 1  # only page 2 was sent
    assert (out_dir / "p-01.md").read_text(encoding="utf-8") == "already done"
