def test_skeleton_imports():
    import src  # noqa: F401


def test_pymupdf_available():
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()
    assert doc.page_count == 1
