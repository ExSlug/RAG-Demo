import src.mcp_server as server


def test_tool_returns_file_page_score_text(monkeypatch):
    chunk = {"id": "FYP_notes:per_page:0014", "file": "FYP_notes.pdf", "pages": [14], "text": "K_r"}
    monkeypatch.setattr(server, "search", lambda q, index, k: {
        "hits": [{"chunk": chunk, "score": 0.81234567}],
        "latency_ms": {"embed": 1.0, "retrieve": 1.0},
    })
    monkeypatch.setattr(server, "log_query", lambda **kwargs: None)
    assert server.search_notes("Krawtchouk", k=1) == [
        {"file": "FYP_notes.pdf", "page": 14, "score": 0.8123, "text": "K_r"}
    ]


def test_tool_has_a_description_the_client_model_can_read():
    assert server.search_notes.__doc__ and "notes" in server.search_notes.__doc__
