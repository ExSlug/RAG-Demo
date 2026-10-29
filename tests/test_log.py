import json

from src.log import log_query
from src.log_report import compute_report, read_log


def test_log_query_appends_one_json_line_per_call(tmp_path):
    path = tmp_path / "logs" / "queries.jsonl"
    log_query("q one", "cfg", {"embed": 10.0, "retrieve": 1.0}, ["a:0001"], path=path)
    log_query("q two", "cfg", {"embed": 20.0, "retrieve": 2.0}, ["a:0002"], 100, 20, {"ok": True}, path=path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    second = json.loads(lines[1])
    assert second["query"] == "q two"
    assert second["input_tokens"] == 100
    assert second["citation_check"] == {"ok": True}
    assert "ts" in second


def test_report_percentiles_and_token_totals():
    records = [
        {"latency_ms": {"embed": ms}, "input_tokens": 100, "output_tokens": 10}
        for ms in (10.0, 20.0, 30.0, 40.0, 50.0)
    ]
    records.append({"latency_ms": {"embed": 60.0}, "input_tokens": None, "output_tokens": None})
    report = compute_report(records)
    assert report["n_queries"] == 6
    assert report["latency_ms"]["embed"]["p50"] == 35.0
    assert report["input_tokens"] == 500
    assert report["output_tokens"] == 50


def test_read_log_of_missing_file_is_empty(tmp_path):
    assert read_log(tmp_path / "nope.jsonl") == []
