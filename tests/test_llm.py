import pytest

from src.llm import require_env


def test_missing_key_exits_with_one_clear_line(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as info:
        require_env("GEMINI_API_KEY")
    assert "Missing GEMINI_API_KEY in .env" in str(info.value)


def test_present_key_is_returned(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "  abc  ")
    assert require_env("GEMINI_API_KEY") == "abc"
