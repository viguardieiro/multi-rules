"""Tests for Ollama selector backend with mocked HTTP calls."""

from __future__ import annotations

import json
import urllib.error

import pytest

from src.dynamic_boost.selector_llm import OllamaSelectorBackend


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_ollama_backend_builds_request_and_returns_content(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse({"message": {"content": "ok-json"}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    backend = OllamaSelectorBackend(model="gpt-oss:20b", base_url="http://127.0.0.1:11434")
    out = backend.generate("sys", {"k": "v"}, timeout_s=12.5)

    assert out == "ok-json"
    assert captured["url"].endswith("/api/chat")
    assert captured["timeout"] == 12.5
    assert captured["body"]["model"] == "gpt-oss:20b"
    assert captured["body"]["messages"][0]["role"] == "system"


def test_ollama_backend_raises_on_bad_payload(monkeypatch):
    def fake_urlopen(req, timeout):
        return _FakeHTTPResponse({"message": {"not_content": "x"}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    backend = OllamaSelectorBackend(model="llama3.2:latest")
    with pytest.raises(RuntimeError, match="message content"):
        backend.generate("sys", {"k": "v"}, timeout_s=5)


def test_ollama_backend_wraps_url_errors(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("down")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    backend = OllamaSelectorBackend(model="llama3.2:latest")
    with pytest.raises(RuntimeError, match="Failed to call Ollama"):
        backend.generate("sys", {"k": "v"}, timeout_s=5)
