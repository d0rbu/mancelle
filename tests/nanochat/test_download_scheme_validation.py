"""
Tests for URL scheme validation in `nanochat.common.download_file_with_lock`.

`download_file_with_lock` accepted `file://` scheme, which permits arbitrary
local-file read if URL is user-configurable.

Invariant under test: only `http` and `https` schemes are accepted.
All other schemes — including empty scheme, `file`, case variants of `file`,
`ftp`, `gopher`, `data`, `javascript`, `ssh`, etc. — must raise `ValueError`
with a message containing "unsupported URL scheme".
"""

from __future__ import annotations

import os

import pytest

from nanochat.common import download_file_with_lock


REJECTED_SCHEMES = [
    "ftp://host/x",
    "gopher://host/x",
    "data:text/plain,x",
    "javascript:alert(1)",
    "ssh://host/x",
    "file:/etc/passwd",
    "",
]


CASE_INSENSITIVE_FILE_SCHEMES = [
    "FILE://x",
    "FiLe:///etc/passwd",
    "file://Y",
]


def test_download_rejects_file_scheme(tmp_path, monkeypatch):
    """`file:///etc/passwd` must be rejected outright, before any filesystem I/O."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="unsupported URL scheme"):
        download_file_with_lock("file:///etc/passwd", "passwd_leak.bin")


@pytest.mark.parametrize("url", REJECTED_SCHEMES)
def test_download_rejects_scheme_parametrize(tmp_path, monkeypatch, url):
    """All non-http(s) schemes (and the bare-string edge case) must raise."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="unsupported URL scheme"):
        download_file_with_lock(url, "rejected_scheme.bin")


def test_download_rejects_empty_scheme(tmp_path, monkeypatch):
    """Bare filesystem-like paths (no scheme at all) must be rejected."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="unsupported URL scheme"):
        download_file_with_lock("/tmp/whatever", "empty_scheme.bin")


@pytest.mark.parametrize("url", CASE_INSENSITIVE_FILE_SCHEMES)
def test_download_rejects_case_insensitive_file_scheme(tmp_path, monkeypatch, url):
    """
    A naive `startswith("file://")` check would miss `FILE://`/`FiLe://`.
    This regression test enforces that the scheme check is case-insensitive.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="unsupported URL scheme"):
        download_file_with_lock(url, "case_scheme.bin")


def test_download_accepts_https(tmp_path, monkeypatch):
    """
    Positive path: an `https://` URL must pass scheme validation and
    reach the underlying HTTP call (which we stub out so no network is touched).
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))

    captured_url: dict[str, str] = {}

    class _FakeResponse:
        def __init__(self, url: str) -> None:
            captured_url["url"] = url

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return b"ok"

    def _fake_urlopen(url, *args, **kwargs):
        return _FakeResponse(url)

    monkeypatch.setattr("nanochat.common.urllib.request.urlopen", _fake_urlopen)

    out_path = download_file_with_lock("https://example.com/x", "https_ok.bin")

    # Validation passed and the stubbed downloader actually executed.
    assert captured_url["url"] == "https://example.com/x"
    assert os.path.exists(out_path)
    with open(out_path, "rb") as f:
        assert f.read() == b"ok"


def test_download_accepts_http(tmp_path, monkeypatch):
    """`http://` must also be accepted (companion to the https positive test)."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))

    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return b"hello"

    monkeypatch.setattr(
        "nanochat.common.urllib.request.urlopen",
        lambda url, *a, **kw: _FakeResponse(),
    )

    out_path = download_file_with_lock("http://example.com/y", "http_ok.bin")
    assert os.path.exists(out_path)
    with open(out_path, "rb") as f:
        assert f.read() == b"hello"
