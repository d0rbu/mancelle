"""Offline and exact-content contracts for the vendored spelling asset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import tasks.spellingbee as spellingbee
from errors import DataLoadError


def test_vendored_asset_has_the_pinned_bytes() -> None:
    path = spellingbee.VENDORED_WORD_LIST_PATH
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == spellingbee.VENDORED_WORD_LIST_SHA256
    assert path.read_text(encoding="utf-8").count("\n") == 370_105


def test_offline_resolution_prefers_operator_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = tmp_path / spellingbee.WORD_LIST_FILENAME
    cached.write_text("alpha\nbeta\n", encoding="utf-8")
    monkeypatch.setattr(spellingbee, "get_base_dir", lambda: tmp_path)
    monkeypatch.setattr(
        spellingbee,
        "VENDORED_WORD_LIST_PATH",
        tmp_path / "missing-vendored-asset.txt",
    )
    monkeypatch.setattr(
        spellingbee,
        "download_file_with_lock",
        lambda *_args, **_kwargs: pytest.fail("offline resolution attempted network"),
    )

    assert spellingbee.ensure_word_list_available(allow_network=False) == cached
    assert spellingbee.resolve_word_list_path(allow_network=False) == cached


def test_offline_resolution_uses_the_vendored_copy_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(spellingbee, "get_base_dir", lambda: tmp_path)
    monkeypatch.setattr(
        spellingbee,
        "download_file_with_lock",
        lambda *_args, **_kwargs: pytest.fail("offline resolution attempted network"),
    )

    path = spellingbee.ensure_word_list_available(allow_network=False)
    assert path == spellingbee.VENDORED_WORD_LIST_PATH
    assert spellingbee.SimpleSpelling(size=3, split="train").words
    assert spellingbee.SpellingBee(size=3, split="train").words


def test_vendored_and_cached_asset_task_outputs_are_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_bytes = spellingbee.VENDORED_WORD_LIST_PATH.read_bytes()
    cached_dir = tmp_path / "cached"
    cached_dir.mkdir()
    (cached_dir / spellingbee.WORD_LIST_FILENAME).write_bytes(asset_bytes)

    monkeypatch.setattr(spellingbee, "get_base_dir", lambda: cached_dir)
    cached_simple = spellingbee.SimpleSpelling(size=64, split="train")
    cached_bee = spellingbee.SpellingBee(size=64, split="train")

    monkeypatch.setattr(spellingbee, "get_base_dir", lambda: tmp_path / "empty")
    vendored_simple = spellingbee.SimpleSpelling(size=64, split="train")
    vendored_bee = spellingbee.SpellingBee(size=64, split="train")

    assert cached_simple.words == vendored_simple.words
    assert cached_bee.words == vendored_bee.words
    assert [cached_simple[i] for i in range(len(cached_simple))] == [
        vendored_simple[i] for i in range(len(vendored_simple))
    ]
    assert [cached_bee[i] for i in range(len(cached_bee))] == [
        vendored_bee[i] for i in range(len(vendored_bee))
    ]


def test_offline_resolution_fails_actionably_when_both_sources_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing" / spellingbee.WORD_LIST_FILENAME
    monkeypatch.setattr(spellingbee, "get_base_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(spellingbee, "VENDORED_WORD_LIST_PATH", missing)

    with pytest.raises(DataLoadError, match="NANOCHAT_BASE_DIR.*live GitHub"):
        spellingbee.ensure_word_list_available(allow_network=False)


def test_online_resolution_keeps_the_legacy_downloader_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    vendor_path = tmp_path / "missing-vendored.txt"
    downloaded = cache_dir / spellingbee.WORD_LIST_FILENAME
    calls: list[tuple[str, str]] = []

    def fake_download(url: str, filename: str) -> str:
        calls.append((url, filename))
        cache_dir.mkdir()
        downloaded.write_text("alpha\n", encoding="utf-8")
        return str(downloaded)

    monkeypatch.setattr(spellingbee, "get_base_dir", lambda: cache_dir)
    monkeypatch.setattr(spellingbee, "VENDORED_WORD_LIST_PATH", vendor_path)
    monkeypatch.setattr(spellingbee, "download_file_with_lock", fake_download)

    assert spellingbee.resolve_word_list_path(allow_network=True) == downloaded
    assert calls == [(spellingbee.WORD_LIST_URL, spellingbee.WORD_LIST_FILENAME)]


def test_corrupted_vendored_copy_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corrupted = tmp_path / spellingbee.WORD_LIST_FILENAME
    corrupted.write_text("not-the-pinned-list\n", encoding="utf-8")
    monkeypatch.setattr(spellingbee, "get_base_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(spellingbee, "VENDORED_WORD_LIST_PATH", corrupted)

    with pytest.raises(DataLoadError, match="unexpected bytes"):
        spellingbee.ensure_word_list_available(allow_network=False)
