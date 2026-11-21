"""Unit tests for the offline JSON description service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.chat_tui.descriptions import OfflineDescriptionService


def _write(tmp_path: Path, payload) -> Path:
    path = tmp_path / "regenerated_descriptions.json"
    path.write_text(json.dumps(payload))
    return path


def test_lookup_resolves_synchronously_with_on_start(tmp_path: Path) -> None:
    service = OfflineDescriptionService(_write(tmp_path, {"3": "threes", 24: "two-fours"}))
    calls: list[tuple[str, int, str | None]] = []

    service.request_description(
        3,
        lambda feature_id, text: calls.append(("done", feature_id, text)),
        on_start=lambda feature_id: calls.append(("start", feature_id, None)),
    )
    service.request_description(
        24, lambda feature_id, text: calls.append(("done", feature_id, text))
    )

    assert calls == [("start", 3, None), ("done", 3, "threes"), ("done", 24, "two-fours")]
    assert len(service) == 2


def test_missing_feature_resolves_to_empty_text(tmp_path: Path) -> None:
    service = OfflineDescriptionService(_write(tmp_path, {"3": "threes"}))
    calls: list[tuple[int, str]] = []

    service.request_description(999, lambda feature_id, text: calls.append((feature_id, text)))

    assert calls == [(999, "")]


def test_non_object_payload_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON object mapping feature ids"):
        OfflineDescriptionService(_write(tmp_path, ["not", "a", "mapping"]))


def test_non_integer_key_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a feature id"):
        OfflineDescriptionService(_write(tmp_path, {"feature-seven": "text"}))


def test_non_string_value_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a string"):
        OfflineDescriptionService(_write(tmp_path, {"7": {"description": "nested"}}))


def test_missing_file_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        OfflineDescriptionService(tmp_path / "absent.json")


def test_shutdown_is_a_noop_for_protocol_parity(tmp_path: Path) -> None:
    service = OfflineDescriptionService(_write(tmp_path, {}))
    service.shutdown()
