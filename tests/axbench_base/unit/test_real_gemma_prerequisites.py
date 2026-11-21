"""Unit tests for the prerequisite helpers in the Phase 9 very-slow
real-Gemma integration test.

The very-slow test in
``tests/axbench_base/integration/test_full_pipeline_real_gemma.py``
calls three fail-loud helpers (``_require_nanochat_cache``,
``_require_openrouter_key``, ``_require_feature_dir``) before touching
any real resource. Those helpers need to be exercised by regular unit
tests so the fail-loud logic itself doesn't rot — this module pins
their contract by monkey-patching the module-level path constants onto
``tmp_path`` mock caches and asserting the right branch raises.

These tests are NOT marked ``very_slow``; they run as part of the
default suite so every CI run validates the gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.axbench_base.integration import test_full_pipeline_real_gemma as _mod


def _patch_cache_constants(
    monkeypatch: pytest.MonkeyPatch, cache_root: Path,
) -> None:
    """Redirect the module-level cache constants at ``cache_root``.

    The three constants ``_NANOCHAT_CACHE``, ``_D12_BASE_REL``, and
    ``_D12_ENCDEC_REL`` are used by ``_require_nanochat_cache`` as
    absolute / relative path fragments. Pointing them at
    ``tmp_path`` lets the unit tests construct exactly the subset of
    layout they want to exercise.
    """
    monkeypatch.setattr(_mod, "_NANOCHAT_CACHE", cache_root)


def _make_valid_nanochat_cache(cache_root: Path) -> None:
    """Materialize a minimal but valid nanochat cache structure.

    Layout:

    - ``<root>/base_checkpoints/d12_base/model_054600.pt``
    - ``<root>/conditional_encdec_checkpoints/d12_baseline/s128/model_054600.pt``

    The ``.pt`` files are empty placeholders; only their existence is
    checked by the prerequisite helper, not their contents.
    """
    decoder_dir = cache_root / _mod._D12_BASE_REL
    decoder_dir.mkdir(parents=True)
    (decoder_dir / "model_054600.pt").touch()

    encdec_dir = cache_root / _mod._D12_ENCDEC_REL / "s128"
    encdec_dir.mkdir(parents=True)
    (encdec_dir / "model_054600.pt").touch()


def test_require_nanochat_cache_raises_when_dir_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent ``_NANOCHAT_CACHE`` root → ``pytest.fail`` with guidance
    to populate the directory."""
    missing_root = tmp_path / "does-not-exist"
    _patch_cache_constants(monkeypatch, missing_root)
    with pytest.raises(pytest.fail.Exception) as excinfo:
        _mod._require_nanochat_cache()
    message = str(excinfo.value)
    assert str(missing_root) in message
    assert "Nanochat cache directory" in message


def test_require_nanochat_cache_raises_when_base_d12_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Root exists but ``base_checkpoints/d12_base`` missing → fail
    with a message naming the missing directory and the pretrain
    pipeline as the remediation path."""
    cache_root = tmp_path / "nanochat"
    cache_root.mkdir()
    _patch_cache_constants(monkeypatch, cache_root)

    with pytest.raises(pytest.fail.Exception) as excinfo:
        _mod._require_nanochat_cache()
    message = str(excinfo.value)
    assert "d12_base" in message
    assert "decoder-only" in message


def test_require_nanochat_cache_raises_when_encdec_d12_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Base d12 dir present and populated but encdec d12 missing → fail
    with a message naming the missing encdec directory and the encdec
    SFT pipeline as the remediation path."""
    cache_root = tmp_path / "nanochat"
    decoder_dir = cache_root / _mod._D12_BASE_REL
    decoder_dir.mkdir(parents=True)
    (decoder_dir / "model_054600.pt").touch()
    _patch_cache_constants(monkeypatch, cache_root)

    with pytest.raises(pytest.fail.Exception) as excinfo:
        _mod._require_nanochat_cache()
    message = str(excinfo.value)
    assert "encdec" in message.lower()
    assert "conditional_encdec_checkpoints" in message


def test_require_nanochat_cache_returns_when_all_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fully populated cache → helper returns ``None`` without raising."""
    cache_root = tmp_path / "nanochat"
    cache_root.mkdir()
    _patch_cache_constants(monkeypatch, cache_root)
    _make_valid_nanochat_cache(cache_root)

    # No raise, no return value.
    assert _mod._require_nanochat_cache() is None


def test_real_gemma_does_not_require_a_provider_credential() -> None:
    """The integration injects its validation-only dummy key itself."""

    assert not hasattr(_mod, "_require_openrouter_key")
