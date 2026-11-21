"""Tests for diagnostic logging added in PR-1.

These tests assert that the new `logger.info` bracketing inside
``_load_feature_extraction_pair`` emits the expected sub-step lines so
operators can see where the silent ~14m window goes. They monkey-patch
the heavy loaders so the test runs on CPU without GPU or network.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from steering.eval.axbench_base import runtime_state as rs_mod
from steering.eval.axbench_base.runtime_state import _load_feature_extraction_pair
from tests.axbench_base._loguru_capture import capture_loguru_info


@pytest.fixture
def stub_loaders(monkeypatch):
    """Replace heavy loaders so the test runs on CPU."""
    fake_tl_model = MagicMock(name="tl_model")
    fake_sae = SimpleNamespace(to=MagicMock())
    fake_nn_model = MagicMock(name="nn_model")
    monkeypatch.setattr(rs_mod, "load_sae_and_model",
                        lambda **kwargs: (fake_tl_model, fake_sae, MagicMock()))
    monkeypatch.setattr(rs_mod, "build_nnterp_model",
                        lambda *a, **kw: fake_nn_model)
    return fake_tl_model, fake_sae, fake_nn_model


def test_load_feature_extraction_pair_emits_substep_logs(stub_loaders):
    """Each sub-step (gemma load, sae load, nnterp wrap, bf16 cast) emits
    a begin/end log line at INFO level."""
    cfg = MagicMock()
    cfg.neuronpedia_source = "8-gemmascope-res-16k"
    cfg.neuronpedia_model_id = "google/gemma-2-2b"
    cfg.feature_extraction_model_id = "gemma-2-2b"
    cfg.device = "cpu"

    with capture_loguru_info() as log_messages:
        _load_feature_extraction_pair(cfg)

    log_text = "\n".join(log_messages)
    assert "[runtime_state] resolving Neuronpedia source" in log_text
    assert "[runtime_state] resolving SAE release/sae_id" in log_text
    assert "[runtime_state] loading TransformerLens base model + SAE" in log_text
    assert "[runtime_state] building nnterp wrapper" in log_text
    assert "[runtime_state] casting SAE to bf16" in log_text
    assert "[runtime_state] feature-extraction pair ready" in log_text


def test_load_feature_extraction_pair_substep_logs_include_timings(stub_loaders):
    """Each end-of-substep log line includes an elapsed_ms value."""
    cfg = MagicMock()
    cfg.neuronpedia_source = "8-gemmascope-res-16k"
    cfg.neuronpedia_model_id = "google/gemma-2-2b"
    cfg.feature_extraction_model_id = "gemma-2-2b"
    cfg.device = "cpu"

    with capture_loguru_info() as log_messages:
        _load_feature_extraction_pair(cfg)

    end_lines = [m for m in log_messages if "elapsed_ms" in m]
    assert len(end_lines) >= 4, (
        f"Expected at least 4 elapsed_ms log lines (one per sub-step end); "
        f"got {len(end_lines)}: {end_lines}"
    )


def test_load_feature_extraction_pair_substep_log_values_are_meaningful(stub_loaders):
    """Beyond ``elapsed_ms`` appearing in 4+ lines, the values must
    parse as non-negative integer milliseconds. Catches a regression
    that emits ``elapsed_ms=None`` / ``elapsed_ms=`` / non-integer."""
    cfg = MagicMock()
    cfg.neuronpedia_source = "8-gemmascope-res-16k"
    cfg.neuronpedia_model_id = "google/gemma-2-2b"
    cfg.feature_extraction_model_id = "gemma-2-2b"
    cfg.device = "cpu"

    with capture_loguru_info() as log_messages:
        _load_feature_extraction_pair(cfg)

    # Pull all elapsed_ms / total_elapsed_ms numerics out of the lines.
    import re
    pattern = re.compile(r"\b(?:total_)?elapsed_ms=([^\s\]]+)")
    parsed: list[int] = []
    for line in log_messages:
        for raw in pattern.findall(line):
            try:
                parsed.append(int(raw))
            except ValueError:
                pytest.fail(
                    f"non-integer elapsed_ms substitution in line: {line!r}"
                )
    assert len(parsed) >= 5, (
        f"expected ≥5 elapsed_ms values (4 sub-steps + total); "
        f"got {len(parsed)}: {parsed!r}"
    )
    for value in parsed:
        assert value >= 0, f"negative elapsed_ms: {value}"


def test_load_feature_extraction_pair_silenced_on_non_zero_rank(
    monkeypatch, stub_loaders,
):
    """Workers with RANK>0 must not emit any progress logs from
    ``_load_feature_extraction_pair`` regardless of how the loguru sink
    was configured."""
    monkeypatch.setenv("RANK", "1")
    cfg = MagicMock()
    cfg.neuronpedia_source = "8-gemmascope-res-16k"
    cfg.neuronpedia_model_id = "google/gemma-2-2b"
    cfg.feature_extraction_model_id = "gemma-2-2b"
    cfg.device = "cpu"
    with capture_loguru_info() as log_messages:
        _load_feature_extraction_pair(cfg)
    assert log_messages == [], (
        f"non-zero rank emitted progress lines: {log_messages!r}"
    )


def test_load_feature_extraction_pair_exception_path_logs_begin_not_end(
    monkeypatch,
):
    """When an exception happens inside the function (here:
    ``parse_neuronpedia_source`` raises), the begin log emits but the
    final ``feature-extraction pair ready`` log does NOT, and the
    raised :class:`AxBenchBaseError` chains the original exception as
    ``__cause__``."""
    from steering.eval.axbench_base.types import AxBenchBaseError

    def boom(_src):
        raise ValueError("malformed neuronpedia source for test")

    monkeypatch.setattr(rs_mod, "parse_neuronpedia_source", boom)

    cfg = MagicMock()
    cfg.neuronpedia_source = "intentionally-bad"
    cfg.neuronpedia_model_id = "google/gemma-2-2b"
    cfg.device = "cpu"

    with capture_loguru_info() as log_messages:
        with pytest.raises(AxBenchBaseError) as exc_info:
            _load_feature_extraction_pair(cfg)

    log_text = "\n".join(log_messages)
    assert "[runtime_state] resolving Neuronpedia source" in log_text, (
        "begin log line must still emit before the failing call"
    )
    assert "[runtime_state] feature-extraction pair ready" not in log_text, (
        "completion log must NOT emit when the load fails"
    )

    cause = exc_info.value.__cause__
    assert isinstance(cause, ValueError), (
        f"AxBenchBaseError must chain the original ValueError; "
        f"got __cause__={cause!r}"
    )
    assert "malformed neuronpedia source for test" in str(cause)


def test_load_feature_extraction_pair_logs_when_nnterp_unavailable(monkeypatch):
    """When build_nnterp_model returns None, the 'nnterp wrapper unavailable'
    log emits and 'casting SAE to bf16' does NOT — covers the else-branch
    of the nn_model is None guard."""
    fake_tl_model = MagicMock(name="tl_model")
    fake_sae = SimpleNamespace(to=MagicMock())
    monkeypatch.setattr(rs_mod, "load_sae_and_model",
                        lambda **kwargs: (fake_tl_model, fake_sae, MagicMock()))
    # Variation: build_nnterp_model returns None (nnterp wrapper unsupported).
    monkeypatch.setattr(rs_mod, "build_nnterp_model",
                        lambda *a, **kw: None)

    cfg = MagicMock()
    cfg.neuronpedia_source = "8-gemmascope-res-16k"
    cfg.neuronpedia_model_id = "google/gemma-2-2b"
    cfg.feature_extraction_model_id = "gemma-2-2b"
    cfg.device = "cpu"

    with capture_loguru_info() as log_messages:
        _load_feature_extraction_pair(cfg)

    log_text = "\n".join(log_messages)
    assert "[runtime_state] nnterp wrapper unavailable" in log_text
    assert "[runtime_state] casting SAE to bf16" not in log_text
