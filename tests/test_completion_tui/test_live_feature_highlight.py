"""Tests for live SAE feature token highlighting in the completion TUI."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as t

import scripts.completion_tui.live_feature_highlight as live_highlight_module
from scripts.completion_tui.highlight_types import (
    FeatureHighlightSpan,
    build_highlight_request,
)
from scripts.completion_tui.live_feature_highlight import (
    LiveFeatureHighlightService,
    ProbeSpec,
    _coalesce_token_highlights,
    _percentiles_to_levels,
    resolve_probe_spec,
)
from scripts.completion_tui.state import Segment
from steering.eval.probe_model import ProbeModel


class CharTokenizer:
    """One-character tokenizer with reversible integer IDs."""

    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


class FakeModel:
    def __init__(self, *, n_ctx: int | None = None) -> None:
        self.cfg = SimpleNamespace(n_ctx=n_ctx)
        self._param = t.nn.Parameter(t.zeros(1))

    def parameters(self):
        return iter([self._param])

    def to(self, device: t.device | str) -> FakeModel:
        self._param = t.nn.Parameter(self._param.to(device))
        return self

    def eval(self) -> None:
        return None


class FakeSAEConfig:
    apply_b_dec_to_input = True
    activation_fn = "relu"


class FakeSAE:
    def __init__(self, *, d_model: int = 4, n_features: int = 8) -> None:
        self.W_enc = t.zeros(d_model, n_features)
        self.W_enc[2, 3] = 1.0
        self.b_enc = t.zeros(n_features)
        self.b_dec = t.zeros(d_model)
        self.cfg = FakeSAEConfig()

    def parameters(self):
        return iter([self.W_enc, self.b_enc, self.b_dec])

    def to(self, device: t.device | str) -> FakeSAE:
        self.W_enc = self.W_enc.to(device)
        self.b_enc = self.b_enc.to(device)
        self.b_dec = self.b_dec.to(device)
        return self


def _fake_probe_model(*, n_ctx: int | None = None) -> ProbeModel:
    """Build the real probe dataclass around lightweight test doubles."""
    return ProbeModel(
        model=FakeModel(n_ctx=n_ctx),
        sae=FakeSAE(),
        tokenizer=CharTokenizer(),
        hook_name="blocks.8.hook_resid_post",
        layer=8,
    )


def _percentile_table() -> t.Tensor:
    table = t.zeros(8, 99, dtype=t.float64)
    table[3] = t.arange(1, 100, dtype=t.float64)
    return table


def test_live_probe_releases_transformerlens_fallback_on_load(monkeypatch) -> None:
    probe = _fake_probe_model()
    calls = []

    def fake_load_probe_model(**kwargs):
        calls.append(kwargs)
        return probe

    monkeypatch.setattr(live_highlight_module, "load_probe_model", fake_load_probe_model)
    service = LiveFeatureHighlightService(
        ProbeSpec(
            model_name="gemma-2-2b",
            release="gemmascope",
            activation_type="residual",
            layer=8,
            width="16k",
        ),
        _percentile_table(),
        device="cuda",
    )

    assert service._ensure_probe_model() is probe
    assert calls[0]["keep_transformerlens_model"] is False


def _segments() -> list[Segment]:
    return [
        Segment(segment_idx=0, char_start=0, char_end=10, color_idx=0, features=[]),
        Segment(segment_idx=1, char_start=10, char_end=20, color_idx=1, features=[]),
    ]


def test_percentiles_to_levels_colors_any_activation_top_ten_and_top_one() -> None:
    acts = t.tensor([0.0, 1.0, 1.0, 1.0, 1.0])
    pcts = t.tensor([99.5, 0.0, 89.99, 90.0, 99.0], dtype=t.float64)

    levels = _percentiles_to_levels(acts, pcts)

    assert levels.tolist() == [0, 1, 1, 2, 3]


def test_coalesce_token_highlights_merges_adjacent_equal_levels() -> None:
    spans = _coalesce_token_highlights(
        token_starts=t.tensor([5, 6, 7, 8]),
        token_ends=t.tensor([6, 7, 8, 9]),
        levels=t.tensor([1, 1, 2, 0]),
        percentiles=t.tensor([91.0, 92.0, 96.0, 50.0], dtype=t.float64),
        activations=t.tensor([91.0, 92.0, 96.0, 50.0]),
    )

    assert spans == (
        FeatureHighlightSpan(
            char_start=5, char_end=7, level=1, percentile=92.0, activation=92.0
        ),
        FeatureHighlightSpan(
            char_start=7, char_end=8, level=2, percentile=96.0, activation=96.0
        ),
    )


def test_coalesce_token_highlights_drops_zero_width_spans() -> None:
    spans = _coalesce_token_highlights(
        token_starts=t.tensor([5, 5, 6]),
        token_ends=t.tensor([5, 6, 7]),
        levels=t.tensor([3, 1, 3]),
        percentiles=t.tensor([99.5, 91.0, 99.5], dtype=t.float64),
        activations=t.tensor([10.0, 1.0, 10.0]),
    )

    assert spans == (
        FeatureHighlightSpan(
            char_start=5, char_end=6, level=1, percentile=91.0, activation=1.0
        ),
        FeatureHighlightSpan(
            char_start=6, char_end=7, level=3, percentile=99.5, activation=10.0
        ),
    )


def test_compute_highlight_uses_target_segment_and_corpus_percentiles() -> None:
    probe = _fake_probe_model()
    values_by_char = {
        10: 0.0,
        11: 89.9,
        12: 90.0,
        13: 94.9,
        14: 95.0,
        15: 98.9,
        16: 99.0,
        17: 80.0,
        18: 0.0,
        19: 96.0,
    }

    def activation_provider(_probe_model, tokens: t.Tensor) -> t.Tensor:
        acts = t.zeros(tokens.shape[0], tokens.shape[1], 4)
        for pos, token_id in enumerate(tokens[0].tolist()):
            acts[0, pos, 2] = values_by_char.get(token_id - ord("0"), 0.0)
        return acts

    # Use digits whose ordinal offset maps to 0..19 above.
    text = "".join(chr(ord("0") + i) for i in range(20))
    request = build_highlight_request(text, _segments(), 1, feature_id=3)
    service = LiveFeatureHighlightService(
        ProbeSpec(
            model_name="gemma-2-2b",
            release="gemmascope",
            activation_type="residual",
            layer=8,
            width="16k",
        ),
        _percentile_table(),
        probe_model_factory=lambda: probe,
        activation_provider=activation_provider,
    )

    result = service.compute(request)

    assert result.status == "ready"
    assert result.segment_idx == 1
    assert result.n_tokens == 10
    assert result.max_percentile == pytest.approx(99.5)
    assert [s.level for s in result.spans] == [1, 2, 3, 1, 2]
    assert [(s.char_start, s.char_end) for s in result.spans] == [
        (11, 12),
        (12, 16),
        (16, 17),
        (17, 18),
        (19, 20),
    ]


def test_compute_highlight_maxes_composite_members_and_reports_each_member() -> None:
    probe = _fake_probe_model()
    probe.sae.W_enc[1, 4] = 1.0
    table = _percentile_table()
    table[4] = t.arange(1, 100, dtype=t.float64)

    def activation_provider(_probe_model, tokens: t.Tensor) -> t.Tensor:
        acts = t.zeros(tokens.shape[0], tokens.shape[1], 4)
        for pos, token_id in enumerate(tokens[0].tolist()):
            value = float(token_id - ord("0"))
            acts[0, pos, 2] = value
            acts[0, pos, 1] = 20.0 - value
        return acts

    request = build_highlight_request(
        "0123456789" * 2,
        _segments(),
        1,
        feature_id=(3, 4),
    )
    service = LiveFeatureHighlightService(
        ProbeSpec(
            model_name="gemma-2-2b",
            release="gemmascope",
            activation_type="residual",
            layer=8,
            width="16k",
        ),
        table,
        probe_model_factory=lambda: probe,
        activation_provider=activation_provider,
    )
    result = service.compute(request)

    assert result.status == "ready"
    assert dict(result.per_feature_max_activation) == {3: 9.0, 4: 20.0}
    assert result.max_activation == 20.0


def test_compute_highlight_uses_suffix_window_when_context_is_short() -> None:
    probe = _fake_probe_model(n_ctx=5)
    seen_tokens: list[int] = []

    def activation_provider(_probe_model, tokens: t.Tensor) -> t.Tensor:
        seen_tokens.extend(tokens[0].tolist())
        acts = t.zeros(tokens.shape[0], tokens.shape[1], 4)
        acts[:, :, 2] = 99.0
        return acts

    text = "".join(chr(ord("0") + i) for i in range(20))
    request = build_highlight_request(text, _segments(), 1, feature_id=3)
    service = LiveFeatureHighlightService(
        ProbeSpec(
            model_name="gemma-2-2b",
            release="gemmascope",
            activation_type="residual",
            layer=8,
            width="16k",
        ),
        _percentile_table(),
        probe_model_factory=lambda: probe,
        activation_provider=activation_provider,
    )

    result = service.compute(request)

    assert seen_tokens == [ord("0") + i for i in range(15, 20)]
    assert result.status == "ready"
    assert result.n_tokens == 5
    assert result.spans == (
        FeatureHighlightSpan(
            char_start=15, char_end=20, level=3, percentile=99.5, activation=99.0
        ),
    )


def test_compute_highlight_returns_empty_when_target_has_no_visible_tokens() -> None:
    probe = _fake_probe_model()
    text = "abcdefghij"
    segments = [
        Segment(segment_idx=0, char_start=0, char_end=10, color_idx=0, features=[]),
        Segment(segment_idx=1, char_start=10, char_end=10, color_idx=1, features=[]),
    ]
    request = build_highlight_request(text, segments, 1, feature_id=3)
    service = LiveFeatureHighlightService(
        ProbeSpec(
            model_name="gemma-2-2b",
            release="gemmascope",
            activation_type="residual",
            layer=8,
            width="16k",
        ),
        _percentile_table(),
        probe_model_factory=lambda: probe,
        activation_provider=lambda _pm, tokens: t.zeros(tokens.shape[0], tokens.shape[1], 4),
    )

    result = service.compute(request)

    assert result.status == "empty"
    assert result.spans == ()
    assert "no visible tokens" in (result.message or "")


def test_compute_highlight_rejects_feature_missing_percentile_statistics() -> None:
    probe = _fake_probe_model()
    text = "".join(chr(ord("0") + i) for i in range(20))
    request = build_highlight_request(text, _segments(), 1, feature_id=4)
    service = LiveFeatureHighlightService(
        ProbeSpec(
            model_name="gemma-2-2b",
            release="gemmascope",
            activation_type="residual",
            layer=8,
            width="16k",
        ),
        _percentile_table(),
        probe_model_factory=lambda: probe,
        activation_provider=lambda _pm, tokens: t.ones(tokens.shape[0], tokens.shape[1], 4),
    )

    result = service.compute(request)

    assert result.status == "unavailable"
    assert "missing corpus percentile" in (result.message or "")


def test_resolve_probe_spec_prefers_checkpoint_sae_config_and_checks_feature_dir() -> None:
    feature_dir = "/tmp/features/s64/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base"
    user_cfg = {
        "sae": {
            "model": "gemma-2-2b",
            "release": "gemmascope",
            "activation_type": "residual",
            "layer": 8,
            "width": "16k",
            "l0": 71,
        }
    }

    spec = resolve_probe_spec(user_cfg, feature_dir)

    assert spec.model_name == "gemma-2-2b"
    assert spec.release == "gemmascope"
    assert spec.activation_type == "residual"
    assert spec.layer == 8
    assert spec.width == "16k"
    assert spec.l0 == 71
    assert spec.sae_id is not None
    assert "average_l0_71" in spec.sae_id


def test_resolve_probe_spec_falls_back_to_feature_dir_when_checkpoint_lacks_sae() -> None:
    feature_dir = "/tmp/features/s64/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base"

    spec = resolve_probe_spec({}, feature_dir)

    assert spec.model_name == "gemma-2-2b"
    assert spec.release == "gemmascope"
    assert spec.activation_type == "residual"
    assert spec.layer == 8
    assert spec.width == "16k"
    assert spec.l0 is None
    assert spec.hook_name == "blocks.8.hook_resid_post"
    assert spec.sae_id == "layer_8/width_16k/canonical"


def test_resolve_probe_spec_rejects_checkpoint_feature_dir_mismatch() -> None:
    feature_dir = "/tmp/features/s64/l9/a_res/p_max_pool/f16k/m_gemma-2-2b/base"
    user_cfg = {
        "sae": {
            "model": "gemma-2-2b",
            "release": "gemmascope",
            "activation_type": "residual",
            "layer": 8,
            "width": "16k",
        }
    }

    with pytest.raises(ValueError, match="does not match feature_dir"):
        resolve_probe_spec(user_cfg, feature_dir)


@pytest.mark.parametrize("seed_leaf", ["seed42", "seed42_opposite_percentile_pool64"])
def test_resolve_probe_spec_accepts_splice_overlay_and_l0_tier(seed_leaf: str) -> None:
    feature_dir = (
        "/tmp/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/"
        f"base/context_independent/no_bos/splice/cut_frac0.15/{seed_leaf}"
    )
    spec = resolve_probe_spec(
        {
            "sae": {
                "model": "gemma-3-4b-pt",
                "release": "gemmascope_2",
                "activation_type": "residual",
                "layer": 22,
                "width": "65k",
                "l0": "medium",
            }
        },
        feature_dir,
    )
    assert spec.sae_id == "layer_22_width_65k_l0_medium"
