"""Unit tests for LinearProbe training (Task 22).

Covers
``steering.eval.axbench_base.baselines.linear_probe.train_linear_probe_all``
and its per-concept helper, which fits an L1-regularised logistic
regression on residual-stream activations and stores the normalised
coefficient vector as a steering direction.

TDD design choices pinned down by these tests:
  * Capture / label-split machinery is IDENTICAL to DiffMean (Task 21),
    shared via ``baselines._capture``. The linear probe fits on the
    stacked per-row mean continuation activations.
  * The probe uses ``LogisticRegression(penalty='l1', C=1.0,
    max_iter=500, solver='liblinear', random_state=...)``.
  * The saved ``direction`` is ``coef_[0] / ||coef_[0]||``. We also
    persist the raw ``coef`` and ``intercept`` tensors so downstream
    callers can reconstruct the probe if needed.
  * Class imbalance is tolerated (e.g., 3:1 negative:positive ratio);
    the probe must still converge and produce a finite direction.
  * NaN activations raise ``AxBenchBaseError``.
  * Empty positives / negatives raise ``AxBenchBaseError``.
  * The degenerate all-same-label case surfaces as one of the empty-
    side errors (message mentions ``"positives"`` or ``"negatives"``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as loguru_logger
from safetensors import safe_open
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression

from nanochat.gpt import GPT, GPTConfig
from steering.eval.axbench_base.artifact_metadata import (
    schema_with_extraction_tokenizer_metadata,
)
from steering.eval.axbench_base.baselines import _capture as capture_mod
from steering.eval.axbench_base.baselines import linear_probe as linear_probe_mod
from steering.eval.axbench_base.baselines.linear_probe import train_linear_probe_all
from steering.eval.axbench_base.config import LayerHooksConfig
from steering.eval.axbench_base.direction_scale import (
    MAX_ACTIVATION_METADATA_KEY,
)
from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
)
from steering.eval.axbench_base.tokenizer_utils import tokenize_labeled_pair
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SteeringMode,
)
from steering.immutable_io import ImmutableArtifactError
from tests.axbench_base import paths as test_paths
from tests.axbench_base.unit.test_config import make_valid_config

# ---------------------------------------------------------------------------
# Tiny GPT fixture + stub tokenizer (mirrors test_diffmean.py conventions)
# ---------------------------------------------------------------------------


TINY_VOCAB: int = 128
TINY_N_EMBD: int = 64
TINY_N_LAYER: int = 2
TINY_N_HEAD: int = 2
TINY_N_KV_HEAD: int = 2
TINY_SEQ_LEN: int = 128
TINY_BOS_ID: int = 1
TINY_LAYER: int = 0  # override so layer-0 is valid for our 2-layer fixture


def _make_tiny_gpt() -> GPT:
    """Tiny CPU GPT in eval mode, fully initialised, deterministic."""
    t.manual_seed(0xD1FF)
    config = GPTConfig(
        sequence_len=TINY_SEQ_LEN,
        vocab_size=TINY_VOCAB,
        n_layer=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        n_embd=TINY_N_EMBD,
        window_pattern="L",
    )
    with t.device("meta"):
        model = GPT(config, pad_vocab_size_to=TINY_VOCAB)
    model = model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    return model


class _StubNanochatTokenizer:
    """Deterministic nanochat-like tokenizer for tests."""

    def __init__(self, vocab_size: int = TINY_VOCAB, bos_id: int = TINY_BOS_ID) -> None:
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self._decodes: dict[tuple[int, ...], str] = {}
        self._token_text: dict[int, str] = {}

    def get_bos_token_id(self) -> int:
        return self.bos_id

    def encode(self, text: str, prepend: int | None = None) -> list[int]:
        base = [(ord(c) % (self.vocab_size - 2)) + 2 for c in text]  # keep away from 0,1
        self._token_text.update(zip(base, text, strict=True))
        for end in range(len(base) + 1):
            self._decodes[tuple(base[:end])] = text[:end]
        if prepend is not None:
            return [prepend] + base
        return base

    def decode(self, tokens: list[int]) -> str:
        exact = self._decodes.get(tuple(tokens))
        if exact is not None:
            return exact
        return "".join(self._token_text[token] for token in tokens)


def _install_stub_tokenizer(monkeypatch: pytest.MonkeyPatch) -> _StubNanochatTokenizer:
    tok = _StubNanochatTokenizer()
    monkeypatch.setattr(linear_probe_mod, "get_tokenizer", lambda: tok)
    return tok


# ---------------------------------------------------------------------------
# Corpus helpers (identical to test_diffmean.py — both baselines consume
# the same parquet schemas)
# ---------------------------------------------------------------------------


_POS_SCHEMA = pa.schema(
    [
        ("concept_id", pa.int32()),
        ("category", pa.string()),
        ("content_class", pa.string()),
        ("prefix_text", pa.string()),
        ("output_text", pa.string()),
        ("output_concept", pa.string()),
        ("n_prefix_nanochat_tokens", pa.int32()),
        ("gen_source", pa.string()),
        ("gen_seed", pa.int64()),
        ("qc_feature_id", pa.int32()),
        ("qc_seed_percentile", pa.float32()),
        ("qc_output_percentile", pa.float32()),
        ("qc_delta", pa.float32()),
        ("qc_attempts", pa.int32()),
        ("qc_accepted", pa.bool_()),
    ]
)


_NEG_SCHEMA = pa.schema(
    [
        *list(_POS_SCHEMA),
        pa.field("prefix_source_shard", pa.int32()),
        pa.field("prefix_source_doc", pa.int32()),
        pa.field("prefix_source_seg_idx", pa.int32()),
        pa.field("qc_zero_activation_verified", pa.bool_()),
    ]
)


def _pos_row(
    concept_id: int,
    content_class: str,
    prefix_text: str,
    output_text: str,
    n_prefix_nanochat_tokens: int,
) -> dict[str, Any]:
    return {
        "concept_id": concept_id,
        "category": "positive" if content_class == "concept_rich" else "negative",
        "content_class": content_class,
        "prefix_text": prefix_text,
        "output_text": output_text,
        "output_concept": "stub concept",
        "n_prefix_nanochat_tokens": n_prefix_nanochat_tokens,
        "gen_source": "test:stub",
        "gen_seed": 0,
        "qc_feature_id": 7,
        "qc_seed_percentile": 0.0,
        "qc_output_percentile": 0.0,
        "qc_delta": 0.0,
        "qc_attempts": 1,
        "qc_accepted": True,
    }


def _neg_row(
    concept_id: int,
    content_class: str,
    prefix_text: str,
    output_text: str,
    n_prefix_nanochat_tokens: int,
) -> dict[str, Any]:
    base = _pos_row(
        concept_id, content_class, prefix_text, output_text, n_prefix_nanochat_tokens,
    )
    base.update(
        {
            "prefix_source_shard": 0,
            "prefix_source_doc": 0,
            "prefix_source_seg_idx": 0,
            "qc_zero_activation_verified": content_class == "concept_avoiding",
        }
    )
    return base


def _write_positive_corpus(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(
        rows,
        schema=schema_with_stage1_continuation_contract(
            schema_with_positive_qc_contract(
                schema_with_extraction_tokenizer_metadata(
                    _POS_SCHEMA,
                    "google/gemma-2-2b",
                ),
                accept_threshold=0.3,
            ),
            role="stage1b_positive",
        ),
    )
    pq.write_table(table, path)


def _write_negative_corpus(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(
        rows,
        schema=schema_with_stage1_continuation_contract(
            schema_with_extraction_tokenizer_metadata(
                _NEG_SCHEMA,
                "google/gemma-2-2b",
            ),
            role="stage1c_negative",
        ),
    )
    pq.write_table(table, path)


@contextmanager
def _capture_loguru_warnings() -> Iterator[list[str]]:
    """Capture WARNING+ level messages from loguru into a plain list.

    Usage::

        with _capture_loguru_warnings() as warnings:
            ...
        assert any("expected phrase" in w for w in warnings)
    """
    collected: list[str] = []
    handler_id = loguru_logger.add(
        lambda msg: collected.append(msg.record["message"]),
        level="WARNING",
    )
    try:
        yield collected
    finally:
        loguru_logger.remove(handler_id)


def _independent_row_continuation_mean(
    model: GPT, layer: int, tok: Any, row: dict[str, Any],
) -> t.Tensor:
    """Recompute one row's continuation mean WITHOUT ``_capture.py``.

    Independent byte-identity oracle for the pre-#1228 formula: own forward
    hook, on-device mean, then ``.cpu()`` — mirroring the load-bearing
    reduction order. NOTE: this suite runs on CPU, where the on-device mean
    and a moved-then-reduced mean coincide bitwise; a CUDA reduction-order
    regression is not observable here and stays guarded by the load-bearing
    comment in ``_capture.py``.
    """
    full_tokens, slice_start = tokenize_labeled_pair(
        tok, row["prefix_text"], row["output_text"]
    )
    device = next(model.parameters()).device
    idx = t.tensor([full_tokens], dtype=t.long, device=device)
    captured: list[t.Tensor] = []

    def _hook(
        module: nn.Module, inputs: tuple[t.Tensor, ...], output: t.Tensor
    ) -> None:
        del module, inputs
        captured.append(output.detach().to(t.float32).clone())

    blocks = cast(nn.ModuleList, model.transformer["h"])
    handle = blocks[layer].register_forward_hook(_hook)
    was_training = model.training
    model.eval()
    try:
        with t.no_grad():
            model(idx)
    finally:
        handle.remove()
        if was_training:
            model.train()
    continuation = captured[-1][0, slice_start:, :]
    return continuation.mean(dim=0).cpu()


def _tiny_cfg(tmp_path: Path) -> Any:
    cfg = make_valid_config(
        run_tag="linear_probe_test",
        output_root=None,
        device="cpu",
    )
    return cfg.replace(
        layer_hooks=LayerHooksConfig(diffmean_and_probe_layer=TINY_LAYER),
    )


def _concept(concept_id: int = 42) -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=7,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description="the concept of hope",
        corpus_count=100,
        description_source="neuronpedia",
    )


def _n(text: str, tok: _StubNanochatTokenizer) -> int:
    return len(tok.encode(text, prepend=tok.get_bos_token_id()))


# ---------------------------------------------------------------------------
# Stub-capture helper: make the probe-training code read synthetic
# per-row vectors instead of running the tiny GPT, so we can construct
# linearly-separable corpora without thinking about token embeddings.
# ---------------------------------------------------------------------------


def _install_stub_capture(
    monkeypatch: pytest.MonkeyPatch,
    pos_vec_fn: Any,
    neg_vec_fn: Any,
) -> None:
    """Replace ``_capture_per_row_continuation`` with a deterministic
    function keyed on prefix_text content: if ``"POS"`` appears in
    ``prefix_text``, use ``pos_vec_fn(prefix_text)``; otherwise use
    ``neg_vec_fn(prefix_text)``. Each fn takes the prefix string and
    returns a ``(TINY_N_EMBD,)`` float tensor.
    """

    def _fake_capture(
        *,
        model,
        layer,
        tokenizer,
        prefix_text,
        output_text,
        n_prefix_nanochat_tokens,
        encoder_conditioning=None,
    ):
        del model, layer, tokenizer, output_text, n_prefix_nanochat_tokens
        if "POS" in prefix_text:
            mean = pos_vec_fn(prefix_text)
        else:
            mean = neg_vec_fn(prefix_text)
        return mean, mean.unsqueeze(0)

    monkeypatch.setattr(
        capture_mod,
        "_capture_per_row_continuation",
        _fake_capture,
    )


# ---------------------------------------------------------------------------
# Tests: fit math + probe behaviour
# ---------------------------------------------------------------------------


class TestProbeConvergence:
    def test_probe_converges_on_separable_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Synthetic linearly-separable corpus: positives are (+3, 0, ..., 0)
        ± noise; negatives are (-3, 0, ..., 0) ± noise. Fit must converge
        and the fitted direction must separate train rows cleanly via the
        saved ``coef`` / ``intercept``.
        """
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        rng = np.random.default_rng(0xC0DE)

        def _pos(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = 3.0 + float(rng.normal(0.0, 0.01))
            return v

        def _neg(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = -3.0 + float(rng.normal(0.0, 0.01))
            return v

        _install_stub_capture(monkeypatch, _pos, _neg)

        rows = []
        for i in range(6):
            rows.append(_pos_row(42, "concept_rich", f"POS_{i}", "out", 3))
        for i in range(6):
            rows.append(_pos_row(42, "concept_avoiding", f"neg_{i}", "out", 3))

        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_linear_probe_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        assert len(out_paths) == 1
        payload = load_file(str(out_paths[0]))
        direction = payload["direction"]
        coef = payload["coef"]
        intercept = payload["intercept"]

        # Reconstruct the probe's decision boundary: coef · x + intercept.
        # For separable data, the probe should score all positives > 0 and
        # all negatives < 0 on their training vectors.
        pos_vecs = t.stack([_pos(f"POS_{i}") for i in range(6)])
        neg_vecs = t.stack([_neg(f"neg_{i}") for i in range(6)])
        pos_scores = pos_vecs @ coef[0] + intercept[0]
        neg_scores = neg_vecs @ coef[0] + intercept[0]
        assert (pos_scores > 0).all(), f"Probe should score positives > 0: {pos_scores}"
        assert (neg_scores < 0).all(), f"Probe should score negatives < 0: {neg_scores}"

        # Direction is the unit-norm coef (coef dominates on dim 0).
        assert direction.shape == (TINY_N_EMBD,)
        assert t.allclose(direction.norm(), t.tensor(1.0), atol=1e-5)
        # The direction of the probe points from neg to pos — the sign
        # of its largest component matches the positive class direction.
        assert direction[0] > 0.9, f"Expected direction[0] ~= 1, got {direction[0]}"


class TestDirectionMath:
    def test_direction_is_normalized_coef(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``direction == coef[0] / ||coef[0]||`` exactly, unit norm."""
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        def _pos(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = 2.0
            return v

        def _neg(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = -2.0
            return v

        _install_stub_capture(monkeypatch, _pos, _neg)

        rows = []
        for i in range(4):
            rows.append(_pos_row(42, "concept_rich", f"POS_{i}", "out", 3))
        for i in range(4):
            rows.append(_pos_row(42, "concept_avoiding", f"neg_{i}", "out", 3))

        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_linear_probe_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        payload = load_file(str(out_paths[0]))
        direction = payload["direction"]
        coef = payload["coef"]

        expected = coef[0] / coef[0].norm()
        assert t.allclose(direction, expected, atol=1e-6)
        assert t.allclose(direction.norm(), t.tensor(1.0), atol=1e-6)


class TestL1Penalty:
    def test_l1_penalty_produces_sparse_coefficients(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Construct a corpus where only dimension 0 carries signal.
        L1 regularisation should drive most other dimensions to zero.
        """
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        rng = np.random.default_rng(0xDEAD)

        def _pos(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = 4.0
            # Tiny noise on other dims — L1 should zero these.
            for j in range(1, TINY_N_EMBD):
                v[j] = float(rng.normal(0.0, 0.01))
            return v

        def _neg(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = -4.0
            for j in range(1, TINY_N_EMBD):
                v[j] = float(rng.normal(0.0, 0.01))
            return v

        _install_stub_capture(monkeypatch, _pos, _neg)

        rows = []
        for i in range(16):
            rows.append(_pos_row(42, "concept_rich", f"POS_{i}", "out", 3))
        for i in range(16):
            rows.append(_pos_row(42, "concept_avoiding", f"neg_{i}", "out", 3))

        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_linear_probe_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        payload = load_file(str(out_paths[0]))
        coef = payload["coef"]  # (1, C)

        # Dimension 0 should have a large nonzero coefficient.
        assert abs(coef[0, 0].item()) > 0.5, (
            f"Expected large coef on signal dim, got {coef[0, 0].item()}"
        )
        # Most other dimensions should be driven to ~zero by L1.
        tail = coef[0, 1:]
        nonzero_count = int((tail.abs() > 1e-6).sum().item())
        total = tail.numel()
        assert nonzero_count < total // 2, (
            f"Expected L1 to zero out most tail dims, got {nonzero_count}/{total} nonzero"
        )

    def test_l1_penalty_and_solver_are_correct(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Spy on ``LogisticRegression`` to assert the config we pass in."""
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        def _pos(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = 1.0
            return v

        def _neg(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = -1.0
            return v

        _install_stub_capture(monkeypatch, _pos, _neg)

        captured_kwargs: list[dict[str, Any]] = []
        real_ctor = LogisticRegression

        def _spy_ctor(*args, **kwargs):
            captured_kwargs.append(dict(kwargs))
            return real_ctor(*args, **kwargs)

        monkeypatch.setattr(linear_probe_mod, "LogisticRegression", _spy_ctor)

        rows = [
            _pos_row(42, "concept_rich", "POS_1", "out", 3),
            _pos_row(42, "concept_rich", "POS_2", "out", 3),
            _pos_row(42, "concept_avoiding", "neg_1", "out", 3),
            _pos_row(42, "concept_avoiding", "neg_2", "out", 3),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        train_linear_probe_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        assert len(captured_kwargs) == 1
        kw = captured_kwargs[0]
        # sklearn 1.8+ uses ``l1_ratio`` instead of ``penalty=`` to pick
        # the regularization type. ``l1_ratio=1.0`` is pure L1.
        assert kw["l1_ratio"] == 1.0
        assert kw["C"] == 1.0
        assert kw["solver"] == "liblinear"
        assert kw["max_iter"] == 500


# ---------------------------------------------------------------------------
# Tests: class imbalance + robustness
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_handles_class_imbalance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """3:1 negative:positive ratio. The probe must converge without
        crashing and produce a finite direction.
        """
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        def _pos(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = 2.0
            return v

        def _neg(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = -2.0
            return v

        _install_stub_capture(monkeypatch, _pos, _neg)

        rows = []
        for i in range(6):
            rows.append(_pos_row(42, "concept_rich", f"POS_{i}", "out", 3))
        for i in range(18):  # 3:1 imbalance
            rows.append(_pos_row(42, "concept_avoiding", f"neg_{i}", "out", 3))

        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_linear_probe_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        direction = load_file(str(out_paths[0]))["direction"]
        assert t.isfinite(direction).all()
        # Norm should be 1 or 0 (0 if the probe degenerates, but it won't
        # for separable-enough data — this just guards against NaN).
        assert direction.norm().item() > 0.99

        # Metadata reflects the imbalance.
        with safe_open(str(out_paths[0]), framework="pt") as f:
            meta = f.metadata() or {}
        assert meta["n_pos"] == "6"
        assert meta["n_neg"] == "18"

    def test_handles_unicode_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unicode in prefix/output text must not break the pipeline."""
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()
        tok = _StubNanochatTokenizer()

        pos_out = "résumé"
        neg_out = "naïveté"

        rows = [
            _pos_row(42, "concept_rich", "café", pos_out, _n("café", tok)),
            _pos_row(42, "concept_rich", "café2", pos_out, _n("café2", tok)),
            _pos_row(42, "concept_rich", "café3", pos_out, _n("café3", tok)),
            _pos_row(42, "concept_avoiding", "élan", neg_out, _n("élan", tok)),
            _pos_row(42, "concept_avoiding", "étude", neg_out, _n("étude", tok)),
            _pos_row(42, "concept_avoiding", "âme", neg_out, _n("âme", tok)),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_linear_probe_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        direction = load_file(str(out_paths[0]))["direction"]
        assert t.isfinite(direction).all()

    def test_nan_in_activations_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Inject a NaN into one captured activation — the training
        function must raise ``AxBenchBaseError`` with ``"NaN"`` in the
        message.
        """
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        def _pos(prefix):
            if "NAN" in prefix:
                return t.full((TINY_N_EMBD,), float("nan"))
            v = t.zeros(TINY_N_EMBD)
            v[0] = 2.0
            return v

        def _neg(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = -2.0
            return v

        _install_stub_capture(monkeypatch, _pos, _neg)

        rows = [
            _pos_row(42, "concept_rich", "POS_NAN", "x", 3),
            _pos_row(42, "concept_rich", "POS_OK", "x", 3),
            _pos_row(42, "concept_avoiding", "neg_1", "x", 3),
            _pos_row(42, "concept_avoiding", "neg_2", "x", 3),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        with pytest.raises(AxBenchBaseError, match="NaN"):
            train_linear_probe_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )


# ---------------------------------------------------------------------------
# Tests: degenerate / empty-side errors
# ---------------------------------------------------------------------------


class TestAbsentConcepts:
    def test_any_absent_concept_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The explicit concept population cannot silently shrink at training."""
        # Isolate paths.run_output_root to tmp_path so stale nanochat_cache
        # artifacts from previous test runs cannot cause false failures.
        monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        pos = t.zeros(TINY_N_EMBD)
        pos[0] = 1.0
        neg = t.zeros(TINY_N_EMBD)
        neg[1] = 1.0

        _install_stub_capture(monkeypatch, lambda _: pos.clone(), lambda _: neg.clone())

        rows = [
            _pos_row(42, "concept_rich", "POS a", "out", 1),
            _pos_row(42, "concept_rich", "POS b", "out", 1),
            _pos_row(42, "concept_avoiding", "NEG a", "out", 1),
            _pos_row(42, "concept_avoiding", "NEG b", "out", 1),
        ]
        corpus = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus, rows)

        with pytest.raises(AxBenchBaseError, match=r"concepts \[99\].*positive"):
            train_linear_probe_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42), _concept(99)],
                training_corpus_path=corpus,
            )

    def test_all_concepts_absent_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ALL requested concepts are absent from the corpus, raise
        AxBenchBaseError instead of silently returning an empty list.

        TDD RED: this test fails before the all-dropped guard is added.
        """
        monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        # Corpus has rows for concept 99 only; requested concepts 1 and 2
        # are completely absent.
        rows = [
            _pos_row(99, "concept_rich", "POS a", "out", 1),
            _pos_row(99, "concept_avoiding", "NEG a", "out", 1),
        ]
        corpus = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus, rows)

        with pytest.raises(AxBenchBaseError, match="absent from the"):
            train_linear_probe_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(1), _concept(2)],
                training_corpus_path=corpus,
            )





class TestEmptySides:
    def test_empty_positives_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()
        tok = _StubNanochatTokenizer()

        rows = [
            _pos_row(42, "concept_avoiding", "NEG", "output", _n("NEG", tok)),
            _pos_row(42, "concept_avoiding", "NEG2", "output", _n("NEG2", tok)),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        with pytest.raises(AxBenchBaseError, match="positives"):
            train_linear_probe_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )

    def test_empty_negatives_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()
        tok = _StubNanochatTokenizer()

        rows = [
            _pos_row(42, "concept_rich", "POS", "output", _n("POS", tok)),
            _pos_row(42, "concept_rich", "POS2", "output", _n("POS2", tok)),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        with pytest.raises(AxBenchBaseError, match="negatives"):
            train_linear_probe_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )

    def test_degenerate_all_same_label_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A corpus with only one label (all positives) must raise with a
        clear message about the missing side. Mirrors the plan item
        ``test_degenerate_all_same_label_raises``.
        """
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()
        tok = _StubNanochatTokenizer()

        # All-positive corpus
        rows_all_pos = [
            _pos_row(42, "concept_rich", f"POS_{i}", "out", _n(f"POS_{i}", tok))
            for i in range(3)
        ]
        p1 = tmp_path / "all_pos.parquet"
        _write_positive_corpus(p1, rows_all_pos)
        with pytest.raises(AxBenchBaseError, match="negatives"):
            train_linear_probe_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=p1,
            )

        # All-negative corpus
        rows_all_neg = [
            _pos_row(42, "concept_avoiding", f"NEG_{i}", "out", _n(f"NEG_{i}", tok))
            for i in range(3)
        ]
        p2 = tmp_path / "all_neg.parquet"
        _write_positive_corpus(p2, rows_all_neg)
        with pytest.raises(AxBenchBaseError, match="positives"):
            train_linear_probe_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=p2,
            )


# ---------------------------------------------------------------------------
# Tests: per-direction max-activation metadata (#1228)
# ---------------------------------------------------------------------------


class TestMaxActivationMetadata:
    @staticmethod
    def _rows(tok: _StubNanochatTokenizer) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for i in range(8):
            pos_prefix = f"POS_{i}"
            neg_prefix = f"NEG_{i}"
            rows.append(
                _pos_row(42, "concept_rich", pos_prefix, "ZZZZ", _n(pos_prefix, tok))
            )
            rows.append(
                _pos_row(
                    42,
                    "concept_avoiding",
                    neg_prefix,
                    "aaaa",
                    _n(neg_prefix, tok),
                )
            )
        return rows

    def test_metadata_value_equals_independent_recomputation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()
        rows = self._rows(tok)
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_path = train_linear_probe_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )[0]
        with safe_open(str(out_path), framework="pt") as handle:
            meta = handle.metadata() or {}
            direction = handle.get_tensor("direction")

        row_acts = [
            capture_mod._capture_per_row_continuation(
                model=model,
                layer=TINY_LAYER,
                tokenizer=tok,
                prefix_text=row["prefix_text"],
                output_text=row["output_text"],
                n_prefix_nanochat_tokens=row["n_prefix_nanochat_tokens"],
            )[1]
            for row in rows
        ]
        expected_max = max(float((acts @ direction).max().item()) for acts in row_acts)
        stored = float(meta[MAX_ACTIVATION_METADATA_KEY])
        assert stored == expected_max
        assert stored > 0.0

    def test_direction_coef_intercept_bytes_identical_to_prefix_formula(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()
        rows = self._rows(tok)
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_path = train_linear_probe_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )[0]
        saved = load_file(str(out_path))

        means = [
            _independent_row_continuation_mean(model, TINY_LAYER, tok, row)
            for row in rows
        ]
        x_numpy = t.stack(means, dim=0).numpy().astype(np.float32)
        y_numpy = np.asarray(
            [1 if row["content_class"] == "concept_rich" else 0 for row in rows],
            dtype=np.int64,
        )
        probe = LogisticRegression(
            l1_ratio=1.0,
            C=1.0,
            max_iter=500,
            solver="liblinear",
            random_state=cfg.generation.master_seed,
        )
        probe.fit(x_numpy, y_numpy)
        coef = t.from_numpy(np.asarray(probe.coef_, dtype=np.float32)).contiguous()
        intercept = t.from_numpy(
            np.asarray(probe.intercept_, dtype=np.float32)
        ).contiguous()
        raw_direction = coef[0].clone()
        direction = (
            raw_direction
            if raw_direction.norm().item() == 0.0
            else F.normalize(raw_direction, dim=-1)
        )
        assert t.equal(saved["direction"], direction)
        assert t.equal(saved["coef"], coef)
        assert t.equal(saved["intercept"], intercept)

    def test_zero_direction_stores_zero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()
        rows: list[dict[str, Any]] = []
        for i in range(4):
            prefix = f"PAIR_{i}"
            n_prefix = _n(prefix, tok)
            rows.append(_pos_row(42, "concept_rich", prefix, "same", n_prefix))
            rows.append(_pos_row(42, "concept_avoiding", prefix, "same", n_prefix))
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        # Duplicated pos/neg feature rows make w=0 the true L1 optimum, and
        # liblinear's coordinate descent soft-thresholds coefficients to EXACT
        # zero (not asymptotically small), so the trainer's bitwise
        # `norm == 0.0` zero-direction branch fires deterministically here.
        with _capture_loguru_warnings() as warnings:
            out_path = train_linear_probe_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )[0]
        with safe_open(str(out_path), framework="pt") as handle:
            meta = handle.metadata() or {}
            direction = handle.get_tensor("direction")
        assert t.equal(direction, t.zeros_like(direction))
        assert float(meta[MAX_ACTIVATION_METADATA_KEY]) == 0.0
        assert any("zero" in message.lower() for message in warnings)

    def test_max_activation_wiring_includes_negative_rows(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A peak planted in a concept_avoiding row must reach the stored value.

        The pure function's pos+neg coverage is pinned in
        ``test_direction_scale.py``; this pins the TRAINER call site
        (``pos_acts + neg_acts``) — a regression dropping ``+ neg_acts`` would
        store 1.0 here instead of the planted 50.0. (Features differ only on
        axis 0, so L1 zeroes every other coefficient and the normalized probe
        direction is exactly ``+e0``.)
        """
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()
        base = t.zeros(TINY_N_EMBD)
        base[0] = 1.0

        def _stub_capture(
            *,
            model: Any,
            layer: int,
            tokenizer: Any,
            prefix_text: str,
            output_text: str,
            n_prefix_nanochat_tokens: int,
            encoder_conditioning: Any = None,
        ) -> tuple[t.Tensor, t.Tensor]:
            del model, layer, tokenizer, output_text, n_prefix_nanochat_tokens
            if "POS" in prefix_text:
                return base.clone(), base.unsqueeze(0).clone()
            neg_acts = t.stack([-base, 50.0 * base], dim=0)
            return -base.clone(), neg_acts

        monkeypatch.setattr(
            capture_mod, "_capture_per_row_continuation", _stub_capture
        )
        rows = [
            _pos_row(42, "concept_rich", "POS a", "out", 1),
            _pos_row(42, "concept_rich", "POS b", "out", 1),
            _pos_row(42, "concept_avoiding", "NEG a", "out", 1),
            _pos_row(42, "concept_avoiding", "NEG b", "out", 1),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_path = train_linear_probe_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )[0]
        with safe_open(str(out_path), framework="pt") as handle:
            meta = handle.metadata() or {}
            direction = handle.get_tensor("direction")
        expected_e0 = t.zeros(TINY_N_EMBD)
        expected_e0[0] = 1.0
        assert t.equal(direction, expected_e0)
        assert float(meta[MAX_ACTIVATION_METADATA_KEY]) == 50.0


# ---------------------------------------------------------------------------
# Tests: output path + metadata
# ---------------------------------------------------------------------------


class TestOutputPersistence:
    def test_output_saved_to_correct_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        def _pos(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = 2.0
            return v

        def _neg(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = -2.0
            return v

        _install_stub_capture(monkeypatch, _pos, _neg)

        rows = [
            _pos_row(42, "concept_rich", "POS_1", "out", 3),
            _pos_row(42, "concept_rich", "POS_2", "out", 3),
            _pos_row(42, "concept_avoiding", "neg_1", "out", 3),
            _pos_row(42, "concept_avoiding", "neg_2", "out", 3),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_linear_probe_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        expected = test_paths.default_paths(cast(str, cfg.run_tag)).linear_probe_direction(
            target_model="decoder_only", mode=SteeringMode.POSITIVE, concept_id=42
        )
        assert out_paths == [expected]
        assert expected.exists()

    def test_saved_artifact_has_coef_intercept_direction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        def _pos(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = 2.0
            return v

        def _neg(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = -2.0
            return v

        _install_stub_capture(monkeypatch, _pos, _neg)

        rows = [
            _pos_row(42, "concept_rich", "POS_1", "out", 3),
            _pos_row(42, "concept_rich", "POS_2", "out", 3),
            _pos_row(42, "concept_avoiding", "neg_1", "out", 3),
            _pos_row(42, "concept_avoiding", "neg_2", "out", 3),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_linear_probe_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        payload = load_file(str(out_paths[0]))
        assert "direction" in payload
        assert "coef" in payload
        assert "intercept" in payload
        assert payload["direction"].shape == (TINY_N_EMBD,)
        assert payload["coef"].shape == (1, TINY_N_EMBD)
        assert payload["intercept"].shape == (1,)

    def test_save_includes_expected_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "cache"))
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        def _pos(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = 2.0
            return v

        def _neg(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = -2.0
            return v

        _install_stub_capture(monkeypatch, _pos, _neg)

        rows = [
            _pos_row(99, "concept_rich", "POS_1", "out", 3),
            _pos_row(99, "concept_rich", "POS_2", "out", 3),
            _pos_row(99, "concept_avoiding", "neg_1", "out", 3),
            _pos_row(99, "concept_avoiding", "neg_2", "out", 3),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_linear_probe_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(99)],
            training_corpus_path=corpus_path,
        )
        with safe_open(str(out_paths[0]), framework="pt") as f:
            meta = f.metadata() or {}
        for key in (
            "concept_id", "mode", "target_model", "layer",
            "n_pos", "n_neg", "C", "penalty", MAX_ACTIVATION_METADATA_KEY,
        ):
            assert key in meta, f"Missing metadata key: {key}"
        assert meta["concept_id"] == "99"
        assert meta["mode"] == "positive"
        assert meta["target_model"] == "decoder_only"
        assert meta["layer"] == str(TINY_LAYER)
        assert meta["n_pos"] == "2"
        assert meta["n_neg"] == "2"
        assert meta["C"] == "1.0"
        assert meta["penalty"] == "l1"
        assert "feature_contract_fingerprint" not in meta
        assert "source_corpus_sha256" not in meta
        assert "artifact_kind" not in meta

        legacy_payload = {
            key: value.clone()
            for key, value in load_file(str(out_paths[0])).items()
        }
        fingerprint = "b" * 64
        clustered_cfg = cfg.replace(run_tag="linear_probe_test_clustered")
        clustered_paths = train_linear_probe_all(
            cfg=clustered_cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(99)],
            training_corpus_path=corpus_path,
            feature_contract_fingerprint=fingerprint,
        )
        clustered_path = clustered_paths[0]
        with safe_open(str(clustered_path), framework="pt") as f:
            clustered_meta = f.metadata() or {}
        assert clustered_meta["feature_contract_fingerprint"] == fingerprint
        assert clustered_meta["source_corpus_sha256"] == hashlib.sha256(
            corpus_path.read_bytes()
        ).hexdigest()
        assert clustered_meta["artifact_kind"] == "linear_probe_direction_v1"
        assert (
            clustered_meta[MAX_ACTIVATION_METADATA_KEY]
            == meta[MAX_ACTIVATION_METADATA_KEY]
        )
        clustered_payload = load_file(str(clustered_path))
        assert clustered_payload.keys() == legacy_payload.keys()
        for key, legacy_value in legacy_payload.items():
            assert t.equal(clustered_payload[key], legacy_value)

        exact_bytes = clustered_path.read_bytes()
        assert train_linear_probe_all(
            cfg=clustered_cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(99)],
            training_corpus_path=corpus_path,
            feature_contract_fingerprint=fingerprint,
        ) == clustered_paths
        assert clustered_path.read_bytes() == exact_bytes

        clustered_path.write_bytes(b"different existing bytes")
        with pytest.raises(AxBenchBaseError, match="refusing to overwrite") as exc_info:
            train_linear_probe_all(
                cfg=clustered_cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(99)],
                training_corpus_path=corpus_path,
                feature_contract_fingerprint=fingerprint,
            )
        assert isinstance(exc_info.value.__cause__, ImmutableArtifactError)

    def test_multiple_concepts_produce_separate_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        def _pos(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = 2.0
            return v

        def _neg(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = -2.0
            return v

        _install_stub_capture(monkeypatch, _pos, _neg)

        rows = []
        for cid in (1, 2):
            for i in range(2):
                rows.append(
                    _pos_row(cid, "concept_rich", f"POS_{cid}_{i}", "out", 3)
                )
                rows.append(
                    _pos_row(cid, "concept_avoiding", f"neg_{cid}_{i}", "out", 3)
                )
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_linear_probe_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(1), _concept(2)],
            training_corpus_path=corpus_path,
        )
        assert len(out_paths) == 2
        assert len({str(p) for p in out_paths}) == 2  # distinct files
        for cid, path in zip([1, 2], out_paths, strict=True):
            with safe_open(str(path), framework="pt") as f:
                meta = f.metadata() or {}
            assert meta["concept_id"] == str(cid)


# ---------------------------------------------------------------------------
# Tests: global negatives + negative-mode schema + error handling
# ---------------------------------------------------------------------------


class TestCorpusSemantics:
    def test_per_concept_negatives_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Positive-mode negatives are now per-concept ``concept_avoiding``
        rows keyed to the same ``concept_id`` (no global pool, no
        ``concept_id == -1`` sentinel). They must be consumed as the
        negative side, and a ``concept_avoiding`` row keyed to a different
        concept must be ignored.
        """
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        def _pos(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = 2.0
            return v

        def _neg(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = -2.0
            return v

        _install_stub_capture(monkeypatch, _pos, _neg)

        rows = [
            _pos_row(42, "concept_rich", "POS_1", "x", 3),
            _pos_row(42, "concept_rich", "POS_2", "x", 3),
            _pos_row(42, "concept_avoiding", "neg_1", "x", 3),
            _pos_row(42, "concept_avoiding", "neg_2", "x", 3),
            # A different concept's avoiding row must NOT leak in.
            _pos_row(7, "concept_avoiding", "other_neg", "x", 3),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_linear_probe_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        with safe_open(str(out_paths[0]), framework="pt") as f:
            meta = f.metadata() or {}
        assert meta["n_pos"] == "2"
        # Only the two concept-42 avoiding rows; concept-7's row excluded.
        assert meta["n_neg"] == "2"

    def test_negative_mode_training_corpus_uses_content_class_split(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Negative-mode training corpus has 19 columns. Same positive/
        negative split via ``content_class``."""
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        def _pos(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = 2.0
            return v

        def _neg(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = -2.0
            return v

        _install_stub_capture(monkeypatch, _pos, _neg)

        rows = [
            _neg_row(42, "concept_rich", "POS_1", "x", 3),
            _neg_row(42, "concept_rich", "POS_2", "x", 3),
            _neg_row(42, "concept_avoiding", "neg_1", "x", 3),
            _neg_row(42, "concept_avoiding", "neg_2", "x", 3),
        ]
        corpus_path = tmp_path / "corpus_neg.parquet"
        _write_negative_corpus(corpus_path, rows)

        out_paths = train_linear_probe_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.NEGATIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        with safe_open(str(out_paths[0]), framework="pt") as f:
            meta = f.metadata() or {}
        assert meta["mode"] == "negative"
        assert meta["n_pos"] == "2"
        assert meta["n_neg"] == "2"


class TestErrorHandling:
    def test_empty_concepts_list_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        corpus_path = tmp_path / "empty.parquet"
        _write_positive_corpus(corpus_path, [])

        with pytest.raises(AxBenchBaseError, match="empty concepts"):
            train_linear_probe_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[],
                training_corpus_path=corpus_path,
            )

    def test_run_tag_none_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path).replace(run_tag=None)
        model = _make_tiny_gpt()

        def _pos(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = 2.0
            return v

        def _neg(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = -2.0
            return v

        _install_stub_capture(monkeypatch, _pos, _neg)

        rows = [
            _pos_row(42, "concept_rich", "POS_1", "x", 3),
            _pos_row(42, "concept_rich", "POS_2", "x", 3),
            _pos_row(42, "concept_avoiding", "neg_1", "x", 3),
            _pos_row(42, "concept_avoiding", "neg_2", "x", 3),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        with pytest.raises(AxBenchBaseError, match="run_tag"):
            train_linear_probe_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )

    def test_fit_producing_nan_coef_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the fitted probe produces NaN coefficients (degenerate fit),
        the training function must raise ``AxBenchBaseError``. We stub
        ``LogisticRegression`` to return a fake probe with NaN coef_.
        """
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        def _pos(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = 2.0
            return v

        def _neg(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = -2.0
            return v

        _install_stub_capture(monkeypatch, _pos, _neg)

        class _NaNProbe:
            def __init__(self, **kwargs):
                del kwargs
                self.coef_ = np.full((1, TINY_N_EMBD), float("nan"), dtype=np.float32)
                self.intercept_ = np.zeros(1, dtype=np.float32)

            def fit(self, x, y):
                del x, y
                return self

        monkeypatch.setattr(linear_probe_mod, "LogisticRegression", _NaNProbe)

        rows = [
            _pos_row(42, "concept_rich", "POS_1", "x", 3),
            _pos_row(42, "concept_rich", "POS_2", "x", 3),
            _pos_row(42, "concept_avoiding", "neg_1", "x", 3),
            _pos_row(42, "concept_avoiding", "neg_2", "x", 3),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        with pytest.raises(AxBenchBaseError, match="NaN"):
            train_linear_probe_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )

    def test_fit_producing_zero_coef_warns_and_saves_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If L1 drives every coefficient to zero, the direction must be
        a zero vector (not a NaN from ``0 / 0``), and a warning must be
        logged. We stub ``LogisticRegression`` to guarantee this.
        """
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        def _pos(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = 2.0
            return v

        def _neg(prefix):
            v = t.zeros(TINY_N_EMBD)
            v[0] = -2.0
            return v

        _install_stub_capture(monkeypatch, _pos, _neg)

        class _ZeroProbe:
            def __init__(self, **kwargs):
                del kwargs
                self.coef_ = np.zeros((1, TINY_N_EMBD), dtype=np.float32)
                self.intercept_ = np.zeros(1, dtype=np.float32)

            def fit(self, x, y):
                del x, y
                return self

        monkeypatch.setattr(linear_probe_mod, "LogisticRegression", _ZeroProbe)

        rows = [
            _pos_row(42, "concept_rich", "POS_1", "x", 3),
            _pos_row(42, "concept_rich", "POS_2", "x", 3),
            _pos_row(42, "concept_avoiding", "neg_1", "x", 3),
            _pos_row(42, "concept_avoiding", "neg_2", "x", 3),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        collected: list[str] = []
        handler_id = loguru_logger.add(
            lambda msg: collected.append(msg.record["message"]),
            level="WARNING",
        )
        try:
            out_paths = train_linear_probe_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )
        finally:
            loguru_logger.remove(handler_id)

        direction = load_file(str(out_paths[0]))["direction"]
        assert t.isfinite(direction).all(), "Direction must be nan-free."
        assert direction.abs().max().item() == 0.0, "Direction must be exact zero."
        zero_warnings = [w for w in collected if "zero" in w.lower()]
        assert zero_warnings, f"Expected a zero-coef warning, got {collected}"

    def test_row_with_empty_continuation_fails_loudly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A committed empty continuation must abort instead of changing data."""
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        prefix_full = "FULL"
        full_n = _n(prefix_full, tok)

        # This row has output_text="" and must make the artifact fail loudly.
        rows = [
            # Corrupt row:
            _pos_row(42, "concept_rich", prefix_full, "", full_n),
            _pos_row(42, "concept_rich", "GOOD1", "AAA", _n("GOOD1", tok)),
            _pos_row(42, "concept_rich", "GOOD2", "BBB", _n("GOOD2", tok)),
            _pos_row(42, "concept_avoiding", "bad1", "xyz", _n("bad1", tok)),
            _pos_row(42, "concept_avoiding", "bad2", "qrs", _n("bad2", tok)),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        def _unexpected_save(*args, **kwargs):
            del args, kwargs
            raise AssertionError(
                "LinearProbe wrote an artifact after invalid synth input"
            )

        monkeypatch.setattr(linear_probe_mod, "save_file", _unexpected_save)

        with pytest.raises(
            AxBenchBaseError,
            match=r"LinearProbe concept 42 positive row 0.*empty output_text",
        ):
            train_linear_probe_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )
