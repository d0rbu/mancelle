"""Unit tests for DiffMean training (Task 21).

Covers ``steering.eval.axbench_base.baselines.diffmean.train_diffmean_all``
and its per-concept helper, which computes a unit-norm residual-stream
DiffMean direction from a training corpus parquet and writes it to the
path returned by ``paths.diffmean_direction(...)``.

TDD design choices pinned down by these tests:
  * Label column is ``content_class``: ``"concept_rich"`` is positive,
    ``"concept_avoiding"`` is negative, for BOTH positive-mode and
    negative-mode training corpora.
  * Per-row continuation activations start at the exact first joint-encoding
    token that overlaps ``output_text``. ``n_prefix_nanochat_tokens`` stores
    that owned-BOS-inclusive index and must match the shared pair tokenizer.
  * Empty continuations are corrupt synth rows and fail loudly.
  * The resulting direction has unit norm after ``normalize``; the
    degenerate pos==neg case logs a warning and returns a zero vector.
  * NaN in the direction is a hard error.
  * Metadata is written via the ``metadata=`` kwarg of ``safetensors.torch.save_file``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as loguru_logger
from safetensors import safe_open
from safetensors.torch import load_file

from nanochat.gpt import GPT, GPTConfig
from steering.eval.axbench_base.artifact_metadata import (
    schema_with_extraction_tokenizer_metadata,
)
from steering.eval.axbench_base.baselines import _capture as capture_mod
from steering.eval.axbench_base.baselines import diffmean as diffmean_mod
from steering.eval.axbench_base.baselines._capture import (
    _ResidualCaptureHook,
)
from steering.eval.axbench_base.baselines.diffmean import (
    train_diffmean_all,
)
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
# Tiny GPT fixture + stub tokenizer
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
    """Deterministic nanochat-like tokenizer for tests.

    ``encode(text, prepend=bos)`` returns ``[bos] + [ord(c) % VOCAB for c in text]``.
    This mirrors the real tokenizer's BOS-prepend convention exactly.
    """

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
    monkeypatch.setattr(diffmean_mod, "get_tokenizer", lambda: tok)
    return tok


# ---------------------------------------------------------------------------
# Helpers: corpus construction
# ---------------------------------------------------------------------------


# Positive-mode training schema (15 columns). Mirrors
# ``steering.eval.axbench_base.synth.positive_mode.TRAINING_SCHEMA`` verbatim.
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


# Negative-mode training schema (19 columns). Mirrors
# ``synth.negative_mode.NEG_TRAINING_SCHEMA``.
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


def _write_positive_corpus(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
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


def _write_negative_corpus(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
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
        run_tag="diffmean_test",
        output_root=None,
        device="cpu",
    )
    # Override the diffmean layer to 0 so our 2-layer tiny fixture works.
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


# ---------------------------------------------------------------------------
# Tests: unit norm + direction math
# ---------------------------------------------------------------------------


class TestUnitNorm:
    def test_direction_has_unit_norm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """4 positives vs 4 negatives → unit-norm direction."""
        _install_stub_tokenizer(monkeypatch)
        model = _make_tiny_gpt()
        cfg = _tiny_cfg(tmp_path)
        tok = _StubNanochatTokenizer()

        prefix = "AAAA"
        pos_output = "cdef"
        neg_output = "xyz!"
        n_prefix = len(tok.encode(prefix, prepend=tok.get_bos_token_id()))

        rows = []
        for i in range(4):
            rows.append(
                _pos_row(
                    concept_id=42,
                    content_class="concept_rich",
                    prefix_text=f"AAA{chr(65 + i)}",  # vary a little
                    output_text=pos_output,
                    n_prefix_nanochat_tokens=n_prefix,
                )
            )
        for i in range(4):
            rows.append(
                _pos_row(
                    concept_id=42,
                    content_class="concept_avoiding",
                    prefix_text=f"BBB{chr(65 + i)}",
                    output_text=neg_output,
                    n_prefix_nanochat_tokens=n_prefix,
                )
            )
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_diffmean_all(
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
        assert direction.shape == (TINY_N_EMBD,)
        assert t.allclose(direction.norm(), t.tensor(1.0), atol=1e-5)

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

        vec = t.zeros(TINY_N_EMBD)
        vec[0] = 1.0

        def _fake_capture(*, model, layer, tokenizer, prefix_text, output_text, n_prefix_nanochat_tokens, encoder_conditioning=None):
            del model, layer, tokenizer, output_text, n_prefix_nanochat_tokens
            mean = vec.clone() if "POS" in prefix_text else (vec.clone() * -1.0)
            return mean, mean.unsqueeze(0)

        monkeypatch.setattr(capture_mod, "_capture_per_row_continuation", _fake_capture)

        # Corpus has rows for concept 42 only; concept 99 is completely absent.
        rows = [
            _pos_row(42, "concept_rich", "POS a", "out", 1),
            _pos_row(42, "concept_avoiding", "NEG a", "out", 1),
        ]
        corpus = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus, rows)

        with pytest.raises(AxBenchBaseError, match=r"concepts \[99\].*positive"):
            train_diffmean_all(
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
            train_diffmean_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(1), _concept(2)],
                training_corpus_path=corpus,
            )


    def test_direction_is_pos_minus_neg(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """By stubbing ``_capture_per_row_continuation`` to return deterministic
        per-row vectors, verify ``normalize(mean(pos) - mean(neg))`` exactly.

        This isolates the aggregation math from any GPT forward behaviour.
        """
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        pos_vec = t.zeros(TINY_N_EMBD)
        pos_vec[0] = 1.0
        neg_vec = t.zeros(TINY_N_EMBD)
        neg_vec[1] = 1.0

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
            del model, layer, tokenizer, n_prefix_nanochat_tokens
            if "POS" in prefix_text:
                mean = pos_vec.clone()
            else:
                mean = neg_vec.clone()
            return mean, mean.unsqueeze(0)

        monkeypatch.setattr(
            capture_mod,
            "_capture_per_row_continuation",
            _fake_capture,
        )

        rows = [
            _pos_row(42, "concept_rich", "POS_A", "out", 3),
            _pos_row(42, "concept_rich", "POS_B", "out", 3),
            _pos_row(42, "concept_avoiding", "NEG_A", "out", 3),
            _pos_row(42, "concept_avoiding", "NEG_B", "out", 3),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_diffmean_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        direction = load_file(str(out_paths[0]))["direction"]

        expected = (pos_vec - neg_vec)
        expected = expected / expected.norm()
        assert t.allclose(direction, expected, atol=1e-6)




# ---------------------------------------------------------------------------
# Tests: empty-side errors
# ---------------------------------------------------------------------------


class TestEmptySides:
    def test_empty_positives_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()
        tok = _StubNanochatTokenizer()

        rows = [
            _pos_row(
                42, "concept_avoiding", "NEG", "output",
                len(tok.encode("NEG", prepend=tok.get_bos_token_id())),
            ),
            _pos_row(
                42, "concept_avoiding", "NEG2", "output",
                len(tok.encode("NEG2", prepend=tok.get_bos_token_id())),
            ),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        with pytest.raises(AxBenchBaseError, match="positives"):
            train_diffmean_all(
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
            _pos_row(
                42, "concept_rich", "POS", "output",
                len(tok.encode("POS", prepend=tok.get_bos_token_id())),
            ),
            _pos_row(
                42, "concept_rich", "POS2", "output",
                len(tok.encode("POS2", prepend=tok.get_bos_token_id())),
            ),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        with pytest.raises(AxBenchBaseError, match="negatives"):
            train_diffmean_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )


# ---------------------------------------------------------------------------
# Tests: continuation slicing
# ---------------------------------------------------------------------------


class TestActivationSlicing:
    def test_activation_extraction_slices_at_n_prefix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Instrument the capture hook to record the slice length seen
        per row and verify it matches the expected continuation count.

        ``prefix_text`` encodes to ``N`` chars (ord-hashed per stub) + BOS.
        ``output_text`` encodes to ``M`` chars. ``n_prefix_nanochat_tokens``
        as recorded by Stage 0b/1 is ``N + 1`` (including BOS). After the
        forward pass, the continuation slice should span positions
        ``[N+1, N+M+1)``, i.e. exactly ``M`` positions.
        """
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        # Collect slice lengths observed by the real capture helper.
        observed_slice_lengths: list[int] = []

        real_capture = capture_mod._capture_per_row_continuation

        def _tracing_capture(**kwargs):
            # Do the forward ourselves to observe the slice length, but
            # still delegate to the real function for the actual mean.
            prefix = kwargs["prefix_text"]
            output = kwargs["output_text"]
            n_prefix = kwargs["n_prefix_nanochat_tokens"]
            full_tokens, exact_start = tokenize_labeled_pair(tok, prefix, output)
            assert n_prefix == exact_start
            # Expected slice: positions at and after the exact output-overlap
            # token boundary committed by Stage 1.
            slice_len = max(0, len(full_tokens) - n_prefix)
            observed_slice_lengths.append(slice_len)
            return real_capture(**kwargs)

        monkeypatch.setattr(
            capture_mod,
            "_capture_per_row_continuation",
            _tracing_capture,
        )

        prefix = "PREFIX"  # 6 chars
        output = "OUTPUT"  # 6 chars
        _tokens, n_prefix = tokenize_labeled_pair(tok, prefix, output)
        assert n_prefix == 7  # 1 BOS + 6 chars

        rows = [
            _pos_row(42, "concept_rich", prefix, output, n_prefix),
            _pos_row(42, "concept_avoiding", prefix, output, n_prefix),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        train_diffmean_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )

        # Every row's continuation slice must cover exactly the 6 output chars.
        assert all(s == 6 for s in observed_slice_lengths), (
            f"Expected every slice length to be 6, got {observed_slice_lengths}"
        )

    def test_row_with_empty_continuation_fails_loudly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A committed empty continuation must abort instead of changing data."""
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        prefix = "FULL"
        full_n = len(tok.encode(prefix, prepend=tok.get_bos_token_id()))

        # Positives and negatives use DIFFERENT continuation tokens so that
        # the direction is nonzero even for an untrained tiny GPT (whose
        # init_weights zeros the attention/MLP c_proj, making block-output
        # a function of token embeddings alone). Choose chars with wide
        # ord spread so the token-embedding difference is material.
        rows = [
            # This corrupt row must abort training (output_text=""):
            _pos_row(42, "concept_rich", prefix, "", full_n),
            _pos_row(42, "concept_rich", "GOOD1", "AAA",
                     len(tok.encode("GOOD1", prepend=tok.get_bos_token_id()))),
            _pos_row(42, "concept_rich", "GOOD2", "BBB",
                     len(tok.encode("GOOD2", prepend=tok.get_bos_token_id()))),
            _pos_row(42, "concept_avoiding", "bad1", "xyz",
                     len(tok.encode("bad1", prepend=tok.get_bos_token_id()))),
            _pos_row(42, "concept_avoiding", "bad2", "qrs",
                     len(tok.encode("bad2", prepend=tok.get_bos_token_id()))),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        def _unexpected_save(*args, **kwargs):
            del args, kwargs
            raise AssertionError("DiffMean wrote an artifact after invalid synth input")

        monkeypatch.setattr(diffmean_mod, "save_file", _unexpected_save)

        with pytest.raises(
            AxBenchBaseError,
            match=r"DiffMean concept 42 positive row 0.*empty output_text",
        ):
            train_diffmean_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )


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


# ---------------------------------------------------------------------------
# Tests: unicode + degenerate cases
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_handles_unicode_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()
        tok = _StubNanochatTokenizer()

        # Positives and negatives use distinct continuation texts so that
        # mean_pos != mean_neg on our (untrained) tiny GPT fixture (whose
        # projection weights are zero, making the block output a function
        # of token embeddings alone).
        pos_out = "résumé"
        neg_out = "naïveté"

        rows = [
            _pos_row(42, "concept_rich", "café", pos_out,
                     len(tok.encode("café", prepend=tok.get_bos_token_id()))),
            _pos_row(42, "concept_rich", "café2", pos_out,
                     len(tok.encode("café2", prepend=tok.get_bos_token_id()))),
            _pos_row(42, "concept_avoiding", "élan", neg_out,
                     len(tok.encode("élan", prepend=tok.get_bos_token_id()))),
            _pos_row(42, "concept_avoiding", "étude", neg_out,
                     len(tok.encode("étude", prepend=tok.get_bos_token_id()))),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_diffmean_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        direction = load_file(str(out_paths[0]))["direction"]
        assert t.isfinite(direction).all()
        assert t.allclose(direction.norm(), t.tensor(1.0), atol=1e-5)

    def test_degenerate_pos_equals_neg_near_zero_direction(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When all positives equal all negatives, direction norm is ~0.
        The function must log a warning and return a zero (nan-free) vector.
        """
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        identical_vec = t.ones(TINY_N_EMBD) * 0.5

        def _fake_capture(**kwargs):
            del kwargs
            mean = identical_vec.clone()
            return mean, mean.unsqueeze(0)

        monkeypatch.setattr(
            capture_mod,
            "_capture_per_row_continuation",
            _fake_capture,
        )

        rows = [
            _pos_row(42, "concept_rich", "a", "b", 2),
            _pos_row(42, "concept_rich", "c", "d", 2),
            _pos_row(42, "concept_avoiding", "e", "f", 2),
            _pos_row(42, "concept_avoiding", "g", "h", 2),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        with _capture_loguru_warnings() as warnings:
            out_paths = train_diffmean_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )

        direction = load_file(str(out_paths[0]))["direction"]
        assert t.isfinite(direction).all(), "Direction must be nan-free."
        assert direction.abs().max().item() == 0.0, "Direction must be exact zero."
        zero_warnings = [w for w in warnings if "zero" in w.lower()]
        assert zero_warnings, (
            f"Expected a zero-direction warning, got {warnings}"
        )


# ---------------------------------------------------------------------------
# Tests: per-direction max-activation metadata (#1228)
# ---------------------------------------------------------------------------


class TestMaxActivationMetadata:
    def test_metadata_value_equals_independent_recomputation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        def _n(text: str) -> int:
            return len(tok.encode(text, prepend=tok.get_bos_token_id()))

        rows = [
            _pos_row(42, "concept_rich", "POS_A", "cdef", _n("POS_A")),
            _pos_row(42, "concept_rich", "POS_B", "defg", _n("POS_B")),
            _pos_row(42, "concept_avoiding", "NEG_A", "xyz!", _n("NEG_A")),
            _pos_row(42, "concept_avoiding", "NEG_B", "wxyz", _n("NEG_B")),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_path = train_diffmean_all(
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

    def test_direction_bytes_identical_to_prefix_formula(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        def _n(text: str) -> int:
            return len(tok.encode(text, prepend=tok.get_bos_token_id()))

        rows = [
            _pos_row(42, "concept_rich", "POS_A", "cdef", _n("POS_A")),
            _pos_row(42, "concept_rich", "POS_B", "defg", _n("POS_B")),
            _pos_row(42, "concept_avoiding", "NEG_A", "xyz!", _n("NEG_A")),
            _pos_row(42, "concept_avoiding", "NEG_B", "wxyz", _n("NEG_B")),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_path = train_diffmean_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )[0]
        saved_direction = load_file(str(out_path))["direction"]

        means = [
            _independent_row_continuation_mean(model, TINY_LAYER, tok, row)
            for row in rows
        ]
        recomputed = F.normalize(
            t.stack(means[:2], dim=0).mean(dim=0)
            - t.stack(means[2:], dim=0).mean(dim=0),
            dim=-1,
        )
        assert t.equal(saved_direction, recomputed)

    def test_zero_direction_stores_zero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        def _n(text: str) -> int:
            return len(tok.encode(text, prepend=tok.get_bos_token_id()))

        rows = [
            _pos_row(42, "concept_rich", "PAIR_A", "same", _n("PAIR_A")),
            _pos_row(42, "concept_avoiding", "PAIR_A", "same", _n("PAIR_A")),
            _pos_row(42, "concept_rich", "PAIR_B", "same", _n("PAIR_B")),
            _pos_row(42, "concept_avoiding", "PAIR_B", "same", _n("PAIR_B")),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        with _capture_loguru_warnings() as warnings:
            out_path = train_diffmean_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )[0]
        with safe_open(str(out_path), framework="pt") as handle:
            meta = handle.metadata() or {}
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
        store 1.0 here instead of the planted 50.0.
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
            # Negative-row MEAN points away from the concept (so the fitted
            # direction is +base), but one TOKEN projects at +50 along it —
            # the population peak lives in a negative row.
            neg_acts = t.stack([-base, 50.0 * base], dim=0)
            return -base.clone(), neg_acts

        monkeypatch.setattr(
            capture_mod, "_capture_per_row_continuation", _stub_capture
        )
        rows = [
            _pos_row(42, "concept_rich", "POS a", "out", 1),
            _pos_row(42, "concept_avoiding", "NEG a", "out", 1),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_path = train_diffmean_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )[0]
        with safe_open(str(out_path), framework="pt") as handle:
            meta = handle.metadata() or {}
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
        tok = _StubNanochatTokenizer()

        n_prefix = len(tok.encode("p", prepend=tok.get_bos_token_id()))
        rows = [
            _pos_row(42, "concept_rich", "p", "o", n_prefix),
            _pos_row(42, "concept_rich", "p2", "o",
                     len(tok.encode("p2", prepend=tok.get_bos_token_id()))),
            _pos_row(42, "concept_avoiding", "n", "o",
                     len(tok.encode("n", prepend=tok.get_bos_token_id()))),
            _pos_row(42, "concept_avoiding", "n2", "o",
                     len(tok.encode("n2", prepend=tok.get_bos_token_id()))),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_diffmean_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        expected = test_paths.default_paths(cast(str, cfg.run_tag)).diffmean_direction(
            target_model="decoder_only", mode=SteeringMode.POSITIVE, concept_id=42
        )
        assert out_paths == [expected]
        assert expected.exists()
        payload = load_file(str(expected))
        assert "direction" in payload
        assert payload["direction"].shape == (TINY_N_EMBD,)

    def test_save_includes_expected_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "cache"))
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()
        tok = _StubNanochatTokenizer()

        n_prefix = len(tok.encode("p", prepend=tok.get_bos_token_id()))
        rows = [
            _pos_row(99, "concept_rich", "p", "out", n_prefix),
            _pos_row(99, "concept_rich", "q", "out",
                     len(tok.encode("q", prepend=tok.get_bos_token_id()))),
            _pos_row(99, "concept_avoiding", "r", "out",
                     len(tok.encode("r", prepend=tok.get_bos_token_id()))),
            _pos_row(99, "concept_avoiding", "s", "out",
                     len(tok.encode("s", prepend=tok.get_bos_token_id()))),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_diffmean_all(
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
            "concept_id",
            "mode",
            "target_model",
            "layer",
            "n_pos",
            "n_neg",
            MAX_ACTIVATION_METADATA_KEY,
        ):
            assert key in meta, f"Missing metadata key: {key}"
        assert meta["concept_id"] == "99"
        assert meta["mode"] == "positive"
        assert meta["target_model"] == "decoder_only"
        assert meta["layer"] == str(TINY_LAYER)
        assert meta["n_pos"] == "2"
        assert meta["n_neg"] == "2"
        assert "feature_contract_fingerprint" not in meta
        assert "source_corpus_sha256" not in meta
        assert "artifact_kind" not in meta

        legacy_direction = load_file(str(out_paths[0]))["direction"].clone()
        fingerprint = "a" * 64
        clustered_cfg = cfg.replace(run_tag="diffmean_test_clustered")
        clustered_paths = train_diffmean_all(
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
        assert clustered_meta["artifact_kind"] == "diffmean_direction_v1"
        assert (
            clustered_meta[MAX_ACTIVATION_METADATA_KEY]
            == meta[MAX_ACTIVATION_METADATA_KEY]
        )
        assert t.equal(
            load_file(str(clustered_path))["direction"],
            legacy_direction,
        )

        exact_bytes = clustered_path.read_bytes()
        assert train_diffmean_all(
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
            train_diffmean_all(
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
        tok = _StubNanochatTokenizer()

        def _n(text: str) -> int:
            return len(tok.encode(text, prepend=tok.get_bos_token_id()))

        rows = []
        for cid in (1, 2):
            for i in range(2):
                rows.append(
                    _pos_row(cid, "concept_rich", f"p{cid}{i}", "out",
                             _n(f"p{cid}{i}"))
                )
                rows.append(
                    _pos_row(cid, "concept_avoiding", f"n{cid}{i}", "out",
                             _n(f"n{cid}{i}"))
                )
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_diffmean_all(
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
# Tests: NaN guard
# ---------------------------------------------------------------------------


class TestNaNGuard:
    def test_nan_in_direction_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()

        nan_vec = t.full((TINY_N_EMBD,), float("nan"))
        ok_vec = t.zeros(TINY_N_EMBD)

        def _fake_capture(**kwargs):
            if "NAN" in kwargs["prefix_text"]:
                mean = nan_vec.clone()
            else:
                mean = ok_vec.clone()
            return mean, mean.unsqueeze(0)

        monkeypatch.setattr(
            capture_mod, "_capture_per_row_continuation", _fake_capture,
        )

        rows = [
            _pos_row(42, "concept_rich", "NAN_P", "x", 3),
            _pos_row(42, "concept_rich", "NAN_Q", "x", 3),
            _pos_row(42, "concept_avoiding", "NEG", "x", 3),
            _pos_row(42, "concept_avoiding", "NEG2", "x", 3),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        with pytest.raises(AxBenchBaseError, match="NaN"):
            train_diffmean_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )


# ---------------------------------------------------------------------------
# Tests: negative-mode schema (concept_rich vs concept_avoiding via content_class)
# ---------------------------------------------------------------------------


class TestNegativeModeSchema:
    def test_negative_mode_training_corpus_uses_content_class_split(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Negative-mode training corpus has the 19-column schema. The
        positive/negative split for DiffMean is:

          * ``concept_rich`` rows → positives
          * ``concept_avoiding`` rows → negatives

        This mirrors positive-mode's split exactly, because both schemas
        use ``content_class`` as the label column (see
        ``synth.positive_mode.TRAINING_SCHEMA:74`` and
        ``synth.negative_mode.NEG_TRAINING_SCHEMA:74``).
        """
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()
        tok = _StubNanochatTokenizer()

        def _n(text: str) -> int:
            return len(tok.encode(text, prepend=tok.get_bos_token_id()))

        rows = [
            _neg_row(42, "concept_rich", "RICH1", "x", _n("RICH1")),
            _neg_row(42, "concept_rich", "RICH2", "x", _n("RICH2")),
            _neg_row(42, "concept_avoiding", "AVOID1", "x", _n("AVOID1")),
            _neg_row(42, "concept_avoiding", "AVOID2", "x", _n("AVOID2")),
        ]
        corpus_path = tmp_path / "corpus_neg.parquet"
        _write_negative_corpus(corpus_path, rows)

        out_paths = train_diffmean_all(
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


# ---------------------------------------------------------------------------
# Tests: residual capture hook context manager
# ---------------------------------------------------------------------------


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
            train_diffmean_all(
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
        tok = _StubNanochatTokenizer()

        def _n(text: str) -> int:
            return len(tok.encode(text, prepend=tok.get_bos_token_id()))

        rows = [
            _pos_row(42, "concept_rich", "p1", "AAA", _n("p1")),
            _pos_row(42, "concept_rich", "p2", "BBB", _n("p2")),
            _pos_row(42, "concept_avoiding", "n1", "xyz", _n("n1")),
            _pos_row(42, "concept_avoiding", "n2", "qrs", _n("n2")),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        with pytest.raises(AxBenchBaseError, match="run_tag"):
            train_diffmean_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )

    def test_model_training_mode_restored_after_capture(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the target model enters in training mode, the capture helper
        must put it back into training mode after each forward pass."""
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()
        model.train()  # put it in training mode up front
        assert model.training

        tok = _StubNanochatTokenizer()

        def _n(text: str) -> int:
            return len(tok.encode(text, prepend=tok.get_bos_token_id()))

        rows = [
            _pos_row(42, "concept_rich", "p1", "AAA", _n("p1")),
            _pos_row(42, "concept_rich", "p2", "BBB", _n("p2")),
            _pos_row(42, "concept_avoiding", "n1", "xyz", _n("n1")),
            _pos_row(42, "concept_avoiding", "n2", "qrs", _n("n2")),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        train_diffmean_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        assert model.training, "Capture helper must restore training mode."

    def test_unknown_content_class_is_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rows with a ``content_class`` that is neither ``concept_rich``
        nor ``concept_avoiding`` (e.g. a future label) must be filtered
        out by ``_row_side`` and not counted as positives or negatives.
        """
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()
        tok = _StubNanochatTokenizer()

        def _n(text: str) -> int:
            return len(tok.encode(text, prepend=tok.get_bos_token_id()))

        # One "mystery" row that should be ignored, plus enough real rows
        # for the concept to train.
        mystery_row = _pos_row(42, "concept_rich", "m1", "AAA", _n("m1"))
        mystery_row["content_class"] = "unknown_future_label"

        rows = [
            mystery_row,
            _pos_row(42, "concept_rich", "p1", "AAA", _n("p1")),
            _pos_row(42, "concept_rich", "p2", "BBB", _n("p2")),
            _pos_row(42, "concept_avoiding", "n1", "xyz", _n("n1")),
            _pos_row(42, "concept_avoiding", "n2", "qrs", _n("n2")),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_diffmean_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        with safe_open(str(out_paths[0]), framework="pt") as f:
            meta = f.metadata() or {}
        # The mystery row was skipped — n_pos reflects only real positives.
        assert meta["n_pos"] == "2"
        assert meta["n_neg"] == "2"

    def test_per_concept_negatives_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Positive-mode negatives are now per-concept ``concept_avoiding``
        rows keyed to the same ``concept_id`` (no global pool, no
        ``concept_id == -1`` sentinel). They must be consumed; an avoiding
        row keyed to a different concept must be ignored.
        """
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()
        tok = _StubNanochatTokenizer()

        def _n(text: str) -> int:
            return len(tok.encode(text, prepend=tok.get_bos_token_id()))

        rows = [
            # Per-concept positives:
            _pos_row(42, "concept_rich", "p1", "AAA", _n("p1")),
            _pos_row(42, "concept_rich", "p2", "BBB", _n("p2")),
            # Per-concept negatives (same concept_id):
            _pos_row(42, "concept_avoiding", "n1", "xyz", _n("n1")),
            _pos_row(42, "concept_avoiding", "n2", "qrs", _n("n2")),
            # Different concept's avoiding row must NOT leak in:
            _pos_row(7, "concept_avoiding", "other", "abc", _n("other")),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_diffmean_all(
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

    def test_labeled_pair_boundary_mismatch_fails_loudly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stale approximate synth boundaries are rejected, never guessed."""
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()
        tok = _StubNanochatTokenizer()

        def _n(text: str) -> int:
            return len(tok.encode(text, prepend=tok.get_bos_token_id()))

        # Force a large mismatch against the exact combined-token boundary.
        rows = [
            _pos_row(42, "concept_rich", "p1", "AAA", _n("p1") + 10),
            _pos_row(42, "concept_rich", "p2", "BBB", _n("p2") + 10),
            _pos_row(42, "concept_avoiding", "n1", "xyz", _n("n1") + 10),
            _pos_row(42, "concept_avoiding", "n2", "qrs", _n("n2") + 10),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        with pytest.raises(AxBenchBaseError, match="boundary mismatch"):
            train_diffmean_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )


class TestResidualCaptureHook:
    def test_context_manager_captures_block_output(self) -> None:
        model = _make_tiny_gpt()
        idx = t.randint(0, TINY_VOCAB, (1, 4))

        with _ResidualCaptureHook(cast(nn.Module, model), layer=0) as cap:
            with t.no_grad():
                model(idx)
            out = cap.last
        assert out is not None
        assert out.shape == (1, 4, TINY_N_EMBD)

    def test_context_manager_removes_hook_on_exit(self) -> None:
        model = _make_tiny_gpt()
        idx = t.randint(0, TINY_VOCAB, (1, 3))

        with _ResidualCaptureHook(cast(nn.Module, model), layer=0):
            with t.no_grad():
                model(idx)

        # Count hooks on the layer-0 block AFTER exit; must be zero.
        blocks = cast(nn.ModuleList, model.transformer["h"])
        block = blocks[0]
        assert len(block._forward_hooks) == 0

    def test_context_manager_removes_hook_on_exception(self) -> None:
        model = _make_tiny_gpt()

        with pytest.raises(RuntimeError):
            with _ResidualCaptureHook(cast(nn.Module, model), layer=0):
                raise RuntimeError("boom")

        block = cast(nn.ModuleDict, model.transformer)["h"][0]
        assert len(block._forward_hooks) == 0

    def test_last_is_none_before_any_forward(self) -> None:
        model = _make_tiny_gpt()
        with _ResidualCaptureHook(cast(nn.Module, model), layer=0) as cap:
            assert cap.last is None

    def test_exit_without_enter_raises(self) -> None:
        """Direct-construction misuse should raise rather than silently no-op."""
        model = _make_tiny_gpt()
        hook = _ResidualCaptureHook(cast(nn.Module, model), layer=0)
        with pytest.raises(AxBenchBaseError, match="__enter__"):
            hook.__exit__(None, None, None)

    def test_layer_past_end_raises_named_error(self) -> None:
        """The regression this guard exists to prevent.

        Stage 3a died on ``IndexError: index 22 is out of range`` -- a message
        from ``nn.ModuleList`` that names neither the config key nor the model.
        22 was the *gemma-scope-2 SAE* layer copied into a field that indexes
        the *nanochat decoder*. The error must name the field and the real
        block count so the two-model mix-up is readable from the traceback.
        """
        model = _make_tiny_gpt()  # TINY_N_LAYER == 2, so 22 is far past the end
        hook = _ResidualCaptureHook(cast(nn.Module, model), layer=22)

        with pytest.raises(AxBenchBaseError) as excinfo:
            hook.__enter__()

        message = str(excinfo.value)
        # Each assertion pins a distinct fact the operator needs; a generic
        # "out of range" satisfies none of them.
        assert "diffmean_and_probe_layer" in message, "must name the config key"
        assert "22" in message, "must echo the offending value"
        assert f"{TINY_N_LAYER} blocks" in message, "must state the real depth"
        assert "SAE" in message, "must point at the two-model confusion"

    def test_layer_at_exact_end_raises(self) -> None:
        """Off-by-one boundary: ``n_layer`` itself is out of range."""
        model = _make_tiny_gpt()
        hook = _ResidualCaptureHook(cast(nn.Module, model), layer=TINY_N_LAYER)
        with pytest.raises(AxBenchBaseError, match="out of range"):
            hook.__enter__()

    def test_negative_layer_index_still_works(self) -> None:
        """The guard must not reject Python's valid negative indexing.

        ``-1`` is the last block, which is a legitimate way to ask for the
        final residual site; a naive ``layer >= len(blocks)`` check would
        silently keep working here, but a naive ``layer < 0`` one would break
        it. Capture through the hook and compare against the last block.
        """
        model = _make_tiny_gpt()
        idx = t.randint(0, TINY_VOCAB, (1, 4))

        with _ResidualCaptureHook(cast(nn.Module, model), layer=-1) as cap:
            with t.no_grad():
                model(idx)
            out = cap.last

        assert out is not None
        assert out.shape == (1, 4, TINY_N_EMBD)
        # The hook landed on the LAST block, not merely on some valid block.
        blocks = cast(nn.ModuleList, model.transformer["h"])
        assert len(blocks[TINY_N_LAYER - 1]._forward_hooks) == 0  # removed
        with _ResidualCaptureHook(cast(nn.Module, model), layer=-1):
            assert len(blocks[TINY_N_LAYER - 1]._forward_hooks) == 1
