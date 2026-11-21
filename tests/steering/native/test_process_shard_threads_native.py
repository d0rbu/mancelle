"""Regression guard: ``process_shard`` must thread ``native_model`` all the
way to the SAE-activation dispatch boundary (``_get_sae_activations_batched``).

Gap this closes
---------------
``tests/steering/native/test_process_shard_param.py`` only does static
string-introspection (``inspect.signature`` / ``inspect.getsource``). That
proves the *parameter exists* and that ``main`` builds the native model under
``Backend.native`` — but it would happily pass even if a refactor silently
dropped ``native_model=native_model`` at a ``process_shard`` -> dispatch call
site (e.g. forgot the kwarg, shadowed the name, or passed ``None``). An
adversarial pre-merge review flagged exactly this: introspection cannot catch
a dropped kwarg at the *call site*.

This test COMPLEMENTS (does not duplicate) that one: it runs the REAL
``steering.extract_features.process_shard`` end-to-end on a tiny CPU input
with a unique sentinel ``native_model`` and asserts — by object *identity*
(``is``) — that the exact sentinel reached the stubbed
``_get_sae_activations_batched``. It is non-vacuous: it fails (sentinel never
arrives / arrives as ``None``) if any ``native_model=`` kwarg is dropped on
the ``process_shard`` -> dispatch path.

Mocking surface (intentionally minimal so the REAL threading runs)
------------------------------------------------------------------
Only ``steering.extract_features._get_sae_activations_batched`` is patched —
that *is* the native-dispatch boundary. Everything between ``process_shard``'s
entry and that boundary (Phase-1 parquet read, tokenisation pipeline,
context-independent flatten / cd batch loop, and crucially every
``native_model=`` hop) is the real production code. The model/SAE/parquet are
the established tiny CPU fixtures already used by the integration suite (no
GPU, no network, no downloads).

Paths covered (parametrized)
----------------------------
- context-dependent (``main()``'s default): ``process_shard`` ->
  ``_get_sae_activations_batched`` direct call site.
- context-independent (``context_independent_features=True``):
  ``process_shard`` -> ``_run_context_independent_chunk`` ->
  ``_context_independent_forward_and_pool_batch`` ->
  ``_get_sae_activations_batched``.

Both are also crossed with ``pipeline_post`` True/False (the two distinct
in-function branches that each re-pass ``native_model``).
"""

from __future__ import annotations

from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from sae_lens import SAE
from transformer_lens import HookedTransformer

import steering.extract_features as ef
from steering.extract_features import PoolingMode, process_shard
from steering.native_collector import NativeCollector
from tests.steering.test_extract_features_integration import (
    TinyHookedTransformer,
    TinySAE,
)


def _make_parquet(path: str, texts: list[str]) -> None:
    """Minimal single-column text parquet (mirrors the integration helper)."""
    pq.write_table(pa.table({"text": texts}), path)


@pytest.mark.parametrize(
    "context_independent", [False, True],
    ids=["context-dependent", "context-independent"],
)
@pytest.mark.parametrize(
    "pipeline_post", [True, False],
    ids=["pipelined", "sequential"],
)
def test_process_shard_threads_native_model_to_dispatch(
    monkeypatch, tmp_path, context_independent: bool, pipeline_post: bool
):
    """The REAL process_shard must hand our exact sentinel to the dispatch
    boundary. Identity (``is``) assertion — not merely "stub was called" —
    so a dropped ``native_model=`` kwarg (passed as ``None`` / omitted)
    flips this test to FAIL.
    """
    captured: dict[str, object] = {}

    def fake_batched(
        model,
        sae,
        token_id_lists,
        *,
        layer,
        nnterp_model=None,
        native_model=None,
        hook_name=None,
        activation_type=ef.ActivationType.residual,
    ):
        # Record the native_model that actually reached the dispatch boundary.
        # First arrival wins so a later (correct) hop can't mask an earlier
        # dropped one; but in practice every batch must carry the sentinel.
        captured.setdefault("native_model", native_model)
        captured["called"] = True
        # Return a correctly-shaped dummy: list of [len_i, n_features] tensors,
        # exactly the contract of the real _get_sae_activations_batched
        # (steering/extract_features.py: `[sae_acts[i, : lengths[i]] ...]`).
        # Every token must fire >=1 strictly-positive feature so that no pooled
        # segment ends up empty (issue #522: a base shard with n_segments > 0
        # may not contain an empty segment — extract_sparse_gpu/save_features now
        # raise EmptyExtractionSegmentError on an all-zero pooled segment). We
        # set column 0 to 1.0; that keeps the dispatch-threading subject of this
        # test intact while producing a valid base shard.
        n_features = sae.cfg.d_sae
        out: list[t.Tensor] = []
        for ids in token_id_lists:
            acts = t.zeros(len(ids), n_features, dtype=t.float32)
            acts[:, 0] = 1.0
            out.append(acts)
        return out

    monkeypatch.setattr(ef, "_get_sae_activations_batched", fake_batched)

    # Unique, identity-comparable sentinel. Not a NativeCollector — the
    # threading path never inspects it (the only consumer is the stubbed
    # boundary), so a bare marker proves the wiring without any GPU/HF cost.
    native_sentinel = object()

    model = TinyHookedTransformer(d_model=64, vocab_size=1000, device="cpu")
    sae = TinySAE(d_model=64, n_features=256, device="cpu")

    parquet_path = str(tmp_path / "shard.parquet")
    _make_parquet(
        parquet_path,
        [
            "hello world this is a document",
            "another document for the regression test",
        ],
    )

    # `cast` is a pure type annotation (identity no-op at runtime), so the
    # SAME `native_sentinel` object flows in — the `is` check below stays
    # exact. Tiny model/SAE stand-ins are cast to the production param types
    # exactly as the established integration suite does (keeps `ty` clean).
    result = process_shard(
        parquet_path,
        cast(HookedTransformer, model),
        cast(SAE, sae),
        layer=8,
        output_dir=str(tmp_path),
        context_window=128,
        pooling=PoolingMode.max_pool,
        segment_size=4,
        gpu_batch_size=4,
        cpu_batch_size=16,
        # the object whose identity we trace to the dispatch boundary
        native_model=cast(NativeCollector, native_sentinel),
        pipeline_post=pipeline_post,
        context_independent_features=context_independent,
    )

    # process_shard ran the full real threading path and completed.
    assert result.n_docs == 2, (
        "process_shard did not process both fixture docs end-to-end; "
        "the threading path under test never fully ran"
    )
    assert captured.get("called") is True, (
        "process_shard never reached _get_sae_activations_batched — the "
        "dispatch boundary was not exercised, so threading is unverified"
    )

    # THE load-bearing assertion: the EXACT sentinel object — by identity —
    # threaded native_model -> process_shard -> (ci chunk / cd loop) ->
    # _get_sae_activations_batched. Fails if any native_model= kwarg on that
    # path is dropped/None/shadowed.
    assert captured["native_model"] is native_sentinel, (
        "process_shard must thread the exact native_model object through to "
        "_get_sae_activations_batched; got "
        f"{captured['native_model']!r} (is-not sentinel) — a native_model= "
        "kwarg was dropped on the process_shard -> dispatch path "
        f"(context_independent={context_independent}, "
        f"pipeline_post={pipeline_post})"
    )
