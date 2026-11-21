"""Integration pin: a mid-run accelerated-backend failure must FAIL FAST out
of ``steering.extract_features.process_shard`` — it is never swallowed
mid-run, for BOTH the pipelined and the sequential in-function code paths.

Step-1 investigation findings (file:line evidence, commit f6a47725)
-------------------------------------------------------------------
* ``process_shard`` body spans steering/extract_features.py:2469-3185.
  An ``awk`` scan of that range for ``try:`` / ``except`` finds ZERO
  exception handlers (the only textual hit is a comment, not code). So
  ``process_shard`` itself contains NO ``try/except`` — nothing there can
  swallow an ``AcceleratedBackendError``.

* The GPU forward (``_get_sae_activations_batched``) is called DIRECTLY in
  the main thread, NOT in a worker, on BOTH branches:
    - pipelined branch  : steering/extract_features.py:2768
      (inside ``with _region("model_forward_batched")``)
    - sequential branch : steering/extract_features.py:2955
  (the context-independent sub-path reaches the same boundary via
  ``_run_context_independent_chunk``.)

* The only ``ThreadPoolExecutor``/``ProcessPoolExecutor`` work in
  ``process_shard`` is the CPU tokenize/segment pipeline, NOT the GPU
  forward:
    - pipelined  : ``_tokenize_and_segment_chunk[_worker]`` submitted at
      steering/extract_features.py:2673 / 2680; drained via
      ``pending_future.result()`` at :2698.
    - sequential : ``_tokenize_chunk`` submitted at :2869 / :2883;
      drained via ``pending_future.result()`` at :2880.
  Both drains use ``concurrent.futures`` ``Future.result()``, which
  RE-RAISES a worker exception by default (no swallow). The staged input
  prefetcher (``StagedPrefetcher``) likewise only prefetches CPU
  parquet/tokenize work and re-raises via ``future.result()`` in its
  ``get()`` — it is outside ``process_shard`` and not on the
  GPU-forward path.

* The two ``continue`` statements adjacent to the activation call
  (steering/extract_features.py:2809 and :2996) are ``for ... in zip(...)``
  loop filters guarded by ``if len(segments) == 0:`` — a DATA skip, NOT an
  exception handler.

* Site 1a (``_get_sae_activations_batched``,
  steering/extract_features.py:1264-1276): when ``native_model is not
  None`` it calls ``_get_activations_native(...)`` and, on
  ``ExtractionError``, HARD-RAISES ``AcceleratedBackendError`` via
  ``_accelerated_backend_error("native", "forward", exc)``. That
  ``try/except`` lives INSIDE ``_get_sae_activations_batched``, not inside
  ``process_shard``.

CONCLUSION: ``.result()`` already re-raises and ``process_shard`` has no
swallowing ``except`` around the (main-thread) activation call, so NO
production change is needed. This module is therefore a PURE PINNING test:
it drives the REAL ``process_shard`` with an injected accelerated failure
and asserts ``AcceleratedBackendError`` escapes for every
(pipelined|sequential) x (context-dependent|context-independent) variant.

Construction approach
---------------------
Mirrors ``tests/steering/native/test_process_shard_threads_native.py``
EXACTLY (same tiny CPU ``TinyHookedTransformer`` / ``TinySAE`` fixtures
from the integration suite, same minimal single-column parquet helper, same
two-axis parametrization ``pipeline_post`` x ``context_independent``,
identical ``process_shard(...)`` call kwargs). The ONLY deviation: instead
of stubbing ``_get_sae_activations_batched`` (which would bypass site 1a),
we monkeypatch ``_get_activations_native`` to raise
``errors.ExtractionError("injected")`` and pass a sentinel
``native_model=object()`` so the REAL ``_get_sae_activations_batched``
runs site 1a, hits the injected ``ExtractionError``, converts it to
``AcceleratedBackendError`` — and we assert that propagates OUT of
``process_shard``.
"""

from __future__ import annotations

from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sae_lens import SAE
from transformer_lens import HookedTransformer

import steering.extract_features as ef
from errors import AcceleratedBackendError, ExtractionError
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
def test_accelerated_failure_propagates_out_of_process_shard(
    monkeypatch, tmp_path, context_independent: bool, pipeline_post: bool
):
    """A mid-run accelerated ``ExtractionError`` must surface as an
    ``AcceleratedBackendError`` that ESCAPES ``process_shard`` — never
    swallowed mid-run — on the pipelined AND the sequential branch (each
    also crossed with the context-(in)dependent forward path).
    """
    call_count = {"native": 0}

    def boom_native(
        native_model,
        tokens,
        layer,
        activation_type=ef.ActivationType.residual,
        hook_name=None,
    ):
        # The accelerated (native) backend fails mid-run. Site 1a wraps this
        # ExtractionError into a hard AcceleratedBackendError (no silent TL
        # fallback) — this test pins that it then propagates OUT.
        call_count["native"] += 1
        raise ExtractionError("injected")

    monkeypatch.setattr(ef, "_get_activations_native", boom_native)

    # Sentinel native_model: site 1a only checks ``native_model is not None``
    # before calling the (patched-to-raise) _get_activations_native, so a
    # bare marker is enough — no GPU / HF NativeCollector needed.
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

    # Identical call shape to test_process_shard_threads_native.py. ``cast``
    # is a runtime no-op (keeps ``ty`` clean exactly as the integration
    # suite / reference test do).
    with pytest.raises(AcceleratedBackendError) as excinfo:
        process_shard(
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
            native_model=cast(NativeCollector, native_sentinel),
            pipeline_post=pipeline_post,
            context_independent_features=context_independent,
        )

    # The accelerated backend was genuinely attempted mid-run (the raise was
    # exercised, not short-circuited before reaching site 1a).
    assert call_count["native"] >= 1, (
        "process_shard never reached the native accelerated backend — the "
        "injected failure was not exercised, so propagation is unverified "
        f"(context_independent={context_independent}, "
        f"pipeline_post={pipeline_post})"
    )

    # The escaping error is the hard fail-fast error, and it chains the
    # original accelerated ExtractionError (site 1a's ``raise ... from exc``).
    assert isinstance(excinfo.value.__cause__, ExtractionError), (
        "AcceleratedBackendError must chain the underlying accelerated "
        f"ExtractionError as __cause__; got {excinfo.value.__cause__!r}"
    )
