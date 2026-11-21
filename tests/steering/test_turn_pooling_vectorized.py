"""Parity tests for vectorized turn-delimited segment pooling."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch as t
from hypothesis import given, settings
from hypothesis import strategies as st

import steering.extract_features as extract_features_module
from errors import BosOnlySegmentError
from steering.extract_features import (
    extract_document_features,
    extract_sparse_gpu,
    pool_segments_batched_gpu,
)
from steering.feature_paths import PoolingMode
from steering.segmentation import SegmentMode, segment_turn_delimited
from tests.steering.test_extract_features_integration import TinyHookedTransformer, TinySAE

TURN_MODE = SegmentMode.turn_delimited_fixed_tokens

_MEAN_TOLERANCES = {
    t.float32: (1e-5, 1e-6),
    t.float16: (1e-3, 1e-3),
    t.bfloat16: (1e-2, 1e-2),
}


def _legacy_nonfixed_pool(
    sae_acts: t.Tensor,
    segments: list[tuple[int, int]],
    pooling: PoolingMode,
    keep_mask: t.Tensor | None,
) -> t.Tensor:
    """Literal output oracle for the pre-#1619 non-fixed pooling loops."""
    n_segs = len(segments)
    n_feat = sae_acts.shape[1]
    if pooling == PoolingMode.max_pool:
        assert sae_acts.min().item() >= 0
        result = t.empty(n_segs, n_feat, device=sae_acts.device, dtype=sae_acts.dtype)
        for i, (start, end) in enumerate(segments):
            chunk = sae_acts[start:end]
            if keep_mask is not None:
                mask = keep_mask[start:end].to(t.bool)
                if not bool(mask.any()):
                    raise BosOnlySegmentError
                chunk = chunk.masked_fill(~mask.unsqueeze(-1), 0.0)
            result[i] = chunk.max(dim=0).values
        return result

    if pooling == PoolingMode.mean_pool:
        result = t.empty(n_segs, n_feat, device=sae_acts.device, dtype=sae_acts.dtype)
        for i, (start, end) in enumerate(segments):
            chunk = sae_acts[start:end]
            if keep_mask is None:
                result[i] = chunk.mean(dim=0)
            else:
                mask = keep_mask[start:end].to(t.bool)
                if not bool(mask.any()):
                    raise BosOnlySegmentError
                mask_float = mask.unsqueeze(-1).to(chunk.dtype)
                result[i] = (chunk * mask_float).sum(dim=0) / mask.sum().to(chunk.dtype)
        return result

    raise AssertionError(f"Unsupported oracle pooling mode: {pooling}")


def _contiguous_case(
    lengths: list[int],
    base: int,
    suffix: int,
    seed: int,
    dtype: t.dtype,
    masked: bool,
) -> tuple[t.Tensor, list[tuple[int, int]], t.Tensor | None]:
    segments: list[tuple[int, int]] = []
    cursor = base
    for length in lengths:
        segments.append((cursor, cursor + length))
        cursor += length

    generator = t.Generator().manual_seed(seed)
    # Quantization creates exact ties and all-zero feature columns in some
    # segments, which are important max-pool cases for sparse SAE activations.
    acts = t.randint(
        0,
        8,
        (cursor + suffix, 11),
        generator=generator,
        dtype=t.int64,
    ).to(dtype)
    acts[:, 0] = 0

    if not masked:
        return acts, segments, None

    keep_mask = t.rand(cursor + suffix, generator=generator) > 0.35
    # Every segment must keep at least one token, but which one is free. Pin
    # the LAST row rather than the first so a segment's first row can be
    # masked out — the max route reads segment-relative offsets, so a masked
    # leading row is a case worth generating. Length-1 segments keep their
    # only row.
    for _, end in segments:
        keep_mask[end - 1] = True
    # Exercise the masked formula even in the rare all-True random draw.
    for start, end in segments:
        if end - start > 1:
            keep_mask[start] = False
            break
    return acts, segments, keep_mask


@settings(max_examples=150, deadline=None)
@given(
    lengths=st.lists(st.integers(min_value=1, max_value=32), min_size=2, max_size=12),
    base=st.integers(min_value=0, max_value=5),
    suffix=st.integers(min_value=0, max_value=5),
    seed=st.integers(min_value=0, max_value=1_000_000),
    dtype=st.sampled_from([t.float32, t.float16, t.bfloat16]),
    masked=st.booleans(),
)
def test_vectorized_max_is_exactly_legacy(
    lengths: list[int],
    base: int,
    suffix: int,
    seed: int,
    dtype: t.dtype,
    masked: bool,
) -> None:
    acts, segments, keep_mask = _contiguous_case(
        lengths, base, suffix, seed, dtype, masked
    )
    expected = _legacy_nonfixed_pool(
        acts, segments, PoolingMode.max_pool, keep_mask
    )
    actual = pool_segments_batched_gpu(
        acts,
        segments,
        PoolingMode.max_pool,
        TURN_MODE,
        keep_mask=keep_mask,
    )
    assert t.equal(actual, expected)


@settings(max_examples=150, deadline=None)
@given(
    lengths=st.lists(st.integers(min_value=1, max_value=32), min_size=2, max_size=12),
    base=st.integers(min_value=0, max_value=5),
    suffix=st.integers(min_value=0, max_value=5),
    seed=st.integers(min_value=0, max_value=1_000_000),
    dtype=st.sampled_from([t.float32, t.float16, t.bfloat16]),
    masked=st.booleans(),
)
def test_vectorized_mean_matches_legacy_within_reduction_tolerance(
    lengths: list[int],
    base: int,
    suffix: int,
    seed: int,
    dtype: t.dtype,
    masked: bool,
) -> None:
    acts, segments, keep_mask = _contiguous_case(
        lengths, base, suffix, seed, dtype, masked
    )
    # Non-integral inputs ensure the two reduction orders are actually tested.
    acts = acts / 7
    expected = _legacy_nonfixed_pool(
        acts, segments, PoolingMode.mean_pool, keep_mask
    )
    actual = pool_segments_batched_gpu(
        acts,
        segments,
        PoolingMode.mean_pool,
        TURN_MODE,
        keep_mask=keep_mask,
    )
    if dtype in (t.float16, t.bfloat16):
        assert t.equal(actual.view(t.int16), expected.view(t.int16))
    else:
        rtol, atol = _MEAN_TOLERANCES[dtype]
        t.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [t.float32, t.float16, t.bfloat16])
@pytest.mark.parametrize("masked", [False, True])
def test_vectorized_mean_matches_legacy_at_production_segment_scale(
    dtype: t.dtype,
    masked: bool,
) -> None:
    acts, segments, keep_mask = _contiguous_case(
        [128, 1, 127, 256, 2048],
        base=3,
        suffix=2,
        seed=1619,
        dtype=dtype,
        masked=masked,
    )
    acts = acts / 7
    expected = _legacy_nonfixed_pool(
        acts, segments, PoolingMode.mean_pool, keep_mask
    )
    actual = pool_segments_batched_gpu(
        acts,
        segments,
        PoolingMode.mean_pool,
        TURN_MODE,
        keep_mask=keep_mask,
    )
    if dtype in (t.float16, t.bfloat16):
        assert t.equal(actual.view(t.int16), expected.view(t.int16))
    else:
        rtol, atol = _MEAN_TOLERANCES[dtype]
        t.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "segments",
    [
        [(0, 3), (2, 6)],  # overlap
        [(0, 2), (3, 6)],  # gap
        [(3, 6), (0, 3)],  # reordered
        [(1, 5)],  # a single reduction is cheaper without segmented setup
    ],
)
@pytest.mark.parametrize("pooling", [PoolingMode.max_pool, PoolingMode.mean_pool])
def test_arbitrary_layouts_retain_legacy_loop(
    monkeypatch: pytest.MonkeyPatch,
    segments: list[tuple[int, int]],
    pooling: PoolingMode,
) -> None:
    acts = t.arange(1, 29, dtype=t.float32).reshape(7, 4)
    keep_mask = t.tensor([True, False, True, True, False, True, True])
    expected = _legacy_nonfixed_pool(acts, segments, pooling, keep_mask)

    def fail_if_vectorized(*args, **kwargs):
        raise AssertionError("arbitrary layout entered the vectorized path")

    monkeypatch.setattr(t, "segment_reduce", fail_if_vectorized)
    actual = pool_segments_batched_gpu(
        acts, segments, pooling, TURN_MODE, keep_mask=keep_mask
    )
    assert t.equal(actual, expected)


@pytest.mark.parametrize("dtype", [t.int64, t.int32, t.uint8, t.bool])
@pytest.mark.parametrize("pooling", [PoolingMode.max_pool, PoolingMode.mean_pool])
def test_nonfloating_contiguous_layout_retains_legacy_loop(
    monkeypatch: pytest.MonkeyPatch,
    dtype: t.dtype,
    pooling: PoolingMode,
) -> None:
    acts = t.tensor(
        [[0, 1], [2, 3], [4, 5], [6, 7]],
        dtype=dtype,
    )
    segments = [(0, 2), (2, 4)]

    def fail_if_vectorized(*args, **kwargs):
        raise AssertionError("non-floating input entered the vectorized path")

    monkeypatch.setattr(t, "segment_reduce", fail_if_vectorized)
    if pooling == PoolingMode.max_pool:
        expected = _legacy_nonfixed_pool(
            acts, segments, pooling, keep_mask=None
        )
        actual = pool_segments_batched_gpu(
            acts, segments, pooling, TURN_MODE
        )
        assert t.equal(actual, expected)
    else:
        with pytest.raises(RuntimeError) as legacy_error:
            _legacy_nonfixed_pool(acts, segments, pooling, keep_mask=None)
        with pytest.raises(RuntimeError) as actual_error:
            pool_segments_batched_gpu(acts, segments, pooling, TURN_MODE)
        assert str(actual_error.value) == str(legacy_error.value)


@pytest.mark.parametrize(
    "pooling", [PoolingMode.max_pool, PoolingMode.mean_pool, PoolingMode.last_token]
)
def test_fixed_path_does_not_enter_vectorized_turn_route(
    monkeypatch: pytest.MonkeyPatch,
    pooling: PoolingMode,
) -> None:
    acts = t.arange(24, dtype=t.float32).reshape(8, 3)
    segments = [(0, 3), (3, 6), (6, 8)]

    def fail_if_vectorized(*args, **kwargs):
        raise AssertionError("fixed-token pooling entered the turn vectorization")

    # Independent oracle, not a second call to the function under test: a
    # route-only assertion would pass on an implementation that took the right
    # path and returned the wrong values.
    if pooling == PoolingMode.last_token:
        expected = t.stack([acts[end - 1] for _, end in segments])
    elif pooling == PoolingMode.max_pool:
        expected = t.stack([acts[s:e].max(dim=0).values for s, e in segments])
    else:
        expected = t.stack([acts[s:e].mean(dim=0) for s, e in segments])

    monkeypatch.setattr(t, "segment_reduce", fail_if_vectorized)
    actual = pool_segments_batched_gpu(
        acts, segments, pooling, SegmentMode.fixed_tokens
    )
    assert t.equal(actual, expected)


@pytest.mark.parametrize("pooling", [PoolingMode.max_pool, PoolingMode.mean_pool])
def test_uniform_turn_segments_match_fixed_path(
    pooling: PoolingMode,
) -> None:
    generator = t.Generator().manual_seed(1619)
    acts = t.rand(64, 13, generator=generator)
    segments = [(start, start + 8) for start in range(0, 64, 8)]
    fixed = pool_segments_batched_gpu(
        acts, segments, pooling, SegmentMode.fixed_tokens
    )
    turn = pool_segments_batched_gpu(acts, segments, pooling, TURN_MODE)
    if pooling == PoolingMode.max_pool:
        assert t.equal(turn, fixed)
    else:
        t.testing.assert_close(turn, fixed, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [t.float32, t.float16, t.bfloat16])
@pytest.mark.parametrize("pooling", list(PoolingMode))
def test_turn_all_true_mask_is_exactly_no_mask(
    pooling: PoolingMode, dtype: t.dtype
) -> None:
    # float32 alone cannot detect a missing all-True normalization: the masked
    # and unmasked vectorized mean formulas coincide bitwise there. The low
    # precision dtypes use different rounding points, so they do.
    generator = t.Generator().manual_seed(1619)
    acts = t.rand(23, 7, generator=generator).to(dtype)
    segments = [(2, 5), (5, 13), (13, 18), (18, 21)]
    unmasked = pool_segments_batched_gpu(acts, segments, pooling, TURN_MODE)
    all_true = pool_segments_batched_gpu(
        acts,
        segments,
        pooling,
        TURN_MODE,
        keep_mask=t.ones(acts.shape[0], dtype=t.bool),
    )
    assert t.equal(all_true, unmasked)


@pytest.mark.parametrize("pooling", [PoolingMode.max_pool, PoolingMode.mean_pool])
def test_real_turn_segmentation_geometry_takes_vectorized_route(
    monkeypatch: pytest.MonkeyPatch,
    pooling: PoolingMode,
) -> None:
    segments = segment_turn_delimited([0, 3, 12, 19], n=25, segment_size=5)
    assert segments == [(0, 3), (3, 8), (8, 12), (12, 17), (17, 19), (19, 24), (24, 25)]
    acts = t.arange(1, 101, dtype=t.float32).reshape(25, 4)
    original = t.segment_reduce
    calls = 0

    def counted_segment_reduce(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(t, "segment_reduce", counted_segment_reduce)
    actual = pool_segments_batched_gpu(
        acts, segments, pooling, TURN_MODE
    )
    expected = _legacy_nonfixed_pool(
        acts, segments, pooling, keep_mask=None
    )
    assert calls == 1
    if pooling == PoolingMode.max_pool:
        assert t.equal(actual, expected)
    else:
        t.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("pooling", [PoolingMode.max_pool, PoolingMode.mean_pool])
def test_vectorized_empty_mask_error_matches_legacy_message(pooling: PoolingMode) -> None:
    acts = t.ones(6, 2)
    segments = [(0, 2), (2, 5), (5, 6)]
    # TWO empty segments: pins that the LOWEST failing index is reported, as
    # the per-segment loop does by construction. With a single empty segment
    # this only pins the message format.
    keep_mask = t.tensor([True, False, False, False, False, False])
    expected = (
        "Segment 1 [2,5) is entirely <bos>; "
        "ignore_bos_features removed every poolable token."
    )
    with pytest.raises(BosOnlySegmentError) as exc_info:
        pool_segments_batched_gpu(
            acts, segments, pooling, TURN_MODE, keep_mask=keep_mask
        )
    assert str(exc_info.value) == expected


@pytest.mark.parametrize(
    ("acts", "keep_mask"),
    [
        (t.tensor([[-0.0], [0.0], [0.0], [-0.0]]), None),
        (
            t.tensor([[-0.0], [1.0], [0.0], [-0.0]]),
            t.tensor([True, False, False, True]),
        ),
    ],
)
def test_vectorized_max_preserves_legacy_signed_zero_results(
    acts: t.Tensor,
    keep_mask: t.Tensor | None,
) -> None:
    segments = [(0, 2), (2, 4)]
    expected = _legacy_nonfixed_pool(
        acts, segments, PoolingMode.max_pool, keep_mask
    )
    actual = pool_segments_batched_gpu(
        acts,
        segments,
        PoolingMode.max_pool,
        TURN_MODE,
        keep_mask=keep_mask,
    )
    assert t.equal(actual.view(t.int32), expected.view(t.int32))


def test_vectorized_max_preserves_sparse_projection_exactly() -> None:
    acts, segments, keep_mask = _contiguous_case(
        [3, 11, 2, 7],
        base=2,
        suffix=3,
        seed=1619,
        dtype=t.float32,
        masked=True,
    )
    expected_pool = _legacy_nonfixed_pool(
        acts, segments, PoolingMode.max_pool, keep_mask
    )
    actual_pool = pool_segments_batched_gpu(
        acts,
        segments,
        PoolingMode.max_pool,
        TURN_MODE,
        keep_mask=keep_mask,
    )
    expected_ids, expected_mags = extract_sparse_gpu(expected_pool)
    actual_ids, actual_mags = extract_sparse_gpu(actual_pool)
    assert all(
        t.equal(actual, expected)
        for actual, expected in zip(actual_ids, expected_ids, strict=True)
    )
    assert all(
        t.equal(actual, expected)
        for actual, expected in zip(actual_mags, expected_mags, strict=True)
    )


@pytest.mark.parametrize("pooling", [PoolingMode.max_pool, PoolingMode.mean_pool])
def test_document_extraction_matches_legacy(
    monkeypatch: pytest.MonkeyPatch,
    pooling: PoolingMode,
) -> None:
    t.manual_seed(1619)
    model, sae = TinyHookedTransformer(), TinySAE()

    def run_extraction():
        return extract_document_features(
            model,
            sae,
            "abcdefghijklmnopqrstuvwxyz",
            pooling=pooling,
            segment_mode=TURN_MODE,
            segment_size=5,
            ignore_bos_features=True,
            turn_char_starts=[0, 3, 12, 19],
        )

    actual = run_extraction()

    def legacy_pool(
        sae_acts: t.Tensor,
        segments: list[tuple[int, int]],
        pooling: PoolingMode,
        segment_mode: SegmentMode = SegmentMode.fixed_tokens,
        keep_mask: t.Tensor | None = None,
    ) -> t.Tensor:
        assert segment_mode == TURN_MODE
        return _legacy_nonfixed_pool(sae_acts, segments, pooling, keep_mask)

    monkeypatch.setattr(
        extract_features_module,
        "pool_segments_batched_gpu",
        legacy_pool,
    )
    expected = run_extraction()

    assert all(
        t.equal(actual_ids, expected_ids)
        for actual_ids, expected_ids in zip(actual[0], expected[0], strict=True)
    )
    if pooling == PoolingMode.max_pool:
        assert all(
            t.equal(actual_mags, expected_mags)
            for actual_mags, expected_mags in zip(actual[1], expected[1], strict=True)
        )
    else:
        for actual_mags, expected_mags in zip(actual[1], expected[1], strict=True):
            t.testing.assert_close(actual_mags, expected_mags, rtol=1e-5, atol=1e-6)
    assert actual[2:] == expected[2:]


def test_vectorized_max_retains_global_nonnegative_assertion() -> None:
    acts = t.ones(8, 2)
    acts[0, 0] = -1
    segments = [(2, 5), (5, 8)]
    with pytest.raises(AssertionError, match="assumes non-negative input"):
        pool_segments_batched_gpu(
            acts, segments, PoolingMode.max_pool, TURN_MODE
        )


# Deliberately NOT marked ``slow``: the repo's default addopts deselect that
# marker, which would make the only CUDA parity gate dead by default. The
# skipif already keeps it off CPU-only machines, and it runs in ~1.6s.
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [t.float32, t.float16, t.bfloat16])
def test_cuda_vectorized_pooling_matches_legacy(dtype: t.dtype) -> None:
    bits_dtype = t.int32 if dtype == t.float32 else t.int16
    for seed in range(25):
        acts, segments, keep_mask = _contiguous_case(
            [1, 7, 3, 128, 2, 127, 2048, 5],
            base=3,
            suffix=2,
            seed=seed,
            dtype=dtype,
            masked=bool(seed % 2),
        )
        acts = acts.cuda()
        keep_mask = keep_mask.cuda() if keep_mask is not None else None

        expected_max = _legacy_nonfixed_pool(
            acts, segments, PoolingMode.max_pool, keep_mask
        )
        actual_max = pool_segments_batched_gpu(
            acts,
            segments,
            PoolingMode.max_pool,
            TURN_MODE,
            keep_mask=keep_mask,
        )
        assert t.equal(actual_max.view(bits_dtype), expected_max.view(bits_dtype))

        mean_acts = acts / 7
        expected_mean = _legacy_nonfixed_pool(
            mean_acts, segments, PoolingMode.mean_pool, keep_mask
        )
        actual_mean = pool_segments_batched_gpu(
            mean_acts,
            segments,
            PoolingMode.mean_pool,
            TURN_MODE,
            keep_mask=keep_mask,
        )
        if dtype in (t.float16, t.bfloat16):
            mean_ulps = (
                actual_mean.view(bits_dtype).to(t.int32)
                - expected_mean.view(bits_dtype).to(t.int32)
            ).abs()
            assert int(mean_ulps.max().item()) <= 1, (
                f"{dtype} mean differed from the legacy loop by more than one ULP"
            )
        else:
            rtol, atol = _MEAN_TOLERANCES[dtype]
            t.testing.assert_close(actual_mean, expected_mean, rtol=rtol, atol=atol)

    signed_zero_segments = [(0, 2), (2, 4)]
    signed_zero_cases = [
        (
            t.tensor(
                [[-0.0], [0.0], [0.0], [-0.0]], dtype=dtype, device="cuda"
            ),
            None,
        ),
        (
            t.tensor(
                [[-0.0], [1.0], [0.0], [-0.0]], dtype=dtype, device="cuda"
            ),
            t.tensor([True, False, False, True], device="cuda"),
        ),
    ]
    for signed_zero_acts, signed_zero_mask in signed_zero_cases:
        expected_zero = _legacy_nonfixed_pool(
            signed_zero_acts,
            signed_zero_segments,
            PoolingMode.max_pool,
            signed_zero_mask,
        )
        actual_zero = pool_segments_batched_gpu(
            signed_zero_acts,
            signed_zero_segments,
            PoolingMode.max_pool,
            TURN_MODE,
            keep_mask=signed_zero_mask,
        )
        assert t.equal(actual_zero.view(bits_dtype), expected_zero.view(bits_dtype))


# ``docs/performance/profiling.md`` documents ``python -O`` as a supported
# throughput mode, which strips the non-negative-input assertion guarding this
# route. An earlier draft repaired max ties with
# ``t.where(pooled == 0, first_values, pooled)``; with the guard stripped that
# overwrote a legitimate zero maximum on mixed-sign input, silently disagreeing
# with the per-segment loop. This must run in a real subprocess: ``-O`` is fixed
# at interpreter start and cannot be faked in-process.
_DASH_O_PARITY_PROBE = """
import sys
import torch as t

from steering.extract_features import pool_segments_batched_gpu
from steering.feature_paths import PoolingMode
from steering.segmentation import SegmentMode

assert not __debug__, "probe must run under -O"

acts = t.tensor([[-5.0, -1.0], [0.0, -2.0], [-7.0, -3.0], [-0.5, -4.0]])
segments = [(0, 2), (2, 4)]
expected = t.stack([acts[s:e].max(dim=0).values for s, e in segments])
actual = pool_segments_batched_gpu(
    acts, segments, PoolingMode.max_pool,
    SegmentMode.turn_delimited_fixed_tokens,
)
if not t.equal(actual, expected):
    print(f"MISMATCH actual={actual.tolist()} expected={expected.tolist()}")
    sys.exit(1)
print("OK")
"""


def test_vectorized_max_matches_loop_with_assertions_stripped(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "dash_o_probe.py"
    probe.write_text(_DASH_O_PARITY_PROBE)
    repo_root = Path(extract_features_module.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo_root), "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-O", str(probe)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
        check=False,
    )
    assert result.returncode == 0, (
        f"vectorized max diverged from the per-segment loop under `python -O`:\n"
        f"{result.stdout}\n{result.stderr}"
    )
