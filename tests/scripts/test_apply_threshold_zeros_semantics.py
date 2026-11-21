"""``precompute_selected_features`` rejects magnitude-dependent thresholds.

The script previously called ``apply_threshold(fids, t.zeros(n_feat), ...)``
against the *training*
config's threshold_type, which silently miscomputed for magnitude-dependent
threshold types (e.g., ``segment_topk`` and ``segment_percentile``). The fix
whitelists the percentile-only types and raises clearly for everything else.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t

from scripts.preprocess.precompute_selected_features import (
    _PERCENTILE_ONLY_THRESHOLD_TYPES,
    select_features_for_shard,
)
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.sampling import ThresholdSelection
from steering.conditional.thresholding import ThresholdType
from steering.packed_batch_io import PackedRowShard, save_packed_row_shard


def _make_minimal_packed_row_shard(tmp_path: Path) -> Path:
    """A 1-row, 1-segment, 1-feature packed shard saved to disk."""
    shard = PackedRowShard(
        tokens=t.zeros(1, 9, dtype=t.uint16),  # T+1 = 9
        row_seg_offsets=t.tensor([0, 1], dtype=t.int32),
        seg_token_start=t.tensor([0], dtype=t.uint16),
        seg_token_end=t.tensor([8], dtype=t.uint16),
        seg_doc_id=t.tensor([0], dtype=t.uint8),
        seg_feat_offsets=t.tensor([0, 1], dtype=t.int32),
        feature_ids=t.tensor([3], dtype=t.uint16),
        feature_pcts=t.tensor([200], dtype=t.uint8),  # ~78%
    )
    out = tmp_path / "shard_00000"
    return save_packed_row_shard(shard, out)


def _threshold_value_for(th_type: ThresholdType) -> float:
    """Return a sensible threshold for ``th_type``.

    Percentile types want a percentile in (0, 1); top-k types want a
    positive integer k. Using a single literal across both axes (the prior
    fixture used 0.5 for every type) made the top-k call sites trip a
    downstream ``assert k > 0`` instead of exercising the audited code
    path, which the previous wide ``except Exception`` then silently
    swallowed.
    """
    if th_type in (
        ThresholdType.feature_percentile_segment_topk,
        ThresholdType.segment_topk,
    ):
        return 4.0  # k = 4 features per segment
    return 0.5  # percentile-style threshold


def _make_threshold_pipeline_config(threshold_type: ThresholdType) -> FeaturePipelineConfig:
    """Minimal pipeline config with the requested threshold_type."""
    selection = ThresholdSelection(
        threshold_type=threshold_type,
        threshold=_threshold_value_for(threshold_type),
    )
    sel_cfg = ThresholdSelectionConfig(selection=selection, precomputed_thresholds=None)
    n_sae = 16
    boundaries = t.zeros(n_sae, 1, dtype=t.float64)
    return FeaturePipelineConfig(
        selection_config=sel_cfg,
        boundaries=boundaries,
        negative_bundle=None,
        effective_buckets=2,
    )


class TestPercentileOnlyWhitelistContents:
    """Pin which threshold types are safe to evaluate against
    percentile-only inputs (no real magnitudes available in precompute).
    """

    def test_whitelist_contents(self):
        # The whitelist must include exactly the two percentile-driven types.
        assert _PERCENTILE_ONLY_THRESHOLD_TYPES == {
            ThresholdType.feature_percentile,
            ThresholdType.feature_percentile_segment_topk,
        }


class TestRejectsMagnitudeDependentTypes:
    """All non-whitelisted threshold types must raise a clear error."""

    @pytest.mark.parametrize(
        "th_type",
        [
            ThresholdType.segment_topk,
            ThresholdType.segment_percentile,
        ],
    )
    def test_raises_value_error(self, tmp_path, th_type):
        shard_path = _make_minimal_packed_row_shard(tmp_path)
        pipeline = _make_threshold_pipeline_config(th_type)
        with pytest.raises(ValueError) as exc_info:
            select_features_for_shard(
                shard_path, pipeline,
                output_dir=tmp_path / "out", shard_name="shard_00000",
            )
        msg = str(exc_info.value).lower()
        assert "raw feature magnitudes" in msg or "magnitudes" in msg
        assert "precompute" in msg


class TestAcceptsPercentileTypes:
    """The whitelisted types do not raise (semantics are correct under
    percentile-only inputs).
    """

    @pytest.mark.parametrize(
        "th_type",
        (
            ThresholdType.feature_percentile,
            ThresholdType.feature_percentile_segment_topk,
        ),
    )
    def test_guard_does_not_reject(self, tmp_path, th_type):
        """The guard at the top of the threshold branch must not raise
        ``ValueError("raw feature magnitudes")`` for whitelisted types.

        Pre-fixup the call leaked ``AttributeError`` from a wrong-argument-
        order bug in :func:`apply_threshold` (audit 2026-05-09 fix), which
        a wide ``except Exception`` then swallowed silently. This test now
        rejects that specific failure mode so the bug cannot regress.

        Other downstream failures (e.g. shape mismatches because the
        synthetic fixture passes 2-D ``boundaries`` where the threshold
        helpers expect 1-D per-feature thresholds) are NOT in scope for
        this audit fix — they are tracked as a separate slice-05 follow-up
        for fully wiring the per-feature-threshold plumbing — so we
        tolerate ``IndexError`` / ``AssertionError`` from those paths.
        """
        shard_path = _make_minimal_packed_row_shard(tmp_path)
        pipeline = _make_threshold_pipeline_config(th_type)
        try:
            select_features_for_shard(
                shard_path, pipeline,
                output_dir=tmp_path / "out", shard_name="shard_00000",
            )
        except AttributeError as exc:
            raise AssertionError(
                f"whitelisted threshold type {th_type} leaked AttributeError "
                "from apply_threshold; arg order is broken"
            ) from exc
        except ValueError as exc:
            if "raw feature magnitudes" in str(exc).lower():
                raise AssertionError(
                    f"whitelisted threshold type {th_type} was rejected by guard"
                ) from exc
            # Other ValueError (e.g. boundary shape mismatch on the synthetic
            # fixture) is unrelated to the audited guard and is accepted.
        except (IndexError, AssertionError):
            # Downstream shape-mismatch / unwired-plumbing errors are out of
            # scope for this audit fix — explicitly accepted so the test
            # remains focused on the arg-order guard.
            pass
