"""Tests documenting the known bug: percentile checksums are NOT written to
individual shard metadata by the precompute pipeline.

The precompute_percentiles.py script calls save_overlay_shard() without passing
a metadata dict, so MetadataKey.PERCENTILE_CHECKSUM is never embedded in each
shard file.  This causes validate_shard_data() to reject any shard that has
feature_percentiles but no accompanying checksum.

These xfail tests prove the bug exists and will start xpassing if it is ever
accidentally fixed (strict=True ensures CI catches that so the tests can be
updated to reflect the new expected behaviour).
"""

import tempfile
from pathlib import Path

import pytest
import torch as t

from steering.feature_io import MetadataKey, TensorKey
from steering.overlay_io import load_overlay_shard, save_overlay_shard
from steering.shard_validation import validate_shard_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pct_uint8(n_features: int = 200) -> t.Tensor:
    """Build a minimal uint8 percentile tensor (one value per feature)."""
    # Simulate what precompute_percentiles.py produces: a flat uint8 tensor
    # where each entry encodes the precomputed percentile for one activation.
    return t.randint(0, 256, (n_features,), dtype=t.uint8)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestPercentileChecksumInShardMetadata:
    """Document the missing-checksum bug in overlay shard writes."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG: precompute_percentiles.py calls save_overlay_shard() without "
            "a metadata= argument, so MetadataKey.PERCENTILE_CHECKSUM is never "
            "written into shard files.  load_overlay_shard() (which uses "
            "safetensors.torch.load) only returns tensors, never metadata fields, "
            "so the checksum key will never be present in the returned dict."
        ),
    )
    def test_overlay_shard_contains_checksum_metadata(self) -> None:
        """Shard written without metadata should NOT contain a checksum key.

        Mimics exactly what precompute_percentiles.py does:
            save_overlay_shard({TensorKey.FEATURE_PERCENTILES: pct_uint8}, out_path)
        — no metadata= kwarg passed.

        The loaded dict should contain MetadataKey.PERCENTILE_CHECKSUM, but it
        does not because the checksum is never written.  This assertion FAILS,
        proving the bug.
        """
        pct_uint8 = _make_pct_uint8(n_features=200)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "shard_00000"
            # Mimic precompute_percentiles.py: no metadata kwarg
            written_path = save_overlay_shard(
                {TensorKey.FEATURE_PERCENTILES: pct_uint8},
                out_path,
            )

            loaded = load_overlay_shard(written_path)

        # BUG: this assertion fails because load_overlay_shard returns only
        # tensors (via safetensors.torch.load) and the checksum was never
        # written into the file's metadata header anyway.
        assert MetadataKey.PERCENTILE_CHECKSUM in loaded

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG: validate_shard_data() enforces that feature_percentiles and "
            "percentile_checksum must appear together.  A shard built by "
            "precompute_percentiles.py has feature_percentiles but no "
            "percentile_checksum, so validation always fails with the violation "
            "'feature_percentiles present but percentile_checksum missing'."
        ),
    )
    def test_shard_validation_passes_for_precomputed_shard(self) -> None:
        """validate_shard_data() should accept a shard with only feature_percentiles.

        Builds the minimal in-memory data dict that precompute_percentiles.py
        would produce for an overlay shard: only TensorKey.FEATURE_PERCENTILES,
        no MetadataKey.PERCENTILE_CHECKSUM.

        The assertion result.valid is True FAILS because _validate_percentiles()
        requires the checksum to accompany the percentile tensor.
        """
        n_features = 200
        pct_uint8 = _make_pct_uint8(n_features=n_features)

        # Data dict as it would arrive from load_overlay_shard after the
        # precompute step — only the percentile tensor, no checksum key.
        data: dict = {
            TensorKey.FEATURE_PERCENTILES: pct_uint8,
        }

        result = validate_shard_data(data, n_sae_features=n_features)

        # BUG: this is False, not True, because the checksum is missing.
        assert result.valid
