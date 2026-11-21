"""Tests for feature GCG plot-family labeling in _steering_family."""

from steering.eval.axbench_bpb_adv.aggregation import _steering_family


def test_gcg_plain_encdec_returns_feature_gcg() -> None:
    assert _steering_family("ours_gcg_encdec") == "feature GCG"


def test_gcg_neg_roofline_encdec_returns_feature_gcg() -> None:
    assert _steering_family("neg_ours_gcg_roofline_encdec") == "feature GCG"


def test_gcg_decoder_only_returns_feature_gcg() -> None:
    assert _steering_family("ours_gcg_decoder_only") == "feature GCG"


def test_gcg_fair_returns_feature_gcg() -> None:
    assert _steering_family("ours_gcg_fair_encdec") == "feature GCG"


def test_feature_diffmean_returns_feature_diffmean() -> None:
    assert _steering_family("feature_diffmean_encdec") == "feature DiffMean"
    assert _steering_family("neg_feature_diffmean_encdec") == "feature DiffMean"


def test_feature_diffmean_roofline_returns_feature_diffmean() -> None:
    # Mirrors GCG: the fair and roofline siblings share one family label, so the
    # roofline variant must be caught by prefix like ``ours_gcg`` -- not dropped
    # to its raw slug the way an ``==`` check would.
    assert _steering_family("feature_diffmean_roofline_encdec") == "feature DiffMean"
    assert (
        _steering_family("neg_feature_diffmean_roofline_encdec") == "feature DiffMean"
    )


def test_non_gcg_ours_unchanged() -> None:
    assert _steering_family("ours_extract_and_add_encdec") == "ours SAE edits"


def test_non_gcg_ours_decoder_only_unchanged() -> None:
    assert _steering_family("ours_extract_and_add_decoder_only") == "ours SAE edits"
