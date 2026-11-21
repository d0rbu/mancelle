from __future__ import annotations

from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError


def test_axbench_error_subclasses_feature_gcg_error():
    assert issubclass(AxBenchBpbAdvError, FeatureGcgError)


def test_feature_gcg_error_not_caught_by_axbench_matcher():
    # The base FeatureGcgError is NOT an AxBenchBpbAdvError instance.
    err = FeatureGcgError("x")
    assert not isinstance(err, AxBenchBpbAdvError)
    assert isinstance(AxBenchBpbAdvError("y"), FeatureGcgError)
