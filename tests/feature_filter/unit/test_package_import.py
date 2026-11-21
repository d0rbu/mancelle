"""Smoke test: the feature_filter package imports cleanly."""


def test_package_imports() -> None:
    import steering.eval.feature_filter

    assert steering.eval.feature_filter is not None
