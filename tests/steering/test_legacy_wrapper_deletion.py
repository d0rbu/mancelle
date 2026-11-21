"""Pin: EncoderDecoderEvalWrapper class is gone (PR-A final shape)."""

from __future__ import annotations

import pytest


def test_legacy_wrapper_class_no_longer_importable():
    with pytest.raises(ImportError):
        from steering.conditional.encoder_decoder.eval_adapter import (  # noqa: F401
            EncoderDecoderEvalWrapper,  # ty: ignore[unresolved-import]
        )


def test_eval_adapter_module_does_not_define_legacy_class_name():
    import steering.conditional.encoder_decoder.eval_adapter as mod
    assert not hasattr(mod, "EncoderDecoderEvalWrapper"), (
        "EncoderDecoderEvalWrapper deleted in PR-A"
    )


def test_no_remaining_references_in_steering_or_scripts():
    """grep equivalent: no production code imports the legacy class."""
    import subprocess
    result = subprocess.run(
        [
            "grep", "-rln",
            "from steering.conditional.encoder_decoder.eval_adapter import",
            "steering/",
            "scripts/",
        ],
        capture_output=True, text=True, check=False,
    )
    bad_files = []
    for fn in result.stdout.strip().splitlines():
        if not fn:
            continue
        with open(fn) as f:
            for line in f:
                if "EncoderDecoderEvalWrapper" in line and "TargetAware" not in line:
                    bad_files.append((fn, line.strip()))
                    break
    assert not bad_files, f"legacy wrapper still referenced: {bad_files}"
