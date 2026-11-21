from __future__ import annotations

import importlib.util
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch as t

from steering.eval.autointerp_filter.config import (
    AutoInterpFilterConfig,
    TokenExamplesConfig,
)
from steering.eval.autointerp_filter.intruder import build_intruder_messages
from steering.eval.autointerp_filter.renderers import TokenWindowRenderer
from steering.eval.autointerp_filter.token_examples import (
    TokenExampleSampler,
    positive_highlight_indices,
)
from steering.eval.autointerp_filter.types import RenderedExample, TokenWindowRef

DELPHI_REF_PATH = os.environ.get("DELPHI_REF_PATH")
pytestmark = pytest.mark.skipif(
    not DELPHI_REF_PATH,
    reason="set DELPHI_REF_PATH to an EleutherAI/delphi checkout to run parity tests",
)


def _load_ref_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_delphi_stubs() -> list[str]:
    installed: list[str] = []

    def put(name: str, module: types.ModuleType) -> None:
        sys.modules[name] = module
        installed.append(name)

    delphi = types.ModuleType("delphi")
    delphi.__path__ = []
    delphi_module: Any = delphi
    delphi_module.logger = logging.getLogger("delphi-reference-test")
    put("delphi", delphi)

    latents = types.ModuleType("delphi.latents")
    latents_module: Any = latents
    latents_module.ActivatingExample = type("ActivatingExample", (), {})
    latents_module.NonActivatingExample = type("NonActivatingExample", (), {})
    put("delphi.latents", latents)

    scorers = types.ModuleType("delphi.scorers")
    scorers.__path__ = []
    put("delphi.scorers", scorers)

    classifier = types.ModuleType("delphi.scorers.classifier")
    classifier.__path__ = []
    put("delphi.scorers.classifier", classifier)

    return installed


def _load_ref_prompt(ref_root: Path):
    return _load_ref_module(
        "delphi_ref_intruder_prompt",
        ref_root / "delphi/scorers/classifier/prompts/intruder_prompt.py",
    )


def _load_ref_sample(ref_root: Path):
    installed = _install_delphi_stubs()
    try:
        return _load_ref_module(
            "delphi.scorers.classifier.sample",
            ref_root / "delphi/scorers/classifier/sample.py",
        )
    finally:
        for name in installed:
            sys.modules.pop(name, None)


def _ref(
    token_strings: tuple[str, ...],
    activations: tuple[float, ...],
    *,
    active: bool,
) -> TokenWindowRef:
    return TokenWindowRef(
        shard_stem="shard_00000_features",
        shard_index=0,
        doc_id=0,
        source_text_hash="hash",
        token_start=0,
        token_end=len(token_strings),
        token_ids=tuple(range(len(token_strings))),
        token_strings=token_strings,
        target_feature_ids=(0,),
        activations=activations,
        max_activation=max(activations, default=0.0),
        quantile=0,
        active=active,
    )


class _ReferenceExample:
    def __init__(self, tokens: tuple[str, ...], activations: tuple[float, ...]) -> None:
        self.str_tokens = list(tokens)
        self.activations = t.tensor(activations, dtype=t.float32)

    @property
    def max_activation(self) -> float:
        return float(self.activations.max())


class _UnusedProvider:
    def prepare(self, targets):
        del targets
        raise AssertionError("not used")

    def examples_for_target(self, target):
        del target
        raise AssertionError("not used")

    def cache_context(self) -> dict:
        raise AssertionError("not used")


def test_intruder_prompt_matches_delphi_reference() -> None:
    ref_root = Path(str(DELPHI_REF_PATH))
    ref_prompt = _load_ref_prompt(ref_root)
    texts = [
        "alpha <<one>>",
        "beta two",
        "gamma three",
        "delta four",
    ]
    rendered = [
        RenderedExample(
            text=text,
            ref=_ref((text,), (1.0,), active=True),
            activating=True,
            highlighted_token_indices=(0,),
        )
        for text in texts
    ]
    examples_arg = "\n".join(f"Example {idx}:{text}" for idx, text in enumerate(texts))

    assert build_intruder_messages(
        rendered,
        cot=False,
        prompt_variant="delphi",
        few_shot_set="delphi",
    ) == ref_prompt.prompt(examples_arg, cot=False)
    assert build_intruder_messages(
        rendered,
        cot=True,
        prompt_variant="delphi",
        few_shot_set="delphi",
    ) == ref_prompt.prompt(examples_arg, cot=True)


def test_token_rendering_matches_delphi_prepare_text(tmp_path: Path) -> None:
    ref_root = Path(str(DELPHI_REF_PATH))
    ref_sample = _load_ref_sample(ref_root)

    positive_tokens = ("a", " b", " c", " d")
    positive_activations = (0.0, 2.0, 0.5, 4.0)
    positive_example = _ReferenceExample(positive_tokens, positive_activations)
    expected_positive, _ = ref_sample._prepare_text(
        positive_example,
        n_incorrect=0,
        threshold=0.3,
        highlighted=True,
    )
    positive_ref = _ref(positive_tokens, positive_activations, active=True)
    positive_indices = positive_highlight_indices(
        positive_ref,
        activation_threshold_ratio=0.3,
    )
    rendered_positive = TokenWindowRenderer().render(
        positive_ref,
        activating=True,
        highlighted_token_indices=positive_indices,
    )
    assert rendered_positive.text == expected_positive

    intruder_tokens = tuple(f" t{idx}" for idx in range(8))
    intruder_activations = tuple(0.0 for _ in intruder_tokens)
    intruder_example = _ReferenceExample(intruder_tokens, intruder_activations)
    expected_intruder, _ = ref_sample._prepare_text(
        intruder_example,
        n_incorrect=3,
        threshold=0.3,
        highlighted=True,
    )
    intruder_ref = _ref(intruder_tokens, intruder_activations, active=False)
    sampler = TokenExampleSampler(
        AutoInterpFilterConfig(
            feature_dir=str(tmp_path / "features"),
            output_dir=str(tmp_path / "out"),
            token_examples=TokenExamplesConfig(),
        ),
        provider=_UnusedProvider(),
    )
    intruder_indices = sampler._sample_intruder_highlights(  # noqa: SLF001
        intruder_ref,
        n_highlights=3,
    )
    rendered_intruder = TokenWindowRenderer().render(
        intruder_ref,
        activating=False,
        highlighted_token_indices=intruder_indices,
    )
    assert rendered_intruder.text == expected_intruder
