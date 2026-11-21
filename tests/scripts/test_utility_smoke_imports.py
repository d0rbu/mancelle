"""Smoke import tests for slice 18 utility scripts.

Per design spec §4, these scripts are out-of-scope for full-test investment
but must at least import cleanly. Map.md §6 P2 #18 / P2 #19 flags that
``verify_bf16_threshold.py`` imports private names (``_truncate_to_training_window``,
``_select_spot_check_docs``); a future slice 1/3 refactor would silently
break the script. An import-time test catches this.

This test does NOT exercise the scripts' main(); ``main()`` requires GPU,
parquet shards, and SAE checkpoints. Importability is the operator-facing
contract.
"""

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

UTILITY_MODULES = [
    "scripts.utils.hf_pull",
    "scripts.utils.hf_push",
    "scripts.utils.nnterp_weight_transform",
    "scripts.utils.verify_bf16_threshold",
    "scripts.utils.recover_chat_target_real_from_logs",
    "scripts.utils.validate_token_boundary_overlays",
]


@pytest.mark.parametrize("module_name", UTILITY_MODULES)
def test_utility_module_imports_cleanly(module_name: str) -> None:
    """Each utility module must be import-clean (no top-level side effects
    that crash, no broken cross-slice imports)."""
    module = importlib.import_module(module_name)
    assert module is not None, f"importlib returned None for {module_name}"


@pytest.mark.parametrize(
    "module_name,attr",
    [
        ("scripts.utils.hf_pull", "main"),
        ("scripts.utils.hf_push", "main"),
        ("scripts.utils.nnterp_weight_transform", "main"),
        ("scripts.utils.nnterp_weight_transform", "tl_to_hf_state_dict"),
        ("scripts.utils.verify_bf16_threshold", "main"),
        ("scripts.utils.verify_bf16_threshold", "get_hidden_pre"),
    ],
)
def test_utility_module_has_public_attr(module_name: str, attr: str) -> None:
    """Public surface check: each utility script exposes its documented
    entry points."""
    module = importlib.import_module(module_name)
    assert hasattr(module, attr), f"{module_name} missing public attribute ``{attr}``"


def test_verify_bf16_threshold_rejects_turn_delimited_shards() -> None:
    module = importlib.import_module("scripts.utils.verify_bf16_threshold")
    with pytest.raises(
        ValueError,
        match="bf16 threshold verification does not support",
    ):
        module.require_fixed_segment_mode(
            module.segment_mode_from_metadata(
                {"segment_mode": "turn_delimited_fixed_tokens"}, source="test shard"
            ),
            surface="bf16 threshold verification",
            source="feature shard",
        )
    assert "require_fixed_segment_mode(" in inspect.getsource(module.main)


def test_verify_bf16_threshold_main_wires_fixed_mode_guard(monkeypatch, tmp_path) -> None:
    """The GPU-only main path applies the guard after loading each shard."""
    module = importlib.import_module("scripts.utils.verify_bf16_threshold")
    model = Mock()
    sae = Mock(threshold=module.t.ones(2))
    monkeypatch.setattr(module, "resolve_sae", lambda **_kwargs: SimpleNamespace(hook_name="h"))
    monkeypatch.setattr(
        module,
        "load_sae_and_model",
        lambda **_kwargs: (model, sae, object()),
    )
    monkeypatch.setattr(module, "glob_features", lambda _path: [tmp_path / "features"])
    monkeypatch.setattr(module, "shard_stem_from_path", lambda _path: "shard_00000")
    monkeypatch.setattr(
        module,
        "load_features",
        lambda _path: {"segment_mode": "turn_delimited_fixed_tokens"},
    )
    monkeypatch.setattr(module, "PARQUET_DIR", tmp_path)
    (tmp_path / "shard_00000.parquet").touch()

    with pytest.raises(
        ValueError,
        match="bf16 threshold verification does not support",
    ):
        module.main()


def test_chat_serving_clis_import() -> None:
    """Serving CLIs import cleanly. ``chat_web`` and ``steer_chat_cli``
    parse Hydra at module top via compose API; this test confirms the
    module-top compose succeeds against the canonical configs."""
    importlib.import_module("scripts.chat.chat_cli")
    importlib.import_module("scripts.chat.steer_chat_cli")
    # NOTE: chat_web is Hydra-loaded at module top; importing here resolves
    # cfg + autodetects the device. This is the project-blessed pattern
    # (see map.md §3.3).
    importlib.import_module("scripts.chat.chat_web")
