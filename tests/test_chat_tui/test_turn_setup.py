"""Chat consumes the one-shot loader and producer without another policy path."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t

from scripts.chat_tui import extraction as extraction_module
from scripts.chat_tui import setup as chat_setup
from scripts.chat_tui.extraction import TurnGemmaExtractionService
from scripts.chat_tui.steering import contract_digest_for_setup
from scripts.chat_tui.turn_session import _render_prompt
from steering.conditional.encoder_decoder.config import EncoderConfig
from steering.eval.axbench_upstream import methods
from steering.segmentation import SegmentMode
from steering.sft.turn_spans import rendered_text_and_turn_char_starts
from tests.conftest import MockTokenizer
from tests.nanochat.test_structured_conversation_renderer import _CharacterTokenizer
from tests.scripts.eval.test_axbench_project_methods import (
    _install_turn_setup_fixture,
    _real_lineage_block,
    _turn_prompt_pipeline,
)


def loaded_setup(monkeypatch, tmp_path):
    lineage = _real_lineage_block(tmp_path)
    fixture = _install_turn_setup_fixture(
        monkeypatch,
        tmp_path,
        real_lineage_block=lineage,
        real_selection_context=True,
        prompt_pipeline=_turn_prompt_pipeline(seed=42),
    )
    # Only statistics deserialization, model resolution and feature-contract
    # fixtures are doubled. The complete 21-field lineage parser is real.
    stats_path = Path(lineage["statistics_path"])
    setup = chat_setup.load_chat_setup(fixture.meta_path, stats_path)
    return setup, stats_path


def test_turn_chat_uses_checkpoint_owned_selection_seed_sha_and_cadence(monkeypatch, tmp_path):
    setup, _ = loaded_setup(monkeypatch, tmp_path)
    assert setup.segment_mode is SegmentMode.turn_delimited_fixed_tokens
    assert setup.turn_self_cond_segment_size_rustbpe == 37
    assert setup.prompt_segment_size == 17  # Gemma source, NOT generated cadence
    assert setup.prompt_context_window == 333
    assert setup.prompt_selection_context is not None
    assert chat_setup.resolve_conditioning_seed(setup) == 42
    assert (
        chat_setup.resolve_stats_sha(setup)
        == setup.meta["structured_sft_online_sae_lineage_v1"]["statistics_sha256"]
    )
    assert (
        chat_setup.resolve_stats_sha(setup)
        != setup.meta["user_config"]["conditioning"]["stats_source_sha256"]
    )


@pytest.mark.parametrize("parent_error", [None, "tag", "sae", "conditioning"])
def test_real_sft_metadata_shape_requires_validated_pretrain_metadata(
    monkeypatch, tmp_path, parent_error
):
    expected, stats_path = loaded_setup(monkeypatch, tmp_path)
    leaf = json.loads(expected.target_meta_path.read_text())
    parent = json.loads(json.dumps(leaf))
    parent_path = tmp_path / "pretrain_meta.json"
    leaf["user_config"]["sae"].update(model="gemma-2-2b", layer=8, width="16k")
    del leaf["user_config"]["conditioning"]
    expected.target_meta_path.write_text(json.dumps(leaf))
    if parent_error == "tag":
        parent["user_config"]["model_tag"] = "wrong-model"
    elif parent_error == "sae":
        parent["user_config"]["sae"]["layer"] = 8
    elif parent_error == "conditioning":
        del parent["user_config"]["conditioning"]
    parent_path.write_text(json.dumps(parent))
    original_leaf_bytes = expected.target_meta_path.read_bytes()
    original_parent_bytes = parent_path.read_bytes()

    with pytest.raises(ValueError, match="pass pretrain_meta_path"):
        chat_setup.load_chat_setup(expected.target_meta_path, stats_path)
    if parent_error is not None:
        message = {
            "tag": "model_tag does not match",
            "sae": "lineage conflicts",
            "conditioning": "no user_config.conditioning mapping",
        }[parent_error]
        with pytest.raises(ValueError, match=message):
            chat_setup.load_chat_setup(
                expected.target_meta_path, stats_path, pretrain_meta_path=parent_path
            )
    else:
        actual = chat_setup.load_chat_setup(
            expected.target_meta_path, stats_path, pretrain_meta_path=parent_path
        )
        assert actual.meta == leaf  # Never repair or shim the checkpoint metadata.
        assert actual.sae_model == expected.sae_model
        assert actual.sae_layer == expected.sae_layer
        assert actual.n_sae_features == expected.n_sae_features
        assert actual.prompt_pipeline is expected.prompt_pipeline
        context = actual.prompt_selection_context
        reference = expected.prompt_selection_context
        assert context is not None and reference is not None
        assert context.sampling_bundle is reference.sampling_bundle
        assert context.negative_bundle is reference.negative_bundle
        assert context.missing_percentile.policy is reference.missing_percentile.policy
        assert context.missing_percentile.covered is not None
        assert reference.missing_percentile.covered is not None
        assert t.equal(context.missing_percentile.covered, reference.missing_percentile.covered)
        assert actual.output_selection == expected.output_selection
        assert actual.output_negative_selection == expected.output_negative_selection
        assert actual.feature_vocab == expected.feature_vocab
        assert actual.turn_self_cond_segment_size_rustbpe == 37
        assert chat_setup.resolve_conditioning_seed(actual) == 42  # Parent seed is 99.
        assert chat_setup.resolve_stats_sha(actual) == chat_setup.resolve_stats_sha(expected)
    assert expected.target_meta_path.read_bytes() == original_leaf_bytes
    assert parent_path.read_bytes() == original_parent_bytes


def test_fixed_chat_rejects_non_null_turn_pretrain_override_before_loading(monkeypatch, tmp_path):
    path = tmp_path / "meta_000001.json"
    path.write_text(json.dumps({"user_config": {}}))
    monkeypatch.setattr(
        chat_setup, "load_method_setup", lambda *_a, **_k: pytest.fail("shared loader reached")
    )
    with pytest.raises(ValueError, match="only supported for turn checkpoints"):
        chat_setup.load_chat_setup(
            path, tmp_path / "absent.safetensors", pretrain_meta_path=tmp_path / "absent.json"
        )


def _raw_turn_sft_leaf(monkeypatch, tmp_path, *, keep_selfcond=False, keep_conditioning=False):
    """The real 3B export shape: stale default SAE, no pretraining snapshot."""
    lineage = _real_lineage_block(tmp_path)
    fixture = _install_turn_setup_fixture(
        monkeypatch,
        tmp_path,
        real_lineage_block=lineage,
        real_selection_context=True,
        prompt_pipeline=_turn_prompt_pipeline(seed=42),
    )
    meta = json.loads(fixture.meta_path.read_text())
    parent = json.loads(json.dumps(meta))
    if not keep_selfcond:
        meta.pop("selfcond")
        meta.pop("selfcond_data")
    meta["encoder_config"] = asdict(EncoderConfig())
    if not keep_conditioning:
        meta["user_config"].pop("conditioning")
    meta["user_config"]["sae"] = {
        "release": "gemmascope",
        "model": "gemma-2-2b",
        "activation_type": "residual",
        "layer": 8,
        "width": "16k",
    }
    fixture.meta_path.write_text(json.dumps(meta))
    return fixture, lineage, meta, parent


def test_turn_sft_export_uses_lineage_without_rewriting_training_config(monkeypatch, tmp_path):
    fixture, lineage, meta, _ = _raw_turn_sft_leaf(monkeypatch, tmp_path)
    original = fixture.meta_path.read_bytes()
    setup = chat_setup.load_chat_setup(fixture.meta_path, Path(lineage["statistics_path"]))
    assert setup.sae_model == "gemma-3-4b-pt"
    assert setup.n_sae_features == 65_536
    assert setup.prompt_segment_size == lineage["source_segment_size"]
    # The lineage's raw statistics digest still binds the snapshot loader.
    assert fixture.observed["statistics_snapshot"] == (
        Path(lineage["statistics_path"]),
        lineage["statistics_sha256"],
    )
    assert setup.meta == meta
    assert fixture.meta_path.read_bytes() == original


@pytest.mark.parametrize("parent_error", [None, "missing", "tag", "sae"])
def test_raw_turn_sft_leaf_never_ignores_a_supplied_pretrain_meta_path(
    monkeypatch, tmp_path, parent_error
):
    # kills: routing raw SFT leaves around resolve_raw_sae_config /
    # _conditioning_block so an explicit parent is accepted and never read.
    fixture, lineage, meta, parent = _raw_turn_sft_leaf(monkeypatch, tmp_path)
    stats_path = Path(lineage["statistics_path"])
    parent_path = tmp_path / "pretrain_meta.json"
    if parent_error == "tag":
        parent["user_config"]["model_tag"] = "wrong-model"
    elif parent_error == "sae":
        parent["user_config"]["sae"]["layer"] = 8
    if parent_error != "missing":
        parent_path.write_text(json.dumps(parent))
    if parent_error is None:
        # The documented supported configuration: a correct parent loads the
        # same setup the lineage alone produces, and the leaf is untouched.
        without = chat_setup.load_chat_setup(fixture.meta_path, stats_path)
        with_parent = chat_setup.load_chat_setup(
            fixture.meta_path, stats_path, pretrain_meta_path=parent_path
        )
        assert (with_parent.sae_model, with_parent.sae_layer, with_parent.n_sae_features) == (
            without.sae_model,
            without.sae_layer,
            without.n_sae_features,
        )
        assert with_parent.feature_vocab == without.feature_vocab
        assert with_parent.meta == meta
        return
    if parent_error == "missing":
        with pytest.raises(FileNotFoundError, match="pretrain_meta.json"):
            chat_setup.load_chat_setup(
                fixture.meta_path, stats_path, pretrain_meta_path=parent_path
            )
        return
    message = {"tag": "model_tag does not match", "sae": "online-SAE lineage conflicts"}[
        parent_error
    ]
    with pytest.raises(ValueError, match=message):
        chat_setup.load_chat_setup(fixture.meta_path, stats_path, pretrain_meta_path=parent_path)


@pytest.mark.parametrize(
    ("recorded", "message"),
    [
        ("disagreeing-dict", "prompt conditioning SAE width disagrees"),
        (None, "no user_config.conditioning mapping"),
        ("corrupt", "no user_config.conditioning mapping"),
        ([{"n_sae_features": 999}], "no user_config.conditioning mapping"),
    ],
    ids=["disagreeing-dict", "null", "string", "list"],
)
def test_raw_turn_sft_leaf_keeps_a_recorded_conditioning_snapshot_cross_checked(
    monkeypatch, tmp_path, recorded, message
):
    # kills: fabricating the conditioning block from the lineage when the leaf
    # records one (disagreeing or malformed), turning raise into accept.
    fixture, lineage, meta, _ = _raw_turn_sft_leaf(monkeypatch, tmp_path, keep_conditioning=True)
    if recorded == "disagreeing-dict":
        meta["user_config"]["conditioning"]["n_sae_features"] = 999
    else:
        meta["user_config"]["conditioning"] = recorded
    fixture.meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=message):
        chat_setup.load_chat_setup(fixture.meta_path, Path(lineage["statistics_path"]))


def test_post_training_turn_leaf_with_stale_sae_still_requires_the_parent(monkeypatch, tmp_path):
    # kills: dropping the "selfcond" conjunct so post-training exports take the
    # lineage-only branch instead of the independent config/lineage cross-check.
    fixture, lineage, _, _ = _raw_turn_sft_leaf(monkeypatch, tmp_path, keep_selfcond=True)
    with pytest.raises(ValueError, match="online-SAE lineage conflicts.*pass pretrain_meta_path"):
        chat_setup.load_chat_setup(fixture.meta_path, Path(lineage["statistics_path"]))


@pytest.mark.parametrize("cadence", [None, True, False, 0, -1, 2.0, "2"])
def test_turn_chat_invalid_cadence_fails_before_shared_setup(monkeypatch, tmp_path, cadence):
    path = tmp_path / "meta_000001.json"
    path.write_text(
        json.dumps(
            {
                "user_config": {},
                "sft_segment_mode": SegmentMode.turn_delimited_fixed_tokens.value,
                "turn_self_cond_segment_size_rustbpe": cadence,
            }
        )
    )
    monkeypatch.setattr(
        chat_setup, "load_method_setup", lambda *_: pytest.fail("shared loader reached")
    )
    with pytest.raises(ValueError, match="turn_self_cond_segment_size_rustbpe"):
        chat_setup.load_chat_setup(path, tmp_path / "absent.safetensors")


@pytest.mark.parametrize("failure", [False, True])
def test_prompt_selection_rngs_are_restored_and_exact_entries_are_instance_local(
    monkeypatch, tmp_path, failure
):
    setup, stats_path = loaded_setup(monkeypatch, tmp_path)
    positive = setup.prompt_pipeline.selection_config.rng
    negative = setup.prompt_pipeline.negative_bundle.rng
    before = [rng.get_state().clone() for rng in (positive, negative)]
    calls = []
    monkeypatch.setattr(extraction_module, "load_extraction_stack", lambda *_: (object(), object()))

    def prepare(_setup, *_args, token_ids, assistant_end_id):
        calls.append(token_ids)
        t.randint(0, 100, (3,), generator=positive)
        t.randint(0, 100, (7,), generator=negative)
        if failure:
            raise RuntimeError("injected selection failure")
        return methods.PreparedPromptContext(token_ids, assistant_end_id, ())

    monkeypatch.setattr(extraction_module, "prepare_prompt_context", prepare)
    service = TurnGemmaExtractionService(setup, MockTokenizer(), "cpu", stats_path)
    if failure:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="injected"):
                service.prepare((0, 64, 12, 65, 66))
        assert service._cache == {}
    else:
        first = service.prepare((0, 64, 12, 65, 66))
        assert service.prepare(first.token_ids) is first
        service.prepare((0, 64, 13, 65, 66))
        other = TurnGemmaExtractionService(setup, MockTokenizer(), "cpu", stats_path)
        assert other.prepare(first.token_ids) is not first
        assert len(calls) == 3
    for rng, state in zip((positive, negative), before, strict=True):
        assert t.equal(rng.get_state(), state)


@pytest.mark.parametrize("changed", ["statistics", "metadata", "setup"])
def test_file_replacement_invalidates_even_an_exact_cache_hit(monkeypatch, tmp_path, changed):
    setup, stats_path = loaded_setup(monkeypatch, tmp_path)
    service = TurnGemmaExtractionService(setup, MockTokenizer(), "cpu", stats_path)
    tokens = (0, 64, 12, 65, 66)
    service._cache[tokens] = methods.PreparedPromptContext(tokens, 67, ())
    assert service.prepare(tokens) is service._cache[tokens]
    if changed == "statistics":
        stats_path.write_bytes(b"replacement-statistics")
    elif changed == "metadata":
        meta = json.loads(setup.target_meta_path.read_text())
        meta["turn_self_cond_segment_size_rustbpe"] += 1
        setup.target_meta_path.write_text(json.dumps(meta))
    else:
        setup.meta["turn_self_cond_segment_size_rustbpe"] += 1
    before = setup.prompt_pipeline.selection_config.rng.get_state().clone()
    with pytest.raises(ValueError, match="changed"):
        service.prepare(tokens)
    assert t.equal(before, setup.prompt_pipeline.selection_config.rng.get_state())


def test_turn_signal_cache_separates_missing_percentile_policy(monkeypatch, tmp_path):
    setup, _ = loaded_setup(monkeypatch, tmp_path)
    initial = contract_digest_for_setup(setup)
    meta = json.loads(json.dumps(setup.meta))
    meta["structured_sft_online_sae_lineage_v1"]["missing_percentile_policy"] = "error"
    assert contract_digest_for_setup(replace(setup, meta=meta)) != initial


def test_gemma_overflow_is_rejected_before_extraction_and_rng_restored(monkeypatch, tmp_path):
    setup, stats_path = loaded_setup(monkeypatch, tmp_path)
    model = SimpleNamespace(cfg=SimpleNamespace(n_ctx=333), tokenizer=object())
    monkeypatch.setattr(extraction_module, "load_extraction_stack", lambda *_: (model, object()))
    monkeypatch.setattr(methods, "truncate_to_training_window", lambda *_: ([1] * 333, 334))
    monkeypatch.setattr(
        methods, "extract_document_features", lambda **_: pytest.fail("extraction ran")
    )
    rng = setup.prompt_pipeline.selection_config.rng
    before = rng.get_state().clone()
    service = TurnGemmaExtractionService(setup, MockTokenizer(), "cpu", stats_path)
    with pytest.raises(ValueError, match="334 Gemma tokens.*333"):
        service.prepare((0, 64, 12, 65, 66))
    assert t.equal(before, rng.get_state())
    assert service._cache == {}


@pytest.mark.parametrize("text", ["hello", "世界 🌍 e\u0301", "▁width\nkept"])
def test_multiturn_tool_unicode_entry_equals_authoritative_producer(monkeypatch, tmp_path, text):
    setup, stats_path = loaded_setup(monkeypatch, tmp_path)
    tokenizer = _CharacterTokenizer()
    messages = [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": text},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "calculate"},
                {"type": "python", "text": "1 + 1"},
                {"type": "python_output", "text": "2"},
                {"type": "text", "text": text},
            ],
        },
        {"role": "user", "content": "next"},
    ]
    tokens = _render_prompt(tokenizer, messages)
    full, _ = tokenizer.render_conversation(
        {"messages": [*messages, {"role": "assistant", "content": ""}]}, max_tokens=2**31
    )
    assert tokens == tuple(full[:-1])
    rendered, turns = rendered_text_and_turn_char_starts(tokens, tokenizer)
    bos = "<bos>"
    starts = [0, *(len(bos) + start for start in turns[1:])]
    ends = [*starts[1:], len(bos) + len(rendered)]
    stack = SimpleNamespace(
        cfg=SimpleNamespace(n_ctx=333),
        tokenizer=SimpleNamespace(bos_token_id=1, decode=lambda _ids: bos),
    )
    observed = []

    def extract(**kwargs):
        observed.append(kwargs)
        return (
            [t.tensor([1, 2], dtype=t.int32) for _ in starts],
            [t.tensor([2.0, 8.0]) for _ in starts],
            list(range(len(starts))),
            starts,
            ends,
            333,
        )

    monkeypatch.setattr(extraction_module, "load_extraction_stack", lambda *_: (stack, object()))
    monkeypatch.setattr(methods, "truncate_to_training_window", lambda *_: ([1] * 333, 333))
    monkeypatch.setattr(methods, "extract_document_features", extract)
    service = TurnGemmaExtractionService(setup, tokenizer, "cpu", stats_path)
    actual = service.prepare(tokens)
    setup.prompt_pipeline.selection_config.rng.manual_seed(42)
    reference = methods.build_extracted_context_entry(
        setup,
        stack,
        object(),
        tokenizer,
        token_ids=tokens,
        source="independent one-shot oracle",
    )
    assert actual.token_ids == reference.token_ids
    assert actual.segments == reference.segments
    assert all(row["text"] == rendered and row["turn_char_starts"] == turns for row in observed)
    assert actual.segments[-1].char_end == len(tokenizer.decode(tokens))
