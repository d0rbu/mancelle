"""Tests for steering.gemma3_loading — VLM/text-only loader and guard."""

from __future__ import annotations

import os

import pytest
import torch as t
import transformers
from safetensors.torch import load_file, save_file
from transformers import (
    Gemma3Config,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    Gemma3TextConfig,
    SiglipVisionConfig,
)

import steering.gemma3_loading as gl
from errors import ExtractionError, ModelWeightLoadError, NanoGPTError
from steering.gemma3_loading import (
    _read_remapped_text_state_dict,
    _resolve_snapshot,
    assert_weights_materialized,
    from_pretrained_checked,
    is_gemma3_vlm,
    load_gemma3_text_decoder,
)


def _tiny_text_config() -> Gemma3TextConfig:
    return Gemma3TextConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        sliding_window=8, max_position_embeddings=32,
    )


def _tiny_vlm_config() -> Gemma3Config:
    vision = SiglipVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, image_size=16, patch_size=8,
    )
    return Gemma3Config(text_config=_tiny_text_config(), vision_config=vision)


# ---------------------------------------------------------------------------
# Task 1: assert_weights_materialized + ModelWeightLoadError
# ---------------------------------------------------------------------------


def test_guard_passes_on_clean_load():
    assert_weights_materialized(
        missing_keys=[], unexpected_keys=[], mismatched_keys=[],
        tie_word_embeddings=False, context="x",
    )


def test_guard_allows_tied_lm_head_only_when_tied():
    assert_weights_materialized(
        missing_keys=["lm_head.weight"], unexpected_keys=[],
        tie_word_embeddings=True, context="x",
    )
    with pytest.raises(ModelWeightLoadError):
        assert_weights_materialized(
            missing_keys=["lm_head.weight"], unexpected_keys=[],
            tie_word_embeddings=False, context="x",
        )


def test_guard_raises_on_missing_layer_keys():
    with pytest.raises(ModelWeightLoadError, match="model.layers.0.self_attn.q_proj.weight"):
        assert_weights_materialized(
            missing_keys=["model.layers.0.self_attn.q_proj.weight"],
            unexpected_keys=[], tie_word_embeddings=True, context="gemma-3-4b-pt",
        )


def test_guard_raises_on_unexpected_keys():
    with pytest.raises(ModelWeightLoadError):
        assert_weights_materialized(
            missing_keys=[], unexpected_keys=["language_model.model.embed_tokens.weight"],
            tie_word_embeddings=True, context="x",
        )


def test_model_weight_load_error_is_not_extraction_error():
    assert issubclass(ModelWeightLoadError, NanoGPTError)
    assert not issubclass(ModelWeightLoadError, ExtractionError)


# ---------------------------------------------------------------------------
# Task 2: is_gemma3_vlm discriminator
# ---------------------------------------------------------------------------


def test_is_gemma3_vlm_true_for_vlm_config():
    assert is_gemma3_vlm(_tiny_vlm_config()) is True


def test_is_gemma3_vlm_false_for_text_only_config():
    assert is_gemma3_vlm(_tiny_text_config()) is False


# ---------------------------------------------------------------------------
# Task 3: load_gemma3_text_decoder helpers + tests
# ---------------------------------------------------------------------------


def _save_tiny_vlm(path) -> Gemma3ForConditionalGeneration:
    t.manual_seed(0)
    vlm = Gemma3ForConditionalGeneration(_tiny_vlm_config())
    vlm.save_pretrained(path, safe_serialization=True)
    return vlm


def _save_tiny_text_only(path) -> Gemma3ForCausalLM:
    t.manual_seed(1)
    model = Gemma3ForCausalLM(_tiny_text_config())
    model.save_pretrained(path, safe_serialization=True)
    return model


def test_vlm_remap_text_weights_byte_identical(tmp_path):
    vlm = _save_tiny_vlm(tmp_path)
    loaded = load_gemma3_text_decoder(str(tmp_path), dtype=t.float32)
    ref = vlm.model.language_model  # Gemma3TextModel with correct weights
    for i in range(len(ref.layers)):
        assert t.equal(
            loaded.model.layers[i].self_attn.q_proj.weight,  # ty: ignore[invalid-argument-type, unresolved-attribute]
            ref.layers[i].self_attn.q_proj.weight,
        )
    assert t.equal(loaded.model.embed_tokens.weight, ref.embed_tokens.weight)
    assert t.equal(loaded.model.norm.weight, ref.norm.weight)


def test_vlm_forward_matches_text_submodule(tmp_path):
    vlm = _save_tiny_vlm(tmp_path)
    loaded = load_gemma3_text_decoder(str(tmp_path), dtype=t.float32)
    ids = t.tensor([[1, 5, 9, 3, 7, 2]])
    with t.no_grad():
        ref_hs = vlm.model.language_model(input_ids=ids, output_hidden_states=True).hidden_states
        got_hs = loaded.model(input_ids=ids, output_hidden_states=True).hidden_states
    for a, b in zip(ref_hs, got_hs, strict=True):
        assert t.equal(a, b)


def test_vlm_embed_scale_buffer_correct(tmp_path):
    # The meta/to_empty trap: embed_scale (=sqrt(hidden_size), non-persistent)
    # must be correct, not garbage.
    _save_tiny_vlm(tmp_path)
    loaded = load_gemma3_text_decoder(str(tmp_path), dtype=t.float32)
    assert float(loaded.model.embed_tokens.embed_scale) == pytest.approx(16 ** 0.5)  # ty: ignore[invalid-argument-type]


def test_text_only_load_byte_identical(tmp_path):
    ref = _save_tiny_text_only(tmp_path)
    loaded = load_gemma3_text_decoder(str(tmp_path), dtype=t.float32)
    for i in range(len(ref.model.layers)):
        assert t.equal(
            loaded.model.layers[i].self_attn.q_proj.weight,  # ty: ignore[invalid-argument-type, unresolved-attribute]
            ref.model.layers[i].self_attn.q_proj.weight,  # ty: ignore[invalid-argument-type, unresolved-attribute]
        )


def test_guard_fires_on_corrupt_vlm_checkpoint(tmp_path):
    _save_tiny_vlm(tmp_path)
    shard = os.path.join(tmp_path, "model.safetensors")
    sd = load_file(shard)
    victim = next(k for k in sd if k.startswith("language_model.model.layers.0.self_attn.q_proj"))
    del sd[victim]
    save_file(sd, shard)
    with pytest.raises(ModelWeightLoadError):
        load_gemma3_text_decoder(str(tmp_path), dtype=t.float32)


# ---------------------------------------------------------------------------
# Coverage-gap tests (issue #549 hardening audit)
# ---------------------------------------------------------------------------


def test_read_remapped_raises_on_empty_dir(tmp_path):
    """A dir with no .safetensors files → _read_remapped_text_state_dict raises
    ModelWeightLoadError matching 'no safetensors shards'.

    Non-vacuity: the guard is the *only* raise in _read_remapped_text_state_dict
    at that point; removing it makes the function return {} silently instead.
    """
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ModelWeightLoadError, match="no safetensors shards"):
        _read_remapped_text_state_dict(str(tmp_path))


def test_vlm_remap_includes_untied_lm_head(tmp_path):
    """An explicit language_model.lm_head.weight in the VLM checkpoint is
    remapped to lm_head.weight in the returned state dict.

    Non-vacuity: removing the elif branch that handles _VLM_LM_HEAD_PREFIX
    means lm_head.weight is absent from the result and the assertion fails.
    """
    _save_tiny_vlm(tmp_path)
    shard = os.path.join(tmp_path, "model.safetensors")
    sd = load_file(shard)
    # Inject a synthetic lm_head tensor under the VLM prefix
    injected = t.ones(64, 16, dtype=t.float32)  # (vocab_size, hidden_size)
    sd["language_model.lm_head.weight"] = injected
    save_file(sd, shard)
    result = _read_remapped_text_state_dict(str(tmp_path))
    assert "lm_head.weight" in result, f"keys: {list(result.keys())[:10]}"
    assert t.equal(result["lm_head.weight"], injected)


def test_vlm_remap_multi_shard(tmp_path):
    """Tensors split across two .safetensors files are both loaded and the
    resulting model weights are byte-identical to the original VLM text submodule.

    Non-vacuity: if we only read the first shard, layer 1 weights would be
    random-initialised and the equality assertions would fail.
    """
    vlm = _save_tiny_vlm(tmp_path)
    shard = os.path.join(tmp_path, "model.safetensors")
    sd = load_file(shard)
    # Split tensors into two halves
    keys = sorted(sd.keys())
    mid = len(keys) // 2
    part_a = {k: sd[k] for k in keys[:mid]}
    part_b = {k: sd[k] for k in keys[mid:]}
    save_file(part_a, os.path.join(tmp_path, "shard_a.safetensors"))
    save_file(part_b, os.path.join(tmp_path, "shard_b.safetensors"))
    os.remove(shard)

    loaded = load_gemma3_text_decoder(str(tmp_path), dtype=t.float32)
    ref = vlm.model.language_model
    for i in range(len(ref.layers)):
        assert t.equal(
            loaded.model.layers[i].self_attn.q_proj.weight,  # ty: ignore[invalid-argument-type, unresolved-attribute]
            ref.layers[i].self_attn.q_proj.weight,
        ), f"layer {i} q_proj mismatch after multi-shard load"
    assert t.equal(loaded.model.embed_tokens.weight, ref.embed_tokens.weight)


def test_vlm_load_bf16_dtype(tmp_path):
    """load_gemma3_text_decoder with dtype=bfloat16 returns a model whose
    embed_tokens weight has dtype bfloat16 (CPU, tiny synthetic model).

    Non-vacuity: removing the .to(dtype) call at the end of load_gemma3_text_decoder
    makes embed_tokens remain float32 and the dtype assertion fails.
    """
    _save_tiny_vlm(tmp_path)
    model = load_gemma3_text_decoder(str(tmp_path), dtype=t.bfloat16)
    assert model.model.embed_tokens.weight.dtype == t.bfloat16


def test_is_gemma3_vlm_str_dispatches_to_autoconfig(monkeypatch):
    """When a string id is passed, is_gemma3_vlm calls AutoConfig.from_pretrained
    and returns True for a VLM config (has text_config), False for text-only.

    Non-vacuity: removing the `isinstance(config, str)` branch makes the function
    call getattr on the string itself, returning None → always False.
    """
    results: list[object] = []

    def fake_from_pretrained(model_id):
        results.append(model_id)
        if "vlm" in str(model_id):
            return _tiny_vlm_config()
        return _tiny_text_config()

    monkeypatch.setattr(gl.AutoConfig, "from_pretrained", fake_from_pretrained)

    assert is_gemma3_vlm("fake/vlm-model") is True
    assert is_gemma3_vlm("fake/text-only") is False
    assert "fake/vlm-model" in results
    assert "fake/text-only" in results


def test_resolve_snapshot_hub_calls_snapshot_download(tmp_path, monkeypatch):
    """When hf_id is NOT a local dir, _resolve_snapshot calls snapshot_download
    with the id and the expected allow_patterns, and returns its result.

    Non-vacuity: removing the snapshot_download branch (or the allow_patterns
    kwarg) makes the assertion that snapshot_download was called fail.
    """
    calls: list[dict] = []

    def fake_snapshot_download(hf_id, *, allow_patterns):
        calls.append({"hf_id": hf_id, "allow_patterns": allow_patterns})
        return str(tmp_path)

    monkeypatch.setattr(gl, "snapshot_download", fake_snapshot_download)

    # A non-existent local path → must go to snapshot_download
    result = _resolve_snapshot("google/gemma-3-4b-pt")
    assert result == str(tmp_path)
    assert len(calls) == 1
    assert calls[0]["hf_id"] == "google/gemma-3-4b-pt"
    assert "*.safetensors" in calls[0]["allow_patterns"]
    assert "*.json" in calls[0]["allow_patterns"]


def test_assert_weights_materialized_mismatched_raises():
    """mismatched_keys (shape/dtype mismatch) triggers ModelWeightLoadError even
    when missing_keys is empty and tie_word_embeddings=True.

    Non-vacuity: removing `mismatched` from the `if bad_missing or unexpected or
    mismatched:` guard silently ignores the mismatch and returns without raising,
    allowing silently-wrong weights to propagate.
    """
    with pytest.raises(ModelWeightLoadError):
        assert_weights_materialized(
            missing_keys=[],
            unexpected_keys=[],
            mismatched_keys=["model.embed_tokens.weight"],
            tie_word_embeddings=True,
            context="x",
        )


# ---------------------------------------------------------------------------
# Task 2 hardening — from_pretrained_checked (F2), shape-mismatch (F1),
# no_init_weights byte-identity (F6), is_gemma3_vlm routing (F9)
# ---------------------------------------------------------------------------


def test_from_pretrained_checked_runs_strict_guard(monkeypatch):
    class _Cfg:
        tie_word_embeddings = True

    class _M:
        config = _Cfg()

    def fake(hf_id, **kw):
        return _M(), {"missing_keys": ["model.layers.0.q"], "unexpected_keys": set(), "mismatched_keys": []}

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(fake))
    with pytest.raises(ModelWeightLoadError):
        from_pretrained_checked(transformers.AutoModelForCausalLM, "x", dtype=t.float32)


def test_vlm_shape_mismatch_raises_model_weight_load_error(tmp_path):
    # corrupt a text tensor's SHAPE (load_state_dict raises RuntimeError, not a
    # missing/unexpected key) → must surface the typed fatal error, not RuntimeError.
    _save_tiny_vlm(tmp_path)
    shard = os.path.join(tmp_path, "model.safetensors")
    sd = load_file(shard)
    victim = next(k for k in sd if k.startswith("language_model.model.layers.0.self_attn.q_proj"))
    sd[victim] = sd[victim][:, :-1].contiguous()  # wrong shape
    save_file(sd, shard)
    with pytest.raises(ModelWeightLoadError):
        load_gemma3_text_decoder(str(tmp_path), dtype=t.float32)


def test_vlm_no_init_byte_identical_to_normal_construction(tmp_path):
    # F6: no_init_weights must not change any param OR buffer vs the prior path.
    vlm = _save_tiny_vlm(tmp_path)
    loaded = load_gemma3_text_decoder(str(tmp_path), dtype=t.float32)
    ref = vlm.model.language_model
    # Materialize both sides and assert equal counts BEFORE zip — zip truncates
    # silently, so a missing/extra tensor would otherwise pass unnoticed.
    p_loaded = list(loaded.model.named_parameters())
    p_ref = list(ref.named_parameters())
    assert len(p_loaded) == len(p_ref), f"param count {len(p_loaded)} != {len(p_ref)}"
    for (ln, lp), (rn, rp) in zip(p_loaded, p_ref, strict=True):
        assert ln == rn and t.equal(lp, rp)
    b_loaded = list(loaded.model.named_buffers())
    b_ref = list(ref.named_buffers())
    assert len(b_loaded) == len(b_ref), f"buffer count {len(b_loaded)} != {len(b_ref)}"
    for (ln, lb), (rn, rb) in zip(b_loaded, b_ref, strict=True):
        assert ln == rn and t.equal(lb, rb)  # embed_scale, rotary inv_freq materialised


def test_loader_routes_branch_through_is_gemma3_vlm(tmp_path, monkeypatch):
    _save_tiny_vlm(tmp_path)
    seen: dict[str, object] = {}
    real = gl.is_gemma3_vlm

    def _spy(config):
        # Transparent spy: record the call AND return the REAL verdict, so the
        # loader routes on the genuine discriminator output (a bare
        # `setdefault(...) or real(c)` short-circuits and never calls real()).
        verdict = real(config)
        seen["verdict"] = verdict
        return verdict

    monkeypatch.setattr(gl, "is_gemma3_vlm", _spy)
    load_gemma3_text_decoder(str(tmp_path), dtype=t.float32)
    # The discriminator was consulted AND correctly classified the VLM fixture
    # as a VLM (proving real() actually ran and the loader took its verdict).
    assert seen.get("verdict") is True
