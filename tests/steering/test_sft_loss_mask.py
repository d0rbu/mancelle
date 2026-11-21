"""Completion-only SFT loss-mask contracts shared by both trainers."""

from __future__ import annotations

import ast
import inspect
import math
from enum import StrEnum
from pathlib import Path
from unittest.mock import patch

import pytest
import torch as t
import torch.nn.functional as F
from hydra import compose, initialize_config_dir

import nanochat.flash_attention as _fa_mod
import scripts.train.conditional_encdec_sft as encdec_sft
from nanochat.flash_attention import FlashAttnImpl
from nanochat.tokenizer import RustBPETokenizer
from scripts.train.conditional_encdec_sft import (
    SFTState,
    make_sft_data_generator,
)
from steering._common.ddp import rank_synchronized_stage
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.sft.iteration import sft_data_generator_bos_bestfit_core
from steering.sft.loss_mask import (
    LossMaskMode,
    apply_completion_only_row_masks,
    assert_min_one_valid_target,
    assert_resume_sft_loss_mask_matches,
    parse_loss_mask_mode,
    render_conversation_with_mask,
    supervised_target_count,
    truncate_conv_and_mask,
)
from steering.sft_schema import SFTSplit
from tests.conftest import _make_gpt
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    build_tiny_model,
)
from tests.steering.test_encdec_sft_conditioned_integration import (
    SFTMockTokenizer,
    SingleTurnTask,
    _make_boundaries,
    _make_gen,
    build_synthetic_features,
    patch_load_sft_features,
)
from tests.steering.test_sft_conv_truncation import (
    TinyConversationTask,
    _make_conditioned_generator,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "configs" / "training"


@pytest.fixture(autouse=True)
def _force_sdpa_backend():
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


class _SpanTokenizer:
    """One-token markers and one-token-per-character content for mask goldens."""

    BOS = 1
    SPECIAL = {
        "<user_start>": 10,
        "<user_end>": 11,
        "<assistant_start>": 12,
        "<assistant_end>": 13,
        "<python_start>": 14,
        "<python_end>": 15,
        "<output_start>": 16,
        "<output_end>": 17,
    }

    def get_bos_token_id(self) -> int:
        return self.BOS

    def encode_special(self, text: str) -> int:
        return self.SPECIAL[text]

    def encode(self, text: str) -> list[int]:
        return [100 + ord(char) for char in text]

    def render_conversation(self, conversation: dict) -> tuple[list[int], list[int]]:
        return RustBPETokenizer.render_conversation(
            self,  # ty: ignore[invalid-argument-type]
            conversation,
        )


def _all_span_conversation() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "A"},
                    {"type": "python", "text": "P"},
                    {"type": "python_output", "text": "O"},
                ],
            },
            {"role": "user", "content": "V"},
            {"role": "assistant", "content": "B"},
        ]
    }


EXPECTED_ALL_SPAN_MASK = [
    0,  # bos
    0,  # user_start
    0,
    0,
    0,
    0,  # merged "S\n\nU"
    0,  # user_end
    0,  # assistant_start
    1,  # assistant text
    1,  # python_start
    1,  # python text
    1,  # python_end
    0,  # output_start
    0,  # output text
    0,  # output_end
    1,  # assistant_end
    0,  # user_start
    0,  # user text
    0,  # user_end
    0,  # assistant_start
    1,  # assistant text
    1,  # assistant_end
]


class TestLossMaskConstruction:
    def test_render_wrapper_rejects_misaligned_token_and_mask_lengths(self) -> None:
        class _BadTokenizer:
            def render_conversation(self, _conversation):
                return [1, 2], [0]

        with pytest.raises(AssertionError, match="misaligned ids/mask lengths"):
            render_conversation_with_mask(_BadTokenizer(), {})

    def test_every_span_matches_hand_written_mask_position_by_position(self) -> None:
        tokenizer = _SpanTokenizer()
        ids, mask = render_conversation_with_mask(tokenizer, _all_span_conversation())

        assert len(ids) == len(EXPECTED_ALL_SPAN_MASK)
        assert len(mask) == len(ids)
        for position, expected in enumerate(EXPECTED_ALL_SPAN_MASK):
            assert mask[position] == expected, position

        special = tokenizer.SPECIAL
        assert mask[0] == 0  # bos
        assert all(mask[i] == 0 for i in range(1, 7))  # user markers + merged text
        assert mask[ids.index(special["<assistant_start>"])] == 0
        assert mask[8] == 1  # assistant text
        assert mask[ids.index(special["<python_start>"])] == 1
        assert mask[10] == 1  # python body
        assert mask[ids.index(special["<python_end>"])] == 1
        assert mask[ids.index(special["<output_start>"])] == 0
        assert mask[13] == 0  # output body
        assert mask[ids.index(special["<output_end>"])] == 0
        assistant_end_positions = [
            idx for idx, token_id in enumerate(ids) if token_id == special["<assistant_end>"]
        ]
        assert assistant_end_positions
        assert all(mask[idx] == 1 for idx in assistant_end_positions)

    def test_rendered_conversation_never_supervises_bos(self) -> None:
        assert render_conversation_with_mask(_SpanTokenizer(), _all_span_conversation())[1][0] == 0

    def test_supervised_target_count_excludes_position_zero(self) -> None:
        assert supervised_target_count([1, 0, 1, 1]) == 2

    def test_target_mask_uses_conv_mask_shifted_by_one(self) -> None:
        ids, mask = render_conversation_with_mask(_SpanTokenizer(), _all_span_conversation())
        targets = t.tensor([ids[1:]], dtype=t.int64)

        apply_completion_only_row_masks(targets, [mask])

        assert (targets[0] != -1).tolist() == [bool(value) for value in EXPECTED_ALL_SPAN_MASK[1:]]
        assert mask[0] == 0  # BOS is omitted, never a target

    def test_truncation_keeps_pair_aligned_at_every_cut_including_tool_call(self) -> None:
        ids, mask = render_conversation_with_mask(_SpanTokenizer(), _all_span_conversation())
        for cut in range(len(ids) + 2):
            truncated_ids, truncated_mask, original_len, was_truncated = truncate_conv_and_mask(
                ids, mask, cut
            )
            expected_len = min(cut, len(ids))
            assert len(truncated_ids) == len(truncated_mask) == expected_len, cut
            assert truncated_ids == ids[:cut]
            assert truncated_mask == mask[:cut]
            assert original_len == len(ids)
            assert was_truncated is (cut < len(ids))
        # cut=11 lands after the python body and before python_end.
        assert ids[9] == _SpanTokenizer.SPECIAL["<python_start>"]
        assert ids[11] == _SpanTokenizer.SPECIAL["<python_end>"]

    def test_packed_conversation_boundary_predicting_next_bos_is_masked(self) -> None:
        conv_a_ids, conv_a_mask = [1, 21, 22], [0, 1, 1]
        conv_b_ids, conv_b_mask = [1, 23, 24], [0, 1, 1]
        row = conv_a_ids + conv_b_ids
        row_mask = conv_a_mask + conv_b_mask
        targets = t.tensor([row[1:]], dtype=t.int64)

        apply_completion_only_row_masks(targets, [row_mask])

        boundary_target_position = len(conv_a_ids) - 1
        assert row[boundary_target_position + 1] == 1
        assert targets[0, boundary_target_position].item() == -1

    def test_tool_output_markers_and_text_have_zero_targets(self) -> None:
        tokenizer = _SpanTokenizer()
        ids, mask = render_conversation_with_mask(tokenizer, _all_span_conversation())
        targets = t.tensor([ids[1:]], dtype=t.int64)
        apply_completion_only_row_masks(targets, [mask])

        output_start = ids.index(tokenizer.SPECIAL["<output_start>"])
        output_end = ids.index(tokenizer.SPECIAL["<output_end>"])
        assert output_start <= output_end
        for token_position in range(output_start, output_end + 1):
            assert targets[0, token_position - 1].item() == -1


class _OtherMode(StrEnum):
    FULL = "full_sequence"


class TestLossMaskValidationAndResume:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("full_sequence", LossMaskMode.FULL_SEQUENCE),
            ("completion_only", LossMaskMode.COMPLETION_ONLY),
            (LossMaskMode.COMPLETION_ONLY, LossMaskMode.COMPLETION_ONLY),
            (_OtherMode.FULL, LossMaskMode.FULL_SEQUENCE),
        ],
    )
    def test_parse_accepts_strings_and_strenums(self, raw: object, expected: LossMaskMode) -> None:
        assert parse_loss_mask_mode(raw) is expected

    @pytest.mark.parametrize("raw", [None, 1, True, "assistant_only", ""])
    def test_parse_rejects_unknown_values_with_valid_set(self, raw: object) -> None:
        with pytest.raises(
            ValueError,
            match=r"sft_loss_mask must be one of \['completion_only', 'full_sequence'\]; got",
        ):
            parse_loss_mask_mode(raw)

    def test_resume_match_and_historical_full_sequence_are_accepted(self) -> None:
        assert_resume_sft_loss_mask_matches(
            LossMaskMode.COMPLETION_ONLY,
            {"sft_loss_mask": "completion_only"},
            checkpoint_dir="/checkpoint",
        )
        assert_resume_sft_loss_mask_matches(
            LossMaskMode.FULL_SEQUENCE,
            {},
            checkpoint_dir="/old-checkpoint",
        )
        assert_resume_sft_loss_mask_matches(
            LossMaskMode.FULL_SEQUENCE,
            None,
            checkpoint_dir="/old-checkpoint",
        )

    def test_resume_mode_change_and_old_to_completion_only_raise(self) -> None:
        with pytest.raises(ValueError, match="sft_loss_mask mismatch on resume"):
            assert_resume_sft_loss_mask_matches(
                LossMaskMode.FULL_SEQUENCE,
                {"sft_loss_mask": "completion_only"},
                checkpoint_dir="/checkpoint",
            )
        with pytest.raises(ValueError, match="sft_loss_mask mismatch on resume"):
            assert_resume_sft_loss_mask_matches(
                LossMaskMode.COMPLETION_ONLY,
                {},
                checkpoint_dir="/old-checkpoint",
            )


def _decoder_generator(
    task,
    tokenizer,
    mode: LossMaskMode,
    *,
    B: int = 1,
    T: int = 16,
    buffer_size: int = 4,
):
    return sft_data_generator_bos_bestfit_core(
        dataset=task,
        tokenizer=tokenizer,
        max_seq_len=T,
        device_batch_size=B,
        ddp_rank=0,
        ddp_world_size=1,
        device=t.device("cpu"),
        pin_memory=False,
        buffer_size=buffer_size,
        loss_mask_mode=mode,
    )


def _conversation(user_chars: int, assistant_chars: int) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "u" * user_chars},
            {"role": "assistant", "content": "a" * assistant_chars},
        ]
    }


class TestAllMaskedAndFilteringEdges:
    def test_all_masked_row_raises_named_failure_and_real_forward_is_finite(self) -> None:
        with pytest.raises(
            ValueError,
            match="completion_only produced an all-masked target row at index 0",
        ):
            assert_min_one_valid_target(t.full((1, 8), -1, dtype=t.int64))

        tokenizer = SFTMockTokenizer()
        task = TinyConversationTask([_conversation(3, 7)])
        inputs, targets, _ = next(_decoder_generator(task, tokenizer, LossMaskMode.COMPLETION_ONLY))
        model = _make_gpt(device="cpu")
        loss = model(inputs.long().contiguous(), targets=targets.contiguous())
        assert bool(t.isfinite(loss).all())
        assert bool(t.isnan(loss).any()) is False

    def test_head_truncated_before_assistant_is_skipped_only_in_completion_mode(self) -> None:
        tokenizer = SFTMockTokenizer()
        long_user = _conversation(40, 2)
        usable = _conversation(1, 4)
        task = TinyConversationTask([long_user, usable])
        long_ids, _ = tokenizer.render_conversation(long_user)

        with patch("steering.sft.iteration.logger.debug") as debug_log:
            full_inputs, _, _ = next(
                _decoder_generator(
                    task,
                    tokenizer,
                    LossMaskMode.FULL_SEQUENCE,
                    buffer_size=2,
                )
            )
            debug_log.assert_not_called()
        assert t.equal(full_inputs[0], t.tensor(long_ids[:16], dtype=t.int32))

        with patch("steering.sft.iteration.logger.debug") as debug_log:
            completion_inputs, completion_targets, _ = next(
                _decoder_generator(
                    task,
                    tokenizer,
                    LossMaskMode.COMPLETION_ONLY,
                    buffer_size=2,
                )
            )
        assert any(
            "sft_loss_mask=completion_only" in str(call.args[0])
            for call in debug_log.call_args_list
        )
        assert not t.equal(completion_inputs[0], full_inputs[0])
        assert bool((completion_targets != -1).any())

    def test_all_head_truncated_conversations_raise_decoder_starvation(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = TinyConversationTask([_conversation(40, 2)])
        with pytest.raises(RuntimeError, match="completion_only buffer starvation"):
            next(_decoder_generator(task, tokenizer, LossMaskMode.COMPLETION_ONLY))
        assert next(_decoder_generator(task, tokenizer, LossMaskMode.FULL_SEQUENCE)) is not None

    def test_min_loss_tokens_counts_three_supervised_among_two_hundred(self) -> None:
        tokenizer = SFTMockTokenizer()
        conversation = _conversation(196, 2)
        _, mask = tokenizer.render_conversation(conversation)
        assert len(mask) >= 200
        assert supervised_target_count(mask) == 3
        task = TinyConversationTask([conversation])

        starving = _make_conditioned_generator(
            task=task,
            tokenizer=tokenizer,
            B=1,
            T=256,
            min_loss_tokens=8,
            loss_mask_mode=LossMaskMode.COMPLETION_ONLY,
        )
        with pytest.raises(RuntimeError, match="supervised loss tokens"):
            with rank_synchronized_stage(
                "conditioned SFT training batch construction",
                device="cpu",
            ):
                next(starving)

        full = _make_conditioned_generator(
            task=task,
            tokenizer=tokenizer,
            B=1,
            T=256,
            min_loss_tokens=8,
            loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
        )
        assert next(full) is not None

    def test_starvation_collective_remains_outside_rank_local_refill(self) -> None:
        source = inspect.getsource(encdec_sft._make_sft_data_generator_impl)
        refill = source[source.index("    def refill_buffer()") : source.index("    use_cuda =")]
        assert "all_reduce" not in refill
        assert "rank_synchronized_stage(" in inspect.getsource(encdec_sft.main)


def _fp_conditioned_capture(
    mode: LossMaskMode,
) -> tuple[tuple, list[tuple[tuple[int, ...], tuple[int, ...]]]]:
    t.manual_seed(1286)
    tokenizer = SFTMockTokenizer()
    task = SingleTurnTask(n=6, user_chars=3, asst_chars=7, prefix="i")
    features = build_synthetic_features(
        task,
        tokenizer,
        n_segments_per_conv=2,
        features_per_segment=3,
    )
    threshold_cfg = ThresholdConfig(ThresholdType.segment_topk, 1000)
    percentile_table = (
        t.arange(1, 100, dtype=t.float64).unsqueeze(0).expand(int(N_SAE_FEATURES), -1).contiguous()
    )
    char_spans: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    real_routing = encdec_sft.build_full_sequence_segment_routing

    def capture_routing(**kwargs):
        seg_features = kwargs["seg_features"]
        char_spans.append((tuple(seg_features[2]), tuple(seg_features[3])))
        return real_routing(**kwargs)

    with (
        patch_load_sft_features(features),
        patch(
            "scripts.train.conditional_encdec_sft.build_full_sequence_segment_routing",
            side_effect=capture_routing,
        ),
    ):
        batch = next(
            make_sft_data_generator(
                task=task,
                tokenizer=tokenizer,
                B=2,
                T=48,
                device="cpu",
                sft_state=SFTState(),
                feature_dir="/dummy/features",
                parquet_dir="/dummy/parquet",
                threshold_cfg=threshold_cfg,
                boundaries=_make_boundaries(),
                min_loss_tokens=1,
                n_sae_features=int(N_SAE_FEATURES),
                fp_percentile_table=percentile_table,
                return_seg_doc_id=True,
                loss_mask_mode=mode,
                alignment_probe_samples=0,
                expected_split=SFTSplit.TRAIN,
            )
        )
    return batch, char_spans


class TestConditioningAndCrossTrainerInvariants:
    def test_i1_only_targets_change_and_every_conditioning_conjunct_is_equal(self) -> None:
        full, full_spans = _fp_conditioned_capture(LossMaskMode.FULL_SEQUENCE)
        completion, completion_spans = _fp_conditioned_capture(LossMaskMode.COMPLETION_ONLY)

        assert t.equal(full[0], completion[0]), "inputs"
        assert not t.equal(full[1], completion[1]), "targets must differ non-vacuously"
        assert t.equal(full[2], completion[2]), "enc_fids"
        assert t.equal(full[3], completion[3]), "enc_mags"
        assert t.equal(full[4], completion[4]), "enc_mask"
        assert t.equal(full[5], completion[5]), "seg_map"
        assert t.equal(full[6], completion[6]), "fp_targets"
        assert t.equal(full[7], completion[7]), "seg_doc_id"
        assert full_spans == completion_spans, "SegFeatures char spans"

    def test_i4_feature_prediction_loss_is_bit_identical_across_modes(self) -> None:
        full, _ = _fp_conditioned_capture(LossMaskMode.FULL_SEQUENCE)
        completion, _ = _fp_conditioned_capture(LossMaskMode.COMPLETION_ONLY)
        fp_config = FeaturePredictionConfig(
            weight=0.1,
            layer=-1,
            loss_mode=FeaturePredLossMode.FULL_SEGMENT,
            loss_fn=FeaturePredLossFn.BCE,
        )
        t.manual_seed(1286)
        model = build_tiny_model(
            n_sae_features=N_SAE_FEATURES,
            num_buckets=NUM_BUCKETS,
            feature_pred_config=fp_config,
            n_layer=2,
            n_embd=64,
            seq_len=64,
            xattn_frequency=1,
        )
        model.eval()

        def fp_loss(batch: tuple) -> t.Tensor:
            with t.no_grad():
                result = model(
                    batch[0],
                    batch[1],
                    batch[2],
                    batch[3],
                    batch[4],
                    batch[5],
                    fp_targets=batch[6],
                    seg_doc_id=batch[7],
                )
            return result.feature_pred_loss

        assert t.equal(fp_loss(full), fp_loss(completion))

    @pytest.mark.parametrize("mode", list(LossMaskMode))
    def test_cross_trainer_target_masks_agree_where_packing_agrees(
        self, mode: LossMaskMode
    ) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=6, user_chars=3, asst_chars=7, prefix="x")
        decoder_inputs, decoder_targets, _ = next(
            _decoder_generator(task, tokenizer, mode, B=2, T=48, buffer_size=100)
        )
        features = build_synthetic_features(task, tokenizer, n_segments_per_conv=2)
        with patch_load_sft_features(features):
            encdec_batch = next(
                _make_gen(
                    task,
                    tokenizer,
                    alignment_probe_samples=0,
                    B=2,
                    T=48,
                    loss_mask_mode=mode,
                )
            )

        assert t.equal(decoder_inputs, encdec_batch[0])
        assert t.equal(decoder_targets == -1, encdec_batch[1] == -1)
        assert t.equal(decoder_targets, encdec_batch[1])


def _literal_dict_keys(path: Path, variable: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        target = None
        value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and target.id == variable and isinstance(value, ast.Dict):
            return {
                key.value
                for key in value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    raise AssertionError(f"dict literal {variable!r} not found in {path}")


class TestConfigMetadataAndSharedWiring:
    def test_both_configs_default_off_and_modes_compose(self) -> None:
        for config_name in ("chat_sft", "conditional_encdec_sft"):
            with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
                default_cfg = compose(config_name=config_name)
                completion_cfg = compose(
                    config_name=config_name,
                    overrides=["sft_loss_mask=completion_only"],
                )
            assert default_cfg.sft_loss_mask == "full_sequence"
            assert completion_cfg.sft_loss_mask == "completion_only"

    def test_off_mode_meta_key_set_delta_is_exactly_sft_loss_mask(self) -> None:
        chat_before = {"step", "val_bpb", "model_config", "user_config"}
        chat_after = _literal_dict_keys(
            _REPO_ROOT / "scripts/train/chat_sft.py",
            "meta_dict",
        )
        assert chat_after - chat_before == {"sft_loss_mask"}
        assert chat_before - chat_after == set()

        encdec_before = {
            "step",
            "dataset",
            "user_config",
            "model_config",
            "encoder_config",
            "xattn_config",
            "num_buckets",
            "feature_prediction_config",
            "min_val_bpb",
            "loop_state",
        }
        encdec_after = _literal_dict_keys(
            _REPO_ROOT / "scripts/train/conditional_encdec_sft.py",
            "_sft_save_meta",
        )
        assert encdec_after - encdec_before == {"sft_loss_mask"}
        assert encdec_before - encdec_after == set()

    def test_i3_both_trainers_delegate_masking_to_shared_module(self) -> None:
        iteration_source = (_REPO_ROOT / "steering/sft/iteration.py").read_text(encoding="utf-8")
        encdec_source = (_REPO_ROOT / "scripts/train/conditional_encdec_sft.py").read_text(
            encoding="utf-8"
        )
        for source in (iteration_source, encdec_source):
            assert "from steering.sft.loss_mask import (" in source
            assert "render_conversation_with_mask(" in source
            assert "apply_completion_only_row_masks(" in source
            assert ".render_conversation(" not in source

    def test_both_trainers_parse_once_and_validation_reuses_training_mode(self) -> None:
        chat_source = (_REPO_ROOT / "scripts/train/chat_sft.py").read_text(encoding="utf-8")
        chat_tree = ast.parse(chat_source)
        chat_calls = [node for node in ast.walk(chat_tree) if isinstance(node, ast.Call)]
        parse_calls = [
            node
            for node in chat_calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "parse_loss_mask_mode"
            and node.args
            and ast.unparse(node.args[0]) == "cfg.sft_loss_mask"
        ]
        assert len(parse_calls) == 1
        split_calls = {
            ast.unparse(node.args[0])
            for node in chat_calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "sft_data_generator_bos_bestfit"
            and node.args
        }
        assert split_calls == {"DatasetSplit.TRAIN", "DatasetSplit.VAL"}
        wrapper = next(
            node
            for node in ast.walk(chat_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "sft_data_generator_bos_bestfit"
        )
        delegate = next(
            node
            for node in ast.walk(wrapper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sft_data_generator_bos_bestfit_core"
        )
        assert any(
            keyword.arg == "loss_mask_mode"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "loss_mask_mode"
            for keyword in delegate.keywords
        )

        main_tree = ast.parse(inspect.getsource(encdec_sft.main))
        main_calls = [node for node in ast.walk(main_tree) if isinstance(node, ast.Call)]
        encdec_parse_calls = [
            node
            for node in main_calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "parse_loss_mask_mode"
            and node.args
            and ast.unparse(node.args[0]) == "cfg.sft_loss_mask"
        ]
        assert len(encdec_parse_calls) == 1
        consumers = [
            node
            for node in main_calls
            if isinstance(node.func, ast.Name)
            and node.func.id in {"make_sft_data_generator", "_build_sft_val_loader"}
        ]
        assert {ast.unparse(node.func) for node in consumers} == {
            "make_sft_data_generator",
            "_build_sft_val_loader",
        }
        for consumer in consumers:
            assert any(
                keyword.arg == "loss_mask_mode"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "loss_mask_mode"
                for keyword in consumer.keywords
            )

    def test_only_non_training_render_mask_discards_are_exempt(self) -> None:
        discards: set[tuple[str, str]] = set()
        for root_name in ("scripts", "steering", "nanochat"):
            for path in (_REPO_ROOT / root_name).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for function in (
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ):
                    for node in ast.walk(function):
                        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                            continue
                        target = node.targets[0]
                        if (
                            not isinstance(target, (ast.Tuple, ast.List))
                            or len(target.elts) != 2
                            or not isinstance(target.elts[0], ast.Name)
                            or target.elts[0].id != "ids"
                            or not isinstance(target.elts[1], ast.Name)
                            or target.elts[1].id != "_"
                            or not isinstance(node.value, ast.Call)
                            or not isinstance(node.value.func, ast.Attribute)
                            or node.value.func.attr != "render_conversation"
                        ):
                            continue
                        discards.add(
                            (path.relative_to(_REPO_ROOT).as_posix(), function.name)
                        )
        assert discards == {
            ("scripts/preprocess/convert_sft_to_parquet.py", "_render_source_ids"),
            ("scripts/preprocess/count_tokens.py", "_count_tokens_for_task_list"),
            # Offline SAE fitting reduces the complete reference conversation;
            # it never computes a decoder training loss.
            ("scripts/eval/build_chat_benchmark_targets.py", "fitting_text"),
        }


class _ToolMixtureTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def render_conversation(self, conversation: dict) -> tuple[list[int], list[int]]:
        user = conversation["user"]
        assistant = conversation["assistant"]
        tool_call = conversation["tool_call"]
        tool_output = conversation["tool_output"]
        tail = conversation["tail"]
        mask = [0] * (1 + user) + [1] * assistant + [1] * tool_call + [0] * tool_output + [1] * tail
        ids = [0] + [2 + (position % 90) for position in range(len(mask) - 1)]
        return ids, mask


class _ToolMixtureDataset:
    def __init__(self, size: int = 512) -> None:
        self.rows = [
            {
                "user": 30 + (index * 37) % 140,
                "assistant": 8 + (index * 17) % 80,
                "tool_call": 5 + index % 20,
                "tool_output": 10 + (index * 13) % 60,
                "tail": 4 + (index * 7) % 30,
            }
            for index in range(size)
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def test_edge4_valid_count_variance_evidence() -> None:
    """Measure mean-reduction weighting variance over 64 realistic micro-batches."""
    generator = _decoder_generator(
        _ToolMixtureDataset(),
        _ToolMixtureTokenizer(),
        LossMaskMode.COMPLETION_ONLY,
        B=8,
        T=256,
        buffer_size=100,
    )
    valid_counts = [int((next(generator)[1] != -1).sum().item()) for _ in range(64)]
    mean = sum(valid_counts) / len(valid_counts)
    variance = sum((count - mean) ** 2 for count in valid_counts) / len(valid_counts)
    cv = math.sqrt(variance) / mean
    print(
        "completion_only valid-target evidence: "
        f"micro_batches={len(valid_counts)} mean={mean:.6f} cv={cv:.6f}"
    )
    assert len(valid_counts) >= 50
    assert mean > 0
    assert 0.0 < cv < 0.25


class _TinyCompiledLossModel(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = t.nn.Embedding(64, 16)
        self.head = t.nn.Linear(16, 64)

    def forward(self, inputs: t.Tensor, targets: t.Tensor) -> t.Tensor:
        logits = self.head(self.embed(inputs))
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-1,
        )


def test_edge6_dynamo_recompile_counters_equal_across_modes() -> None:
    inputs = t.arange(32, dtype=t.int64).reshape(2, 16) % 64
    full_targets = (inputs + 1) % 64
    completion_targets = full_targets.clone()
    completion_targets[:, ::2] = -1
    assert full_targets.shape == completion_targets.shape
    assert full_targets.dtype == completion_targets.dtype
    assert not t.equal(full_targets, completion_targets)

    t._dynamo.reset()
    t._dynamo.utils.counters.clear()
    compiled = t.compile(_TinyCompiledLossModel(), backend="eager", dynamic=False)
    full_loss = compiled(inputs, full_targets)
    after_full = {
        "frames_total": t._dynamo.utils.counters["frames"]["total"],
        "unique_graphs": t._dynamo.utils.counters["stats"]["unique_graphs"],
    }
    completion_loss = compiled(inputs, completion_targets)
    after_completion = {
        "frames_total": t._dynamo.utils.counters["frames"]["total"],
        "unique_graphs": t._dynamo.utils.counters["stats"]["unique_graphs"],
    }

    assert bool(t.isfinite(full_loss))
    assert bool(t.isfinite(completion_loss))
    assert after_completion == after_full
    assert after_full["unique_graphs"] == 1


@pytest.mark.parametrize("mode", list(LossMaskMode))
def test_decoder_only_tiny_cpu_overfit_decreases_in_each_mode(mode: LossMaskMode) -> None:
    t.manual_seed(1286)
    tokenizer = SFTMockTokenizer()
    task = SingleTurnTask(n=4, user_chars=3, asst_chars=7, prefix="o")
    inputs, targets, _ = next(_decoder_generator(task, tokenizer, mode, B=2, T=16))
    model = _make_gpt(device="cpu")
    model.train()
    optimizer = t.optim.AdamW(model.parameters(), lr=0.02)
    losses: list[float] = []
    for _ in range(16):
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs.long().contiguous(), targets=targets.contiguous())
        assert bool(t.isfinite(loss))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    assert losses[-1] < losses[0], (mode, losses)
