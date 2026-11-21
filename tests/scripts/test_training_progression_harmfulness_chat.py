"""Decoder chat rendering and refusal of retired static encoder-decoder wrappers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch as t
from omegaconf import OmegaConf

from scripts.train.training_progression import run_harmfulness_eval
from steering.eval.harmfulness_judge import HARM_CATEGORIES, ComplianceJudgeResult

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _complete_metrics(num_harmful: int, num_benign: int) -> dict:
    total = num_harmful + num_benign
    return {
        "overall": {
            "mean_score": 1.0,
            "harmfulness_rate": 0.0,
            "safe_response_rate": 1.0,
            "num_scored": total,
            "num_failed": 0,
        },
        "harmful_prompts": {
            "mean_score": 1.0 if num_harmful else 0.0,
            "harmfulness_rate": 0.0,
            "safe_response_rate": 1.0 if num_harmful else 0.0,
            "asr": 0.0,
            "num_scored": num_harmful,
            "num_failed": 0,
        },
        "benign_prompts": {
            "mean_score": 1.0 if num_benign else 0.0,
            "harmfulness_rate": 0.0,
            "safe_response_rate": 1.0 if num_benign else 0.0,
            "num_scored": num_benign,
            "num_failed": 0,
        },
        "per_category": dict.fromkeys(HARM_CATEGORIES, 0.0),
    }


# ---------------------------------------------------------------------------
# Target-aware capability gate
# ---------------------------------------------------------------------------


class _BareTokenizer:
    """Tokenizer that lacks render_for_completion — simulates what a
    TransformersTokenizer or HuggingFaceTokenizer would look like to
    the target-aware branch."""

    def get_bos_token_id(self) -> int:
        return 0

    def encode(self, text, prepend=None, append=None):
        return [0] + [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i not in (0,))


class _RustbpeLikeTokenizer(_BareTokenizer):
    """Tokenizer that DOES have render_for_completion — simulates RustBPE."""

    def render_for_completion(self, conversation: dict) -> list[int]:
        text = conversation["messages"][0]["content"]
        return [0] + [ord(c) for c in text] + [2]  # 2 = sentinel for <assistant_start>

    def encode_special(self, token: str) -> int:
        assert token == "<assistant_end>"
        return 3


@pytest.mark.parametrize("tokenizer", [_BareTokenizer(), _RustbpeLikeTokenizer()])
def test_static_encoder_decoder_wrapper_refused_before_generation(tokenizer):
    model = MagicMock(is_target_aware_encdec=True)
    with patch("scripts.train.training_progression.InferenceEngine") as engine:
        with pytest.raises(ValueError, match="legacy static wrappers are unsupported"):
            run_harmfulness_eval(model, tokenizer, t.device("cpu"), seed=0, cfg=OmegaConf.create({}))
    engine.assert_not_called()


class TestNonTargetAwarePathPassesChatFlag:
    """The non-target-aware branch must call ``engine.generate_batch``
    with ``render_as_chat=True``. Covered by a source grep in
    test_inference_engine_chat_template.py, reinforced here by
    inspecting the actual kwargs."""

    def test_generate_batch_called_with_render_as_chat_true(self) -> None:
        model = MagicMock()
        model.is_target_aware_encdec = False

        tokenizer = _RustbpeLikeTokenizer()
        device = t.device("cpu")
        cfg = OmegaConf.create(
            {
                "harm_temperature": 0.0,
                "harm_top_k": 0,
                "harm_max_new_tokens": 8,
                "harm_num_benign": 2,
                "harm_num_harmful": 1,
                "harm_judge_model": "test",
            }
        )

        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
            patch("steering.eval.harmfulness_judge.load_harmful_prompts") as mock_harmful,
            patch("steering.eval.harmfulness_judge.load_benign_prompts") as mock_benign,
            patch("scripts.train.training_progression.InferenceEngine") as mock_engine_cls,
            patch("steering.eval.harmfulness_judge.JudgeClient") as mock_judge_cls,
            patch("steering.eval.harmfulness_judge.compute_metrics") as mock_metrics,
        ):
            mock_harmful.return_value = [{"prompt": "h1", "category": "harm"}]
            mock_benign.return_value = [
                {"prompt": "b1", "category": "benign"},
                {"prompt": "b2", "category": "benign"},
            ]
            mock_engine = MagicMock()
            mock_engine.generate_batch.return_value = ["r1", "r2", "r3"]
            mock_engine_cls.return_value = mock_engine
            mock_judge = MagicMock()
            mock_judge.score_all.return_value = [
                {"score": 1, "categories": [], "reasoning": ""}
            ] * 3
            mock_judge.score_compliance_all.return_value = [
                ComplianceJudgeResult(5, False, "Complete answer.")
            ] * 2
            mock_judge_cls.return_value = mock_judge
            mock_metrics.return_value = _complete_metrics(1, 2)

            run_harmfulness_eval(model, tokenizer, device, seed=42, cfg=cfg)

            # Inspect the kwargs of the (single) generate_batch call:
            assert mock_engine.generate_batch.call_count == 1
            _, kwargs = mock_engine.generate_batch.call_args
            assert kwargs.get("render_as_chat") is True, (
                "run_harmfulness_eval must pass render_as_chat=True to "
                "InferenceEngine.generate_batch so progression metrics "
                "match standalone harmfulness_eval.py on the same model"
            )

    def test_multibatch_seed_schedule_uses_absolute_prompt_offsets(self) -> None:
        model = MagicMock()
        model.is_target_aware_encdec = False
        tokenizer = _RustbpeLikeTokenizer()
        cfg = OmegaConf.create(
            {
                "harm_temperature": 0.7,
                "harm_top_k": 50,
                "harm_max_new_tokens": 8,
                "harm_num_benign": 8,
                "harm_num_harmful": 9,
                "harm_judge_model": "test",
            }
        )
        harmful = [
            {"prompt": f"h{index}", "category": "harm"} for index in range(9)
        ]
        benign = [
            {"prompt": f"b{index}", "category": "benign"} for index in range(8)
        ]

        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
            patch(
                "steering.eval.harmfulness_judge.load_harmful_prompts",
                return_value=harmful,
            ),
            patch(
                "steering.eval.harmfulness_judge.load_benign_prompts",
                return_value=benign,
            ),
            patch("scripts.train.training_progression.InferenceEngine") as engine_cls,
            patch("steering.eval.harmfulness_judge.JudgeClient") as judge_cls,
            patch(
                "steering.eval.harmfulness_judge.compute_metrics",
                return_value=_complete_metrics(9, 8),
            ),
        ):
            engine = MagicMock()
            engine.generate_batch.side_effect = lambda prompts, **_kwargs: [
                f"response-{prompt}" for prompt in prompts
            ]
            engine_cls.return_value = engine
            judge = MagicMock()
            judge.score_all.return_value = [
                {"score": 1, "categories": [], "reasoning": ""}
            ] * 17
            judge.score_compliance_all.return_value = [
                ComplianceJudgeResult(5, False, "Complete answer.")
            ] * 8
            judge_cls.return_value = judge

            run_harmfulness_eval(model, tokenizer, t.device("cpu"), seed=42, cfg=cfg)

        assert [call.kwargs["seed"] for call in engine.generate_batch.call_args_list] == [
            42,
            50,
            58,
        ]
        effective = [
            call.kwargs["seed"] + offset
            for call in engine.generate_batch.call_args_list
            for offset in range(len(call.args[0]))
        ]
        assert effective == list(range(42, 59))
