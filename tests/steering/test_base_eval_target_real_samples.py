"""Regression coverage for base_eval's shared target-real sample builder."""

from __future__ import annotations

from types import SimpleNamespace

import scripts.eval.base_eval as base_eval


class _RawTokenizer:
    """Small non-chat tokenizer exercising base_eval's raw fallback branch."""

    def get_bos_token_id(self) -> int:
        return 0

    def encode(self, text: str, prepend: int | None = None) -> list[int]:
        ids = [ord(char) for char in text]
        return ([prepend] if prepend is not None else []) + ids

    def decode(self, token_ids) -> str:
        return "".join(chr(int(token)) for token in token_ids if int(token) != 0)


class _RecordingEngine:
    def __init__(self, wrapper, tokenizer, device) -> None:
        self.wrapper = wrapper
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt: str, **kwargs) -> str:
        assert prompt == "raw prompt"
        assert kwargs["render_as_chat"] is False
        return "raw response"


def test_raw_target_real_samples_pass_the_full_mask_to_shared_builder(monkeypatch) -> None:
    tokenizer = _RawTokenizer()
    cfg = SimpleNamespace(
        sample_prompts=["raw prompt"],
        sampling=SimpleNamespace(
            per_config_max_new_tokens=7,
            per_config_temperature=0.0,
        ),
    )
    trc = SimpleNamespace(name="raw-target")
    captured: dict[str, object] = {}

    def record_builder(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(wrapper=object())

    monkeypatch.setattr(base_eval, "build_target_real_artifacts_for_example", record_builder)
    monkeypatch.setattr(base_eval, "InferenceEngine", _RecordingEngine)

    responses = base_eval._run_sample_target_real_eval(
        raw_encdec=object(),
        tokenizer=tokenizer,
        device="cpu",
        trc=trc,
        provider=object(),
        cfg=cfg,
    )

    assert responses == ["raw response"]
    # The raw branch has no tokenizer-provided render mask. Its explicit
    # autoregressive loss mask is therefore the exact equivalent full mask
    # required by the shared builder and must not be left undefined.
    assert captured["full_token_ids"] == [0, ord("r"), ord("a"), ord("w"), ord(" "), ord("p"), ord("r"), ord("o"), ord("m"), ord("p"), ord("t")]
    assert captured["mask"] == [0] + [1] * 10
    assert captured["target_token_loss_mask"] == [0] + [1] * 10
    assert captured["context_token_ids"] == captured["full_token_ids"]
    assert captured["target_char_range"] == (0, 10)
