"""Tests for the configurable encoder-decoder SFT loss-token floor."""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from loguru import logger as loguru_logger

from scripts.train.conditional_encdec_sft import (
    _build_sft_val_loader,
    _configure_min_loss_tokens,
    main,
)
from steering.sft.loss_mask import LossMaskMode
from tests.steering.test_sft_conv_truncation import (
    SFTMockTokenizer,
    TinyConversationTask,
    _make_conditioned_generator,
)

CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs", "training")
)
HARDWARE_VARIANTS = (
    "conditional_encdec_sft_4090",
    "conditional_encdec_sft_2xh200",
    "conditional_encdec_sft_4xh200",
    "conditional_encdec_sft_2xb200",
    "conditional_encdec_sft_profile_4090",
)


@contextmanager
def _loguru_to_caplog(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    """Bridge loguru records into pytest's standard-library capture."""
    handler_id = loguru_logger.add(
        lambda msg: caplog.handler.emit(
            logging.LogRecord(
                name=msg.record["name"] or "loguru",
                level=msg.record["level"].no,
                pathname=msg.record["file"].path,
                lineno=msg.record["line"],
                msg=msg.record["message"],
                args=(),
                exc_info=None,
            )
        ),
        level=0,
    )
    try:
        yield
    finally:
        loguru_logger.remove(handler_id)


class TestMinLossTokensConfig:
    def test_default_is_eight(self, hydra_compose) -> None:
        cfg = hydra_compose(CONFIG_DIR, "conditional_encdec_sft")
        assert cfg.min_loss_tokens == 8

    def test_override_composes(self, hydra_compose) -> None:
        cfg = hydra_compose(
            CONFIG_DIR,
            "conditional_encdec_sft",
            ["min_loss_tokens=13"],
        )
        assert cfg.min_loss_tokens == 13

    @pytest.mark.parametrize("config_name", HARDWARE_VARIANTS)
    def test_hardware_variant_inherits_eight(self, hydra_compose, config_name: str) -> None:
        cfg = hydra_compose(CONFIG_DIR, config_name)
        assert cfg.min_loss_tokens == 8


class TestConfigureMinLossTokens:
    @pytest.mark.parametrize("value", [8, 13])
    def test_positive_int_returns_int(self, value: int) -> None:
        result = _configure_min_loss_tokens(value)
        assert result == value
        assert isinstance(result, int)

    @pytest.mark.parametrize("bad_value", [0, -1])
    def test_non_positive_is_rejected(self, bad_value: int) -> None:
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match="min_loss_tokens|PositiveInt",
        ):
            _configure_min_loss_tokens(bad_value)

    @pytest.mark.parametrize("bad_value", [True, False])
    def test_bool_is_rejected(self, bad_value: bool) -> None:
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match="min_loss_tokens|PositiveInt",
        ):
            _configure_min_loss_tokens(bad_value)

    def test_float_is_rejected(self) -> None:
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match="min_loss_tokens|PositiveInt",
        ):
            _configure_min_loss_tokens(8.5)  # ty: ignore[invalid-argument-type]

    def test_info_log_records_value(self, caplog) -> None:
        with _loguru_to_caplog(caplog):
            with caplog.at_level(logging.INFO):
                _configure_min_loss_tokens(13)

        info_records = [
            record
            for record in caplog.records
            if record.levelno == logging.INFO and "min_loss_tokens=13" in record.message
        ]
        assert info_records, "helper must log the configured min_loss_tokens value"


class TestMinLossTokensWiring:
    def test_val_loader_forwards_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_generator(**kwargs):
            captured.update(kwargs)
            return iter(())

        monkeypatch.setattr(
            "scripts.train.conditional_encdec_sft.make_sft_data_generator",
            fake_generator,
        )
        loader = _build_sft_val_loader(
            val_task=object(),
            tokenizer=object(),
            B=2,
            T=16,
            device="cpu",
            ddp_rank=0,
            ddp_world_size=1,
            feature_dir="/features",
            expected_feature_dir="/features",
            parquet_dir="/parquets",
            threshold_cfg=object(),
            boundaries=object(),
            precomputed_thresholds=None,
            sampling_bundle=None,
            loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
            min_loss_tokens=13,
        )
        assert next(loader, None) is None
        assert captured["min_loss_tokens"] == 13

    def test_main_validates_once_and_uses_same_local_for_train_and_val(self) -> None:
        source = inspect.getsource(main)
        assert "_configure_min_loss_tokens(cfg.min_loss_tokens)" in source
        assert source.count("min_loss_tokens=min_loss_tokens") == 2

        train_start = source.index("train_gen = make_sft_data_generator(")
        train_end = source.index("\n    )", train_start)
        assert "min_loss_tokens=min_loss_tokens" in source[train_start:train_end]

        val_start = source.index("def build_val_loader()")
        val_end = source.index("\n\n    token_bytes", val_start)
        assert "min_loss_tokens=min_loss_tokens" in source[val_start:val_end]


def _conversation(user_chars: int, assistant_chars: int) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "u" * user_chars},
            {"role": "assistant", "content": "a" * assistant_chars},
        ]
    }


class TestMinLossTokensBehavior:
    def test_starvation_names_config_knob_and_low_threshold_proceeds(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = TinyConversationTask([_conversation(1, 1)])

        starving = _make_conditioned_generator(
            task=task,
            tokenizer=tokenizer,
            B=1,
            T=32,
            min_loss_tokens=5,
        )
        with pytest.raises(RuntimeError, match="min_loss_tokens config"):
            next(starving)

        proceeds = _make_conditioned_generator(
            task=task,
            tokenizer=tokenizer,
            B=1,
            T=32,
            min_loss_tokens=1,
        )
        assert next(proceeds) is not None

    def test_first_pass_summary_reports_skips_once(self, caplog) -> None:
        tokenizer = SFTMockTokenizer()
        task = TinyConversationTask(
            [
                _conversation(1, 1),
                _conversation(3, 3),
                _conversation(1, 1),
                _conversation(3, 3),
            ]
        )
        generator = _make_conditioned_generator(
            task=task,
            tokenizer=tokenizer,
            B=1,
            T=32,
            min_loss_tokens=6,
        )

        with _loguru_to_caplog(caplog):
            with caplog.at_level(logging.DEBUG):
                next(generator)

        summaries = [
            record for record in caplog.records if "SFT buffer first pass:" in record.message
        ]
        # The repository-wide loguru pytest sink and the temporary bridge
        # both forward the same record into caplog; deduplicate that bridge
        # artifact before asserting the helper emitted exactly once.
        assert {record.message for record in summaries} == {
            "SFT buffer first pass: skipped 2/4 conversations with <6 supervised loss "
            "tokens after truncation (min_loss_tokens=6)"
        }
        assert "2/4" in summaries[0].message
        assert "min_loss_tokens=6" in summaries[0].message
        assert summaries[0].levelno == logging.WARNING

    def test_first_pass_summary_with_no_skips_is_info(self, caplog) -> None:
        tokenizer = SFTMockTokenizer()
        task = TinyConversationTask([_conversation(3, 3), _conversation(3, 3)])
        generator = _make_conditioned_generator(
            task=task,
            tokenizer=tokenizer,
            B=1,
            T=32,
            min_loss_tokens=1,
        )

        with _loguru_to_caplog(caplog):
            with caplog.at_level(logging.DEBUG):
                next(generator)

        summaries = [
            record for record in caplog.records if "SFT buffer first pass:" in record.message
        ]
        assert {record.message for record in summaries} == {
            "SFT buffer first pass: skipped 0/2 conversations with <1 supervised loss "
            "tokens after truncation (min_loss_tokens=1)"
        }
        assert "0/2" in summaries[0].message
        assert summaries[0].levelno == logging.INFO
