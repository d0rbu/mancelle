"""Content-bound conditioned-SFT alignment contracts."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from loguru import logger
from safetensors.torch import save_file

import nanochat.flash_attention as _fa_mod
from errors import DataLoadError
from nanochat.flash_attention import FlashAttnImpl
from scripts.preprocess.convert_sft_to_parquet import (
    convert_mixture_to_parquet,
    prepare_messages_for_full_nanochat_render,
    render_conversation_to_full_nanochat_text,
)
from scripts.train.conditional_encdec_sft import SFTState, _build_sft_val_loader
from steering.compute_statistics import compute_statistics
from steering.conditional.encoder_decoder.dataloader import SegFeatures
from steering.conditional.encoder_decoder.full_sequence_routing import (
    compute_full_coverage_gap,
    source_aligned_char_offsets,
)
from steering.conditional.encoder_decoder.sft_features import load_sft_features
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.extract_features import (
    _assert_existing_output_dir_compatible,
    _load_sft_conversion_fingerprint_json,
    run_extraction_loop,
)
from steering.feature_io import (
    glob_features,
    load_features,
    read_shard_metadata,
    save_features,
)
from steering.refined_types import NonNegativeInt, PositiveInt
from steering.segmentation import SegmentMode
from steering.sft.loss_mask import LossMaskMode, render_conversation_with_mask
from steering.sft.mixture_fingerprint import (
    MIXTURE_FINGERPRINT_KEY,
    SFT_MIXTURE_FINGERPRINT_METADATA_KEY,
    MixtureFingerprint,
    compute_mixture_fingerprint,
    mixture_fingerprint_json,
)
from steering.sft.turn_spans import ChatSpecialTokens, legacy_turn_spans
from steering.sft_schema import SFTSplit
from steering.stats_io import load_statistics_metadata
from tasks.common import Task, TaskMixture
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    setup_optimizer_simple,
)
from tests.nanochat.test_structured_conversation_renderer import _CharacterTokenizer
from tests.steering.goldens.gen_sft_conditioning_alignment_golden import (
    GOLDEN_PATH,
    ORACLE_COMMIT,
    build_capture,
)
from tests.steering.test_encdec_sft_conditioned_integration import (
    SFTMockTokenizer,
    SingleTurnTask,
    TrailingUserTask,
    _build_tiny_encdec_matching_sft,
    _make_gen,
    _run_one_training_step,
    build_synthetic_features,
    patch_load_sft_features,
)
from tests.steering.test_extract_features_integration import (
    TinyHookedTransformer,
    TinySAE,
)
from tests.steering.test_extract_features_loop import (
    _staging,
    make_parquet,
    tiny_params,
)
from tests.steering.test_sft_features import _write_parquet, _write_shard

_DTYPE_BY_NAME = {
    "torch.bool": t.bool,
    "torch.float32": t.float32,
    "torch.int32": t.int32,
    "torch.int64": t.int64,
}


class _CompactCharacterTokenizer(_CharacterTokenizer):
    """Character oracle whose ordinary-token ids fit the tiny SFT model."""

    def __init__(self) -> None:
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for character in text:
            if character not in self._char_to_id:
                token_id = 10 + len(self._char_to_id)
                assert token_id < 100
                self._char_to_id[character] = token_id
                self._id_to_char[token_id] = character
            ids.append(self._char_to_id[character])
        return ids

    def decode(self, token_ids) -> str:
        return "".join(
            self._decoded_specials[token_id]
            if token_id in self._decoded_specials
            else self._id_to_char[token_id]
            for token_id in token_ids
        )


class _AlphaTask(SingleTurnTask):
    pass


class _BetaTask(SingleTurnTask):
    pass


class _VariableLengthTask(SingleTurnTask):
    def get_example(self, index: int) -> dict:
        conversation = super().get_example(index)
        conversation["messages"][-1]["content"] += "." * index
        return conversation


class _UnequalTurnTask(Task):
    """Two rows with different turn counts and within-turn chunk geometry."""

    _CONTENT_LENGTHS = (
        (2, 7, 3, 4, 11, 2),
        (4, 3, 9, 2, 2, 8, 4, 5),
    )

    def num_examples(self) -> int:
        return len(self._CONTENT_LENGTHS)

    def get_example(self, index: int) -> dict:
        lengths = self._CONTENT_LENGTHS[index]
        return {
            "messages": [
                {
                    "role": "user" if message_index % 2 == 0 else "assistant",
                    "content": chr(ord("a") + message_index) * content_length,
                }
                for message_index, content_length in enumerate(lengths)
            ]
        }


class _UserOnlyTask(Task):
    """A task whose rows the converter would skip entirely (``no_assistant``)."""

    def __init__(self, n: int, **task_kwargs) -> None:
        super().__init__(**task_kwargs)
        self._n = n

    def num_examples(self) -> int:
        return self._n

    def get_example(self, index: int) -> dict:
        return {"messages": [{"role": "user", "content": f"u{index:03d}"}]}


def _mixture(task_type, *, n: int = 4, prefix: str = "row") -> TaskMixture:
    return TaskMixture([task_type(n=n, user_chars=4, asst_chars=8, prefix=prefix)])


def _render_texts(task, tokenizer) -> dict[int, str]:
    texts: dict[int, str] = {}
    for index in range(len(task)):
        text, skip_reason = render_conversation_to_full_nanochat_text(
            tokenizer,
            task[index],
        )
        assert skip_reason is None
        assert text is not None
        texts[index] = text
    return texts


def _trimmed_render(tokenizer, conversation) -> tuple[list[int], list[int]]:
    """Render exactly the slice ``convert_sft_to_parquet`` writes to parquet."""
    messages, skip_reason = prepare_messages_for_full_nanochat_render(conversation)
    assert messages is not None, f"conversation unexpectedly skipped: {skip_reason}"
    return render_conversation_with_mask(tokenizer, {"messages": messages})


def _fingerprint(
    task,
    *,
    split: SFTSplit = SFTSplit.TRAIN,
) -> MixtureFingerprint:
    return compute_mixture_fingerprint(
        task,
        mixture_name="capabilities",
        split=split,
    )


def _conversion_meta(
    fingerprint: MixtureFingerprint,
    *,
    split: SFTSplit = SFTSplit.TRAIN,
    mixture: str = "capabilities",
) -> dict[str, object]:
    """Build complete conversion metadata around a fingerprint fixture."""
    return {
        "mixture": mixture,
        "split": split.value,
        "format": "full_nanochat_text",
        "completed_shards": 1,
        "total_shards": 1,
        "total_conversations": fingerprint["len_mixture"],
        MIXTURE_FINGERPRINT_KEY: fingerprint,
    }


def test_aligned_pair_batch_is_byte_identical_to_pre_1285_golden() -> None:
    """Correct alignment must leave every yielded tensor byte-identical."""
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert golden["oracle_commit"] == ORACLE_COMMIT
    actual = build_capture()
    assert len(actual["batch"]) == len(golden["batch"])
    for field_index, (actual_record, expected_record) in enumerate(
        zip(actual["batch"], golden["batch"], strict=True)
    ):
        if expected_record is None:
            assert actual_record is None, field_index
            continue
        assert actual_record is not None, field_index
        assert actual_record["sha256"] == expected_record["sha256"], field_index
        dtype = _DTYPE_BY_NAME[expected_record["dtype"]]
        assert t.equal(
            t.tensor(actual_record["values"], dtype=dtype),
            t.tensor(expected_record["values"], dtype=dtype),
        ), field_index


def test_fingerprint_is_stable_and_sensitive_to_length_and_task_order() -> None:
    alpha = _AlphaTask(n=2, user_chars=4, asst_chars=8, prefix="a")
    beta = _BetaTask(n=3, user_chars=4, asst_chars=8, prefix="b")
    original = TaskMixture([alpha, beta])
    identical = TaskMixture(
        [
            _AlphaTask(n=2, user_chars=4, asst_chars=8, prefix="other-a"),
            _BetaTask(n=3, user_chars=4, asst_chars=8, prefix="other-b"),
        ]
    )
    length_changed = TaskMixture(
        [
            _AlphaTask(n=4, user_chars=4, asst_chars=8, prefix="a"),
            _BetaTask(n=3, user_chars=4, asst_chars=8, prefix="b"),
        ]
    )
    reordered = TaskMixture([beta, alpha])

    original_fingerprint = _fingerprint(original)
    assert original_fingerprint == _fingerprint(identical)
    assert original_fingerprint != _fingerprint(length_changed)
    assert original_fingerprint != _fingerprint(reordered)
    assert (
        original_fingerprint["index_map_sha256"] != _fingerprint(length_changed)["index_map_sha256"]
    )


def test_store_from_mixture_a_with_task_b_raises_on_fingerprint_guard() -> None:
    tokenizer = SFTMockTokenizer()
    mixture_a = _mixture(_AlphaTask, prefix="a")
    mixture_b = _mixture(_BetaTask, prefix="b")
    features = build_synthetic_features(mixture_a, tokenizer)
    fingerprint_a = _fingerprint(mixture_a)

    with (
        patch_load_sft_features(
            features,
            enforce_fingerprints=True,
            split=SFTSplit.TRAIN,
            mixture="capabilities",
            parquet_fingerprint=fingerprint_a,
            feature_fingerprint=fingerprint_a,
        ),
        pytest.raises(
            DataLoadError,
            match=r"task mixture fingerprint mismatch",
        ),
    ):
        next(_make_gen(mixture_b, tokenizer, alignment_probe_samples=0))


def test_val_composition_mismatch_raises_even_when_all_val_indices_are_present() -> None:
    """Both artifacts say val; only the live composition differs from the store."""
    tokenizer = SFTMockTokenizer()
    train_composition = _mixture(_AlphaTask, prefix="train")
    val_composition = _mixture(_BetaTask, prefix="val")
    features = build_synthetic_features(train_composition, tokenizer)
    assert set(features) == set(range(len(val_composition)))
    store_fingerprint = _fingerprint(train_composition, split=SFTSplit.VAL)

    with (
        patch_load_sft_features(
            features,
            enforce_fingerprints=True,
            split=SFTSplit.VAL,
            mixture="capabilities",
            parquet_fingerprint=store_fingerprint,
            feature_fingerprint=store_fingerprint,
        ),
        pytest.raises(
            DataLoadError,
            match=r"task mixture fingerprint mismatch",
        ),
    ):
        next(
            _make_gen(
                val_composition,
                tokenizer,
                expected_split=SFTSplit.VAL,
                alignment_probe_samples=0,
            )
        )


def test_content_probe_catches_forged_matching_fingerprint_and_shifted_text() -> None:
    tokenizer = SFTMockTokenizer()
    task = _mixture(_VariableLengthTask, n=6, prefix="probe")
    features = build_synthetic_features(task, tokenizer)
    fingerprint = _fingerprint(task)
    aligned_texts = _render_texts(task, tokenizer)
    shifted_texts = {index: aligned_texts[(index + 1) % len(task)] for index in range(len(task))}

    with (
        patch_load_sft_features(
            features,
            enforce_fingerprints=True,
            split=SFTSplit.TRAIN,
            mixture="capabilities",
            parquet_fingerprint=fingerprint,
            feature_fingerprint=fingerprint,
            texts=shifted_texts,
        ),
        pytest.raises(
            DataLoadError,
            match=r"content alignment probe mismatch at original_index=0",
        ),
    ):
        next(_make_gen(task, tokenizer, alignment_probe_samples=8))


def test_probe_indices_are_identical_across_real_pythonhashseed_subprocesses() -> None:
    code = (
        "import json; "
        "from steering.sft.mixture_fingerprint import sample_alignment_probe_indices; "
        "print(json.dumps(sample_alignment_probe_indices(17, 8)))"
    )
    outputs: list[str] = []
    for seed in ("0", "1", "8675309"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(completed.stdout.strip())
    assert outputs == ["[0, 2, 4, 6, 9, 11, 13, 16]"] * 3


def test_zero_samples_disables_probe_but_not_fingerprint_checks() -> None:
    tokenizer = SFTMockTokenizer()
    task = _mixture(_AlphaTask, n=4, prefix="zero")
    other = _mixture(_BetaTask, n=4, prefix="other")
    features = build_synthetic_features(task, tokenizer)
    fingerprint = _fingerprint(task)
    forged_texts = dict.fromkeys(range(len(task)), "forged")

    with patch_load_sft_features(
        features,
        enforce_fingerprints=True,
        split=SFTSplit.TRAIN,
        mixture="capabilities",
        parquet_fingerprint=fingerprint,
        feature_fingerprint=fingerprint,
        texts=forged_texts,
    ):
        assert next(_make_gen(task, tokenizer, alignment_probe_samples=0)) is not None

    with (
        patch_load_sft_features(
            features,
            enforce_fingerprints=True,
            split=SFTSplit.TRAIN,
            mixture="capabilities",
            parquet_fingerprint=fingerprint,
            feature_fingerprint=fingerprint,
        ),
        pytest.raises(
            DataLoadError,
            match=r"task mixture fingerprint mismatch",
        ),
    ):
        next(_make_gen(other, tokenizer, alignment_probe_samples=0))


def test_legacy_absent_fingerprints_warn_once_and_probe_still_runs() -> None:
    tokenizer = SFTMockTokenizer()
    task = _mixture(_AlphaTask, n=4, prefix="legacy")
    features = build_synthetic_features(task, tokenizer)
    warnings: list[str] = []
    sink_id = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        with patch_load_sft_features(
            features,
            enforce_fingerprints=True,
            split=SFTSplit.TRAIN,
            mixture="capabilities",
            texts=_render_texts(task, tokenizer),
        ):
            assert next(_make_gen(task, tokenizer, alignment_probe_samples=2)) is not None
    finally:
        logger.remove(sink_id)
    matching = [warning for warning in warnings if "legacy escape hatch" in warning]
    assert len(matching) == 1
    assert "rests on the independent startup content probe alone" in matching[0]
    assert "convert_sft_to_parquet mixture=capabilities split=train" in matching[0]
    assert "output_dir=out/data/sft_data/capabilities_fingerprinted" in matching[0]


def test_legacy_absent_fingerprints_with_zero_probe_raises() -> None:
    tokenizer = SFTMockTokenizer()
    task = _mixture(_AlphaTask, n=4, prefix="legacy-zero")
    features = build_synthetic_features(task, tokenizer)

    with (
        patch_load_sft_features(
            features,
            enforce_fingerprints=True,
            split=SFTSplit.TRAIN,
            mixture="capabilities",
        ),
        pytest.raises(
            DataLoadError,
            match=(
                r"legacy SFT artifacts .*no mixture fingerprint on either side.*"
                r"alignment_probe_samples=0.*no alignment verification.*"
                r"regenerate.*fingerprint.*alignment_probe_samples > 0"
            ),
        ),
    ):
        next(_make_gen(task, tokenizer, alignment_probe_samples=0))


def test_one_sided_fingerprint_raises_as_half_migrated() -> None:
    tokenizer = SFTMockTokenizer()
    task = _mixture(_AlphaTask)
    features = build_synthetic_features(task, tokenizer)

    with (
        patch_load_sft_features(
            features,
            enforce_fingerprints=True,
            split=SFTSplit.TRAIN,
            mixture="capabilities",
            parquet_fingerprint=_fingerprint(task),
            feature_fingerprint=None,
        ),
        pytest.raises(
            DataLoadError,
            match=r"present on exactly one side .*half-migrated state is not legacy",
        ),
    ):
        next(_make_gen(task, tokenizer, alignment_probe_samples=0))


def test_parquet_feature_fingerprint_mismatch_raises_with_distinguishing_fields() -> None:
    tokenizer = SFTMockTokenizer()
    task = _mixture(_AlphaTask)
    other = _mixture(_BetaTask)
    features = build_synthetic_features(task, tokenizer)

    with (
        patch_load_sft_features(
            features,
            enforce_fingerprints=True,
            split=SFTSplit.TRAIN,
            mixture="capabilities",
            parquet_fingerprint=_fingerprint(task),
            feature_fingerprint=_fingerprint(other),
        ),
        pytest.raises(
            DataLoadError,
            match=r"parquet-vs-feature mixture fingerprint mismatch: differing_fields=",
        ),
    ):
        next(_make_gen(task, tokenizer, alignment_probe_samples=0))


def test_recorded_length_guard_has_an_arm_specific_failure() -> None:
    tokenizer = SFTMockTokenizer()
    live_task = _mixture(_AlphaTask, n=4, prefix="length-live")
    recorded_task = _mixture(_AlphaTask, n=3, prefix="length-recorded")
    features = build_synthetic_features(live_task, tokenizer)
    recorded_fingerprint = _fingerprint(recorded_task)

    with (
        patch_load_sft_features(
            features,
            enforce_fingerprints=True,
            split=SFTSplit.TRAIN,
            mixture="capabilities",
            parquet_fingerprint=recorded_fingerprint,
            feature_fingerprint=recorded_fingerprint,
        ),
        pytest.raises(
            DataLoadError,
            match=(
                r"coverage-length guard recorded len_mixture=3 but "
                r"len\(task\)=4"
            ),
        ),
    ):
        next(_make_gen(live_task, tokenizer, alignment_probe_samples=0))


def test_capped_conversion_uses_indexed_prefix_for_sft_iteration() -> None:
    """A capped conversion trains only its self-described indexed prefix.

    The live task and fingerprint remain full-mixture identities, while the
    conversion metadata authorizes the generator to consume only the rows
    actually written by the capped conversion.  Omitting index 1 also covers
    the converter's legitimate skipped-row behavior.
    """
    tokenizer = SFTMockTokenizer()
    task = _mixture(_AlphaTask, n=6, prefix="capped")
    all_features = build_synthetic_features(task, tokenizer)
    capped_features = {index: all_features[index] for index in (0, 2)}
    state = SFTState()

    with patch_load_sft_features(
        capped_features,
        conversion_total_conversations=3,
        texts=_render_texts(task, tokenizer),
    ):
        batch = next(
            _make_gen(
                task,
                tokenizer,
                B=2,
                alignment_probe_samples=2,
                sft_state=state,
            )
        )

    assert batch[0].shape == (2, 64)
    assert state.dataset_size == 2


def test_uncapped_conversion_tolerates_converter_skipped_rows() -> None:
    """A full-mixture run must survive the converter's legitimate skips.

    ``convert_sft_to_parquet`` drops ``no_assistant`` / ``empty_response``
    conversations, so an UNCAPPED conversion's parquet is a strict subset of
    the live task. Demanding ``range(len(task))`` made every production
    full-mixture run fatal with a "re-extract features" instruction that no
    re-extraction could satisfy (review finding 6).
    """
    tokenizer = SFTMockTokenizer()
    task = _mixture(_AlphaTask, n=4, prefix="uncapped-skips")
    all_features = build_synthetic_features(task, tokenizer)
    converted = {index: all_features[index] for index in (0, 2)}
    state = SFTState()

    with patch_load_sft_features(
        converted,
        conversion_total_conversations=4,
        parquet_original_indices=(0, 2),
        texts=_render_texts(task, tokenizer),
    ):
        batch = next(
            _make_gen(
                task,
                tokenizer,
                B=2,
                alignment_probe_samples=2,
                sft_state=state,
            )
        )

    assert batch[0].shape == (2, 64)
    assert state.dataset_size == 2


def test_recorded_artifact_coverage_pins_both_sides_of_identity() -> None:
    """The resume fingerprint must pin the FEATURE side too (finding 7).

    Recording only document counts plus the parquet/task-mixture fingerprint
    lets a resume re-point ``feature_dir`` at a different extraction of the
    same mixture — other layer, SAE width or pooling — and pass every guard.
    The shard counts are deliberately absent: ``load_sft_features`` already
    rejects any inequality, so they are always equal and discriminate nothing
    (finding 13).
    """
    tokenizer = SFTMockTokenizer()
    task = _mixture(_AlphaTask, n=6, prefix="coverage-identity")
    all_features = build_synthetic_features(task, tokenizer)
    converted = {index: all_features[index] for index in (0, 2)}
    parquet_fingerprint = _fingerprint(task)
    feature_fingerprint = _fingerprint(task, split=SFTSplit.VAL)
    state = SFTState()

    with patch_load_sft_features(
        converted,
        conversion_total_conversations=3,
        parquet_fingerprint=parquet_fingerprint,
        feature_fingerprint=feature_fingerprint,
        texts=_render_texts(task, tokenizer),
    ):
        next(
            _make_gen(
                task,
                tokenizer,
                B=2,
                alignment_probe_samples=2,
                sft_state=state,
            )
        )

    assert state.artifact_coverage == {
        "dataset_size": 2,
        "conversion_total_conversations": 3,
        "parquet_mixture_fingerprint": parquet_fingerprint,
        "feature_mixture_fingerprint": feature_fingerprint,
    }


def test_parquet_row_without_features_is_rejected() -> None:
    """Direction 1: the parquet indexes a document extraction never produced."""
    tokenizer = SFTMockTokenizer()
    task = _mixture(_AlphaTask, n=4, prefix="parquet-orphan")
    all_features = build_synthetic_features(task, tokenizer)
    partial_features = {index: all_features[index] for index in (0, 2)}

    with (
        patch_load_sft_features(
            partial_features,
            conversion_total_conversations=4,
            parquet_original_indices=(0, 1, 2, 3),
            texts=_render_texts(task, tokenizer),
        ),
        pytest.raises(ValueError, match=r"2/4 .* examples missing features"),
    ):
        next(_make_gen(task, tokenizer, alignment_probe_samples=2))


def test_feature_row_without_parquet_row_is_rejected() -> None:
    """Direction 2: a feature row the parquet reference set does not contain.

    This is the direction the previous revision could not see at all: the
    reference set was derived from the feature store, so it agreed with itself
    by construction (review finding 14).
    """
    tokenizer = SFTMockTokenizer()
    task = _mixture(_AlphaTask, n=4, prefix="feature-orphan")
    all_features = build_synthetic_features(task, tokenizer)
    features = {index: all_features[index] for index in (0, 1, 2)}

    with (
        patch_load_sft_features(
            features,
            conversion_total_conversations=4,
            parquet_original_indices=(0, 2),
            texts=_render_texts(task, tokenizer),
        ),
        pytest.raises(
            DataLoadError,
            match=r"indexes 1 documents that the parquet .* does not contain: \[1\]",
        ),
    ):
        next(_make_gen(task, tokenizer, alignment_probe_samples=2))


def test_store_without_parquet_reference_set_is_rejected() -> None:
    """A store that cannot name its parquet rows cannot be validated at all."""
    tokenizer = SFTMockTokenizer()
    task = _mixture(_AlphaTask, n=4, prefix="no-reference")
    features = build_synthetic_features(task, tokenizer)

    with patch_load_sft_features(
        features,
        conversion_total_conversations=4,
        texts=_render_texts(task, tokenizer),
    ) as store:
        setattr(type(store), "parquet_original_indices", None)  # noqa: B010
        with pytest.raises(
            DataLoadError,
            match=r"does not expose parquet_original_indices",
        ):
            next(_make_gen(task, tokenizer, alignment_probe_samples=2))


def _write_disk_fingerprint(
    *,
    parquet_dir: Path,
    feature_dir: Path,
    fingerprint: MixtureFingerprint,
    feature_doc_ids: list[int] | None = None,
) -> None:
    (parquet_dir / "conversion_meta.json").write_text(
        json.dumps(_conversion_meta(fingerprint)),
        encoding="utf-8",
    )
    _write_shard(
        feature_dir,
        "shard_00000",
        doc_ids=feature_doc_ids or [0],
        metadata={SFT_MIXTURE_FINGERPRINT_METADATA_KEY: mixture_fingerprint_json(fingerprint)},
    )
    save_file(
        {"placeholder": t.zeros(1)},
        feature_dir / "feature_statistics.safetensors",
        metadata={SFT_MIXTURE_FINGERPRINT_METADATA_KEY: mixture_fingerprint_json(fingerprint)},
    )


def _make_disk_layout(tmp_path: Path) -> tuple[Path, Path]:
    feature_dir = tmp_path / "features"
    parquet_dir = tmp_path / "parquet"
    feature_dir.mkdir()
    parquet_dir.mkdir()
    _write_shard(feature_dir, "shard_00000", doc_ids=[0])
    _write_parquet(parquet_dir, "shard_00000", original_indices=[0])
    return feature_dir, parquet_dir


def test_loader_rejects_one_sided_parquet_fingerprint(tmp_path: Path) -> None:
    feature_dir, parquet_dir = _make_disk_layout(tmp_path)
    fingerprint = _fingerprint(_mixture(_AlphaTask, n=1))
    (parquet_dir / "conversion_meta.json").write_text(
        json.dumps(_conversion_meta(fingerprint)),
        encoding="utf-8",
    )
    save_file(
        {"placeholder": t.zeros(1)},
        feature_dir / "feature_statistics.safetensors",
        metadata={SFT_MIXTURE_FINGERPRINT_METADATA_KEY: mixture_fingerprint_json(fingerprint)},
    )

    with pytest.raises(
        DataLoadError,
        match=r"feature shard-vs-statistics mixture fingerprint mismatch",
    ):
        load_sft_features(str(feature_dir), str(parquet_dir))


def test_loader_rejects_parquet_statistics_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    feature_dir, parquet_dir = _make_disk_layout(tmp_path)
    fingerprint_a = _fingerprint(_mixture(_AlphaTask, n=1))
    fingerprint_b = _fingerprint(_mixture(_BetaTask, n=1))
    (parquet_dir / "conversion_meta.json").write_text(
        json.dumps(_conversion_meta(fingerprint_a)),
        encoding="utf-8",
    )
    _write_shard(
        feature_dir,
        "shard_00000",
        doc_ids=[0],
        metadata={SFT_MIXTURE_FINGERPRINT_METADATA_KEY: mixture_fingerprint_json(fingerprint_a)},
    )
    save_file(
        {"placeholder": t.zeros(1)},
        feature_dir / "feature_statistics.safetensors",
        metadata={SFT_MIXTURE_FINGERPRINT_METADATA_KEY: mixture_fingerprint_json(fingerprint_b)},
    )

    with pytest.raises(
        DataLoadError,
        match=r"feature shard-vs-statistics mixture fingerprint mismatch: differing_fields=",
    ):
        load_sft_features(str(feature_dir), str(parquet_dir))


def test_real_synthetic_parquet_original_index_shift_fires_probe_first(
    tmp_path: Path,
) -> None:
    """Acceptance falsification: cyclic key coverage passes; exact text does not."""
    tokenizer = SFTMockTokenizer()
    task = _mixture(_VariableLengthTask, n=4, prefix="shift")
    fingerprint = _fingerprint(task)
    feature_dir = tmp_path / "features"
    parquet_dir = tmp_path / "parquet"
    feature_dir.mkdir()
    parquet_dir.mkdir()
    _write_shard(
        feature_dir,
        "shard_00000",
        doc_ids=list(range(len(task))),
    )
    aligned_texts = _render_texts(task, tokenizer)
    shifted_original_indices = [(index + 1) % len(task) for index in range(len(task))]
    pq.write_table(
        pa.table(
            {
                "original_index": shifted_original_indices,
                "text": [aligned_texts[index] for index in range(len(task))],
            }
        ),
        parquet_dir / "shard_00000.parquet",
    )
    _write_disk_fingerprint(
        parquet_dir=parquet_dir,
        feature_dir=feature_dir,
        fingerprint=fingerprint,
        feature_doc_ids=list(range(len(task))),
    )

    with pytest.raises(
        DataLoadError,
        match=r"content alignment probe mismatch at original_index=0",
    ):
        next(
            _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=8,
                feature_dir=str(feature_dir),
                parquet_dir=str(parquet_dir),
            )
        )


def test_cache_identity_includes_regenerated_same_path_fingerprint(tmp_path: Path) -> None:
    feature_dir, parquet_dir = _make_disk_layout(tmp_path)
    fingerprint_a = _fingerprint(_mixture(_AlphaTask, n=1, prefix="a"))
    fingerprint_b = _fingerprint(_mixture(_BetaTask, n=1, prefix="b"))

    _write_disk_fingerprint(
        parquet_dir=parquet_dir,
        feature_dir=feature_dir,
        fingerprint=fingerprint_a,
    )
    store_a = load_sft_features(str(feature_dir), str(parquet_dir))
    _write_disk_fingerprint(
        parquet_dir=parquet_dir,
        feature_dir=feature_dir,
        fingerprint=fingerprint_b,
    )
    store_b = load_sft_features(str(feature_dir), str(parquet_dir))

    assert store_b is not store_a
    assert store_a.parquet_mixture_fingerprint == fingerprint_a
    assert store_b.parquet_mixture_fingerprint == fingerprint_b


def test_extraction_shards_and_statistics_carry_conversion_fingerprint(
    tmp_path: Path,
) -> None:
    fingerprint = _fingerprint(_mixture(_AlphaTask, n=2, prefix="carry"))
    fingerprint_json = mixture_fingerprint_json(fingerprint)
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    (parquet_dir / "conversion_meta.json").write_text(
        json.dumps(_conversion_meta(fingerprint)),
        encoding="utf-8",
    )
    parquet_path = make_parquet(
        parquet_dir / "shard_00000.parquet",
        ["First synthetic SFT row.", "Second synthetic SFT row."],
    )
    source_fingerprint_json = _load_sft_conversion_fingerprint_json(parquet_dir)
    assert source_fingerprint_json == fingerprint_json
    feature_dir = tmp_path / "features"
    run_extraction_loop(
        tiny_params(
            feature_dir,
            sft_mixture_fingerprint_json=source_fingerprint_json,
        ),
        [parquet_path],
        rank=0,
        world_size=1,
        local_rank=0,
        shard_start=NonNegativeInt.parse(0),
        shard_end=NonNegativeInt.parse(1),
        staging_cfg=_staging(tmp_path, 0, 0),
        profile_enabled=False,
        profile_output=str(tmp_path / "profile"),
        flops_per_token=1.0,
        gpu_peak_flops=1.0,
    )
    feature_shard = glob_features(feature_dir)[0]
    assert (
        read_shard_metadata(feature_shard)[SFT_MIXTURE_FINGERPRINT_METADATA_KEY] == fingerprint_json
    )

    stats_path = feature_dir / "feature_statistics.safetensors"
    compute_statistics(
        feature_dir=str(feature_dir),
        output_path=str(stats_path),
        skip_per_feature_detail=True,
        streaming=False,
        allow_degenerate_statistics=True,
    )
    assert (
        load_statistics_metadata(stats_path)[SFT_MIXTURE_FINGERPRINT_METADATA_KEY]
        == fingerprint_json
    )


def test_turn_delimited_convert_extract_statistics_generator_and_step(
    tmp_path: Path,
) -> None:
    tokenizer = _CompactCharacterTokenizer()
    task = TaskMixture([_UnequalTurnTask()])
    parquet_dir = tmp_path / "parquet"
    with patch(
        "scripts.preprocess.convert_sft_to_parquet.get_tokenizer",
        return_value=tokenizer,
    ):
        convert_mixture_to_parquet(
            task,
            parquet_dir,
            conversations_per_shard=PositiveInt.parse(8),
            mixture_name="capabilities",
            split=SFTSplit.TRAIN,
        )

    parquet_path = parquet_dir / "shard_00000.parquet"
    feature_dir = tmp_path / "features"
    segment_size = PositiveInt.parse(5)
    run_extraction_loop(
        tiny_params(
            feature_dir,
            model=TinyHookedTransformer(),
            sae=TinySAE(n_features=int(N_SAE_FEATURES)),
            dataset_phase="capabilities_sft",
            segment_mode=SegmentMode.turn_delimited_fixed_tokens,
            segment_size=segment_size,
            context_window=PositiveInt.parse(256),
            sft_mixture_fingerprint_json=_load_sft_conversion_fingerprint_json(
                parquet_dir
            ),
        ),
        [str(parquet_path)],
        rank=0,
        world_size=1,
        local_rank=0,
        shard_start=NonNegativeInt.parse(0),
        shard_end=NonNegativeInt.parse(1),
        staging_cfg=_staging(tmp_path / "turn_staging", 0, 0),
        profile_enabled=False,
        profile_output=str(tmp_path / "turn_profile"),
        flops_per_token=1.0,
        gpu_peak_flops=1.0,
    )
    feature_shard = glob_features(feature_dir)[0]
    feature_metadata = read_shard_metadata(feature_shard)
    shard = load_features(feature_shard)
    doc_ids = shard["doc_ids"].to(t.int64)
    assert set(doc_ids.tolist()) == {0, 1}
    marker_by_doc = {0: 3, 1: 7}
    planted_ids = t.empty_like(shard["feature_ids"], dtype=t.int32)
    offsets = shard["offsets"].tolist()
    for segment_index, doc_id in enumerate(doc_ids.tolist()):
        planted_ids[offsets[segment_index] : offsets[segment_index + 1]] = (
            marker_by_doc[int(doc_id)]
        )
    shard["feature_ids"] = planted_ids
    shard[SFT_MIXTURE_FINGERPRINT_METADATA_KEY] = feature_metadata[
        SFT_MIXTURE_FINGERPRINT_METADATA_KEY
    ]
    save_features(shard, feature_shard)
    stats_path = feature_dir / "feature_statistics.safetensors"
    compute_statistics(
        feature_dir=str(feature_dir),
        output_path=str(stats_path),
        exclude_last_shard=False,
        skip_per_feature_detail=True,
        streaming=False,
        allow_degenerate_statistics=True,
    )

    store = load_sft_features(str(feature_dir), str(parquet_dir))
    assert store.segment_mode is SegmentMode.turn_delimited_fixed_tokens

    rendered = [tokenizer.render_conversation(task[index])[0] for index in range(len(task))]
    assert len(rendered[0]) != len(rendered[1])
    T = max(len(ids) - 1 for ids in rendered)
    batch = next(
        _make_gen(
            task,
            tokenizer,
            B=2,
            T=T,
            alignment_probe_samples=2,
            feature_dir=str(feature_dir),
            parquet_dir=str(parquet_dir),
            feature_source_bos_char_len=1,
            single_pass=True,
            return_seg_doc_id=True,
        )
    )

    inputs, _targets, enc_fids, _enc_mags, enc_mask, seg_map, _fp_targets, seg_doc_id = batch
    row_to_original_index: dict[int, int] = {}
    for row in range(2):
        matches = [
            original_index
            for original_index, ids in enumerate(rendered)
            if t.equal(inputs[row, : len(ids) - 1], t.tensor(ids[:-1]))
        ]
        assert len(matches) == 1
        row_to_original_index[row] = matches[0]

    transition_sets: list[set[int]] = []
    for row, original_index in row_to_original_index.items():
        ids = rendered[original_index]
        spans = legacy_turn_spans(ids, ChatSpecialTokens.from_tokenizer(tokenizer))
        offsets = source_aligned_char_offsets(
            tokenizer,
            ids,
            source_bos_char_len=1,
        )
        turn_starts = [0, *[offsets[span.start + 1] for span in spans[1:]]]
        turn_ends = [*turn_starts[1:], offsets[-1]]
        expected_char_starts = [
            start
            for turn_start, turn_end in zip(turn_starts, turn_ends, strict=True)
            for start in range(turn_start, turn_end, int(segment_size))
        ]
        expected_char_ends = [*expected_char_starts[1:], offsets[-1]]
        assert [segment.char_start for segment in store[original_index]] == expected_char_starts
        assert [segment.char_end for segment in store[original_index]] == expected_char_ends

        expected_transitions: set[int] = set()
        for char_start in expected_char_starts[1:]:
            transition = min(
                range(1, len(offsets) - 1),
                key=lambda token_index: abs(offsets[token_index] - char_start),
            )
            if transition < len(ids) - 1:
                expected_transitions.add(transition)
        actual_labels = seg_map[row, : len(ids) - 1]
        actual_transitions = {
            index
            for index in range(1, len(ids) - 1)
            if actual_labels[index] != actual_labels[index - 1]
        }
        assert actual_transitions == expected_transitions
        transition_sets.append(actual_transitions)
        active_segment_ids = t.unique_consecutive(actual_labels[actual_labels > 0])
        assert t.all(active_segment_ids[1:] > active_segment_ids[:-1])

        row_feature_ids = enc_fids[row][enc_mask[row]]
        assert set(row_feature_ids.tolist()) == {marker_by_doc[original_index]}
        segment_mask = enc_mask[row].any(dim=-1)
        assert set(seg_doc_id[row][segment_mask].tolist()) == {0}

    assert transition_sets[0] != transition_sets[1]
    assert not t.equal(seg_map[0], seg_map[1])

    model = _build_tiny_encdec_matching_sft()
    optimizer = setup_optimizer_simple(model)
    with patch.object(_fa_mod, "_override_impl", FlashAttnImpl.SDPA):
        loss = _run_one_training_step(model, batch[:7], optimizer)
    assert math.isfinite(loss)


def test_val_shard_fingerprints_load_without_statistics_and_probe_is_active(
    tmp_path: Path,
) -> None:
    tokenizer = SFTMockTokenizer()
    val_task = _mixture(_VariableLengthTask, n=2, prefix="val-carrier")
    fingerprint = _fingerprint(val_task, split=SFTSplit.VAL)
    fingerprint_json = mixture_fingerprint_json(fingerprint)
    parquet_dir = tmp_path / "val_parquet"
    parquet_dir.mkdir()
    (parquet_dir / "conversion_meta.json").write_text(
        json.dumps(_conversion_meta(fingerprint, split=SFTSplit.VAL)),
        encoding="utf-8",
    )
    aligned_texts = _render_texts(val_task, tokenizer)
    parquet_path = parquet_dir / "shard_00000.parquet"
    pq.write_table(
        pa.table(
            {
                "original_index": list(range(len(val_task))),
                "text": [aligned_texts[index] for index in range(len(val_task))],
            }
        ),
        parquet_path,
    )

    feature_dir = tmp_path / "val_features"
    run_extraction_loop(
        tiny_params(
            feature_dir,
            model=TinyHookedTransformer(),
            sae=TinySAE(n_features=int(N_SAE_FEATURES)),
            dataset_phase="capabilities_sft_val",
            sft_mixture_fingerprint_json=fingerprint_json,
        ),
        [str(parquet_path)],
        rank=0,
        world_size=1,
        local_rank=0,
        shard_start=NonNegativeInt.parse(0),
        shard_end=NonNegativeInt.parse(1),
        staging_cfg=_staging(tmp_path / "val_staging", 0, 0),
        profile_enabled=False,
        profile_output=str(tmp_path / "val_profile"),
        flops_per_token=1.0,
        gpu_peak_flops=1.0,
    )
    assert not (feature_dir / "feature_statistics.safetensors").exists()
    assert all(
        read_shard_metadata(path)[SFT_MIXTURE_FINGERPRINT_METADATA_KEY] == fingerprint_json
        for path in glob_features(feature_dir)
    )
    store = load_sft_features(str(feature_dir), str(parquet_dir))
    assert store.parquet_mixture_fingerprint == fingerprint
    assert store.feature_mixture_fingerprint == fingerprint

    boundaries = (
        t.linspace(0.5, float(NUM_BUCKETS) - 0.5, int(NUM_BUCKETS) - 1)
        .unsqueeze(0)
        .expand(int(N_SAE_FEATURES), -1)
        .contiguous()
    )

    def build_val_loader():
        return _build_sft_val_loader(
            val_task=val_task,
            tokenizer=tokenizer,
            B=1,
            T=64,
            device="cpu",
            ddp_rank=0,
            ddp_world_size=1,
            feature_dir=str(feature_dir),
            expected_feature_dir=str(feature_dir),
            parquet_dir=str(parquet_dir),
            threshold_cfg=ThresholdConfig(ThresholdType.segment_topk, 1000),
            boundaries=boundaries,
            precomputed_thresholds=None,
            sampling_bundle=None,
            loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
            min_loss_tokens=1,
            feature_source_bos_char_len=1,
            alignment_probe_samples=2,
            mixture_name="capabilities",
        )

    assert next(build_val_loader()) is not None

    pq.write_table(
        pa.table(
            {
                "original_index": list(range(len(val_task))),
                "text": [
                    aligned_texts[(index + 1) % len(val_task)] for index in range(len(val_task))
                ],
            }
        ),
        parquet_path,
    )
    with pytest.raises(
        DataLoadError,
        match=r"content alignment probe mismatch at original_index=0",
    ):
        next(build_val_loader())


def test_extraction_resume_rejects_fingerprint_on_shard_only(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    fingerprint_json = mixture_fingerprint_json(_fingerprint(_mixture(_AlphaTask, n=1)))
    _write_shard(
        feature_dir,
        "shard_00000",
        doc_ids=[0],
        metadata={SFT_MIXTURE_FINGERPRINT_METADATA_KEY: fingerprint_json},
    )

    with pytest.raises(
        DataLoadError,
        match=r"source parquet has no fingerprint but existing shard does",
    ):
        _assert_existing_output_dir_compatible(
            feature_dir,
            SegmentMode.fixed_tokens,
            None,
        )


def test_extraction_resume_rejects_fingerprint_on_source_only(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    _write_shard(feature_dir, "shard_00000", doc_ids=[0])
    fingerprint_json = mixture_fingerprint_json(_fingerprint(_mixture(_AlphaTask, n=1)))

    with pytest.raises(
        DataLoadError,
        match=r"source parquet has a fingerprint but existing shard does not",
    ):
        _assert_existing_output_dir_compatible(
            feature_dir,
            SegmentMode.fixed_tokens,
            fingerprint_json,
        )


def test_extraction_resume_rejects_different_fingerprints(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    shard_fingerprint_json = mixture_fingerprint_json(_fingerprint(_mixture(_AlphaTask, n=1)))
    source_fingerprint_json = mixture_fingerprint_json(_fingerprint(_mixture(_BetaTask, n=1)))
    _write_shard(
        feature_dir,
        "shard_00000",
        doc_ids=[0],
        metadata={SFT_MIXTURE_FINGERPRINT_METADATA_KEY: shard_fingerprint_json},
    )

    with pytest.raises(
        DataLoadError,
        match=r"source parquet and existing shard fingerprints differ",
    ):
        _assert_existing_output_dir_compatible(
            feature_dir,
            SegmentMode.fixed_tokens,
            source_fingerprint_json,
        )


def test_compute_statistics_rejects_half_migrated_sft_shards(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    fingerprint_json = mixture_fingerprint_json(_fingerprint(_mixture(_AlphaTask, n=2)))
    _write_shard(
        feature_dir,
        "shard_00000",
        doc_ids=[0],
        metadata={SFT_MIXTURE_FINGERPRINT_METADATA_KEY: fingerprint_json},
    )
    _write_shard(feature_dir, "shard_00001", doc_ids=[0])

    with pytest.raises(
        ValueError,
        match=r"compute_statistics found a half-migrated SFT feature family",
    ):
        compute_statistics(
            feature_dir=str(feature_dir),
            output_path=str(feature_dir / "feature_statistics.safetensors"),
            exclude_last_shard=False,
        )


def test_compute_statistics_rejects_inconsistent_sft_shard_fingerprints(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    fingerprint_a_json = mixture_fingerprint_json(_fingerprint(_mixture(_AlphaTask, n=2)))
    fingerprint_b_json = mixture_fingerprint_json(_fingerprint(_mixture(_BetaTask, n=2)))
    _write_shard(
        feature_dir,
        "shard_00000",
        doc_ids=[0],
        metadata={SFT_MIXTURE_FINGERPRINT_METADATA_KEY: fingerprint_a_json},
    )
    _write_shard(
        feature_dir,
        "shard_00001",
        doc_ids=[0],
        metadata={SFT_MIXTURE_FINGERPRINT_METADATA_KEY: fingerprint_b_json},
    )

    with pytest.raises(
        ValueError,
        match=r"inconsistent SFT mixture fingerprints across feature shards",
    ):
        compute_statistics(
            feature_dir=str(feature_dir),
            output_path=str(feature_dir / "feature_statistics.safetensors"),
            exclude_last_shard=False,
        )


class TestTrailingUserRowsFollowTheConverterContract:
    """Issue #1465: the generator must render the converter's slice."""

    def test_untrimmed_render_really_would_trip_the_coverage_guard(self) -> None:
        """Falsify the measurement tool: prove the fixture reproduces the bug.

        Without this, the passing generator test below could pass vacuously on
        a fixture whose trailing user turn is too short to matter.
        """
        tokenizer = SFTMockTokenizer()
        task = TrailingUserTask(n=1, tail_chars=40)
        features = build_synthetic_features(task, tokenizer)
        seg_features = [
            SegFeatures(
                feature_ids=seg.feature_ids.to(t.int64),
                buckets=t.zeros(len(seg.feature_ids), dtype=t.int64),
                percentiles=t.zeros(len(seg.feature_ids), dtype=t.float32),
                char_start=seg.char_start,
                char_end=seg.char_end,
            )
            for seg in features[0]
        ]
        untrimmed_ids, _ = render_conversation_with_mask(tokenizer, task[0])
        gap = compute_full_coverage_gap(untrimmed_ids, seg_features, tokenizer)
        assert gap > 20, f"fixture does not reproduce the #1465 gap (gap={gap})"

        trimmed_ids, _ = _trimmed_render(tokenizer, task[0])
        assert compute_full_coverage_gap(trimmed_ids, seg_features, tokenizer) == 0

    def test_generator_packs_trailing_user_rows_without_a_coverage_failure(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = TrailingUserTask(n=4, tail_chars=40)
        features = build_synthetic_features(task, tokenizer)
        state = SFTState()
        with patch_load_sft_features(features, texts=_render_texts(task, tokenizer)):
            batch = next(
                _make_gen(
                    task,
                    tokenizer,
                    B=2,
                    T=64,
                    alignment_probe_samples=4,
                    sft_state=state,
                )
            )
        assert batch[0].shape == (2, 64)

    def test_packed_row_holds_the_trimmed_render_not_the_untrimmed_one(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = TrailingUserTask(n=1, tail_chars=40)
        features = build_synthetic_features(task, tokenizer)
        expected_ids, _ = _trimmed_render(tokenizer, task[0])
        untrimmed_ids, _ = render_conversation_with_mask(tokenizer, task[0])
        assert len(expected_ids) < len(untrimmed_ids)  # non-vacuity

        # The first token the trim drops is the trailing user turn's role
        # marker. ``SFTMockTokenizer`` numbers ids sequentially within a
        # conversation, so that id is strictly greater than every id in the
        # trimmed render and can only appear in a packed row if the generator
        # rendered the untrimmed conversation.
        # ``SFTMockTokenizer`` restarts ids at 2 for every conversation and
        # wraps modulo VOCAB_SIZE=100, so keep the fixture under ~98 tokens
        # (3 + 6 + 40 content chars + 3 markers = 52 here). This assert is the
        # tripwire if that bound is ever exceeded.
        first_dropped = untrimmed_ids[len(expected_ids)]
        assert first_dropped not in expected_ids

        with patch_load_sft_features(features, texts=_render_texts(task, tokenizer)):
            batch = next(_make_gen(task, tokenizer, B=1, T=128, alignment_probe_samples=0))
        row = batch[0][0].tolist()
        assert row[: len(expected_ids)] == expected_ids
        assert first_dropped not in row

    def test_validation_single_pass_covers_trailing_user_rows_and_terminates(
        self,
    ) -> None:
        """Acceptance criterion (b) is about the VALIDATION generator.

        ``single_pass=True`` is the mode ``_build_sft_val_loader`` uses and the
        one the production run died in.
        ``tests/scripts/test_encdec_sft_training_loop.py::
        test_single_pass_validation_never_serves_an_index_twice`` already
        drives ``single_pass=True`` through ``refill_buffer`` via
        ``evaluate_encdec_bpb``, so reaching that code path is not itself
        new here. What IS new: combining ``single_pass=True`` with
        trailing-user rows, so this is the first test to exercise the
        coverage guard on the validation path.
        """
        tokenizer = SFTMockTokenizer()
        task = TrailingUserTask(n=6, tail_chars=40)
        features = build_synthetic_features(task, tokenizer)
        state = SFTState()
        with patch_load_sft_features(
            features,
            split=SFTSplit.VAL,
            texts=_render_texts(task, tokenizer),
        ):
            gen = _make_gen(
                task,
                tokenizer,
                B=2,
                T=64,
                alignment_probe_samples=6,
                sft_state=state,
                expected_split=SFTSplit.VAL,
                single_pass=True,
            )
            # The raw generator never raises StopIteration — only
            # ``sft_state.exhausted`` marks the end of a bounded pass (see
            # ``_SFTValidationLoader``, which delegates ``__next__`` verbatim
            # and leaves termination to ``evaluate_encdec_bpb``). Bound the
            # loop so a regression fails instead of hanging the suite.
            batches = []
            for _ in range(20):
                batches.append(next(gen))
                if state.exhausted:
                    break
            else:
                raise AssertionError("single-pass validation never exhausted")
        assert batches, "single-pass validation produced no batch"
        assert state.consumed == len(task)

    def test_converter_skipped_row_raises_instead_of_being_silently_skipped(
        self,
    ) -> None:
        tokenizer = SFTMockTokenizer()
        task = _UserOnlyTask(n=2)
        features = build_synthetic_features(SingleTurnTask(n=2), tokenizer)
        with patch_load_sft_features(features):
            with pytest.raises(DataLoadError, match=r"no_assistant"):
                next(_make_gen(task, tokenizer, B=1, T=64, alignment_probe_samples=0))


class _MixedShapesTask(Task):
    """One row of each shape the converter distinguishes.

    Index 0 trailing-assistant (no trim), 1 trailing-user (trim), 2 user-only
    (``no_assistant`` skip), 3 empty final assistant (``empty_response`` skip),
    4 multi-turn trailing-user (trim).
    """

    _ROWS = (
        {
            "messages": [
                {"role": "user", "content": "aa"},
                {"role": "assistant", "content": "bbbb"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "cc"},
                {"role": "assistant", "content": "dddd"},
                {"role": "user", "content": "thanks a lot indeed"},
            ]
        },
        {"messages": [{"role": "user", "content": "ee"}]},
        {
            "messages": [
                {"role": "user", "content": "ff"},
                {"role": "assistant", "content": "   "},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "gg"},
                {"role": "assistant", "content": "hhhh"},
                {"role": "user", "content": "ii"},
                {"role": "assistant", "content": "jjjj"},
                {"role": "user", "content": "one more question here"},
            ]
        },
    )

    def num_examples(self) -> int:
        return len(self._ROWS)

    def get_example(self, index: int) -> dict:
        return copy.deepcopy(self._ROWS[index])


class _TurnAwareSFTMockTokenizer(SFTMockTokenizer):
    """Round-trip tokenizer with fixed special IDs for legacy span recovery."""

    _specials = {
        "<user_start>": 2,
        "<assistant_start>": 3,
        "<output_start>": 4,
        "<output_end>": 5,
    }

    def encode_special(self, name: str) -> int:
        return self._specials[name]

    def render_conversation(self, conversation: dict) -> tuple[list[int], list[int]]:
        ids = [self.get_bos_token_id()]
        mask = [0]
        next_token = 6
        for message in conversation["messages"]:
            role = message["role"]
            role_marker = self._specials[f"<{role}_start>"]
            content_tokens = list(range(next_token, next_token + len(message["content"])))
            next_token += len(content_tokens)
            ids.extend([role_marker, *content_tokens])
            mask.extend([int(role == "assistant")] * (len(content_tokens) + 1))
        return ids, mask


class TestConverterTrainerRoundTrip:
    """The real converter's output must feed the real generator unchanged."""

    def test_real_parquet_feeds_the_generator_with_zero_coverage_gap(
        self, tmp_path, monkeypatch
    ) -> None:
        tokenizer = _TurnAwareSFTMockTokenizer()
        task = _MixedShapesTask()
        monkeypatch.setattr(
            "scripts.preprocess.convert_sft_to_parquet.get_tokenizer",
            lambda: tokenizer,
        )
        convert_mixture_to_parquet(
            mixture=task,
            output_dir=str(tmp_path),
            max_conversations=None,
            conversations_per_shard=10,
            mixture_name="roundtrip",
            split=SFTSplit.TRAIN,
        )

        shards = sorted(tmp_path.glob("shard_*.parquet"))
        assert shards, "converter wrote no shard"
        table = pq.read_table(shards[0])
        texts = dict(
            zip(
                table.column("original_index").to_pylist(),
                table.column("text").to_pylist(),
                strict=True,
            )
        )

        # The converter drops the user-only and empty-response rows, so the
        # generator can never be asked to render them.
        assert sorted(texts) == [0, 1, 4]

        features = build_synthetic_features(task, tokenizer, skip_indices=frozenset({2, 3}))
        with patch_load_sft_features(
            features,
            texts=texts,
            conversion_total_conversations=len(task),
            parquet_original_indices=tuple(sorted(texts)),
        ):
            batch = next(
                _make_gen(
                    task,
                    tokenizer,
                    B=1,
                    T=256,
                    alignment_probe_samples=3,
                    sft_state=SFTState(),
                )
            )
        assert batch[0].shape == (1, 256)

        # Row 1's rendered slice must be strictly shorter than its raw render,
        # or this whole test is vacuous.
        trimmed, _ = _trimmed_render(tokenizer, task[1])
        untrimmed, _ = render_conversation_with_mask(tokenizer, task[1])
        assert len(trimmed) < len(untrimmed)


class TestCoverageGuardStillFires:
    """The #1465 fix must not soften the under-coverage guard."""

    def test_genuinely_truncated_features_still_raise(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=2, user_chars=8, asst_chars=40)
        features = build_synthetic_features(task, tokenizer)
        # Chop 30 characters of coverage off every row: this is what a stale
        # or mis-tokenized extraction looks like, and it must still be fatal.
        starved = {
            index: [
                replace(seg, char_end=max(seg.char_start + 1, seg.char_end - 30)) for seg in segs
            ]
            for index, segs in features.items()
        }
        with patch_load_sft_features(starved):
            with pytest.raises(ValueError, match=r"feature coverage ends \d+ chars before"):
                next(_make_gen(task, tokenizer, B=1, T=128, alignment_probe_samples=0))
