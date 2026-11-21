from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest
import torch
import yaml

import scripts.eval.base_eval as base_eval
from nanochat.core_eval import (
    batch_sequences_mc,
    render_prompts_mc,
    should_skip_oversized_core_example,
)
from steering.eval.context_cache_io import ContextFeatureSegment
from steering.eval.pooling_op import PoolingOp
from steering.eval.target_conditioning import PooledTargetRealConfig


class _MixedLengthTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def __call__(self, prompts, prepend=None):
        rows = []
        for prompt in prompts:
            if "OVERSIZED" in prompt:
                prefix = [99] * 199
            elif "short0" in prompt:
                prefix = [10, 20, 10, 20]
            elif "short1" in prompt:
                prefix = [10, 21, 10, 21]
            else:
                prefix = [10, 22, 10, 22]
            choice_token = 30 if prompt.endswith("A") else 31
            row = prefix + [choice_token]
            if prepend is not None:
                row = [prepend] + row
            rows.append(row)
        return rows

    def decode(self, ids) -> str:
        return "x" * len(list(ids))


class _ZeroChoiceModel:
    max_seq_len = 50

    def __init__(self) -> None:
        self.dec_config = SimpleNamespace(sequence_len=50)

    def __call__(self, input_ids):
        return torch.zeros(input_ids.size(0), input_ids.size(1), 256)


class _RecordingTargetAwareZeroChoiceModel(_ZeroChoiceModel):
    is_target_aware_encdec = True

    def __init__(self, context_provider, captured: list[tuple[list[int], tuple]]) -> None:
        super().__init__()
        self.context_provider = context_provider
        self.captured = captured
        self._active_context_token_ids = None

    def conditioning_context(self, *, target_mask=None, context_token_ids=None):
        model = self

        class _Ctx:
            def __enter__(self_inner):
                assert target_mask is not None
                model._active_context_token_ids = context_token_ids
                return model

            def __exit__(self_inner, exc_type, exc, tb):
                model._active_context_token_ids = None
                return False

        return _Ctx()

    def __call__(self, input_ids):
        assert self._active_context_token_ids is not None
        for raw_row_tokens in self._active_context_token_ids:
            row_tokens = [int(tok) for tok in raw_row_tokens]
            segments = self.context_provider.get(row_tokens)
            assert segments is not None, "wrapper context provider must resolve prefix rows"
            self.captured.append((row_tokens, tuple(segments)))
        return super().__call__(input_ids)


class _RecordingProvider:
    def __init__(
        self,
        token_to_idx: dict[tuple[int, ...], int],
        token_to_char_end: dict[tuple[int, ...], int],
        full_rows: set[tuple[int, ...]],
    ):
        self.token_to_idx = token_to_idx
        self.token_to_char_end = token_to_char_end
        self.full_rows = full_rows
        self.scored_indices: list[int] = []
        self.lookup_lengths: list[int] = []
        self.full_lookup_lengths: list[int] = []
        self.context_lookup_lengths: list[int] = []

    def get(self, token_ids):
        tokens = tuple(int(tok) for tok in token_ids)
        self.lookup_lengths.append(len(tokens))
        idx = self.token_to_idx[tokens]
        if tokens in self.full_rows:
            self.full_lookup_lengths.append(len(tokens))
        else:
            self.context_lookup_lengths.append(len(tokens))
        if tokens in self.full_rows and idx not in self.scored_indices:
            self.scored_indices.append(idx)
        return (
            ContextFeatureSegment(
                feature_ids=(0,),
                buckets=(1,),
                char_start=0,
                char_end=self.token_to_char_end[tokens],
            ),
        )


def _mixed_data() -> list[dict]:
    return [
        {"query": "OVERSIZED", "choices": ["A", "B"], "gold": 1},
        {"query": "short0", "choices": ["A", "B"], "gold": 0},
        {"query": "short1", "choices": ["A", "B"], "gold": 1},
    ]


def _write_core_bundle(tmp_path, tasks: list[dict[str, object]]) -> None:
    bundle_dir = tmp_path / "eval_bundle"
    bundle_dir.mkdir()
    (bundle_dir / "eval_data").mkdir()
    config_tasks = []
    meta_lines = ["Eval Task,Random baseline"]
    for task in tasks:
        label = str(task["label"])
        dataset_uri = f"{label}.jsonl"
        config_tasks.append(
            {
                "label": label,
                "icl_task_type": "multiple_choice",
                "num_fewshot": [0],
                "continuation_delimiter": " ",
                "dataset_uri": dataset_uri,
            }
        )
        rows = task["rows"]
        assert isinstance(rows, list)
        (bundle_dir / "eval_data" / dataset_uri).write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
        meta_lines.append(f"{label},0.0")
    (bundle_dir / "core.yaml").write_text(yaml.safe_dump({"icl_tasks": config_tasks}))
    (bundle_dir / "eval_meta_data.csv").write_text("\n".join(meta_lines) + "\n")


def _shuffled(rows: list[dict]) -> list[dict]:
    task_data = list(rows)
    random.Random(1337).shuffle(task_data)
    return task_data


def _expected_scored_mc_pairs(
    rows: list[dict],
    tokenizer: _MixedLengthTokenizer,
    *,
    max_tokens: int,
    label: str = "mixed_mc",
) -> list[tuple[int, bool]]:
    expected: list[tuple[int, bool]] = []
    for idx, item in enumerate(rows):
        prompts = render_prompts_mc(item, " ", [])
        tokens, _starts, _ends = batch_sequences_mc(tokenizer, prompts)
        if should_skip_oversized_core_example(
            tokens,
            max_tokens=max_tokens,
            label=label,
            idx=idx,
        ):
            continue
        expected.append((idx, item["gold"] == 0))
    return expected


def _build_provider(
    rows: list[dict],
    tokenizer: _MixedLengthTokenizer,
) -> _RecordingProvider:
    task_data = _shuffled(rows)
    token_to_idx: dict[tuple[int, ...], int] = {}
    token_to_char_end: dict[tuple[int, ...], int] = {}
    full_rows: set[tuple[int, ...]] = set()
    for idx, item in enumerate(task_data):
        prompts = render_prompts_mc(item, " ", [])
        tokens, start_idxs, _end_idxs = batch_sequences_mc(tokenizer, prompts)
        for row, si in zip(tokens, start_idxs, strict=True):
            full_key = tuple(row)
            context_key = tuple(row[: max(si - 1, 0) + 1])
            token_to_idx[full_key] = idx
            token_to_idx[context_key] = idx
            token_to_char_end[full_key] = len(tokenizer.decode(full_key))
            token_to_char_end[context_key] = len(tokenizer.decode(context_key))
            full_rows.add(full_key)
    return _RecordingProvider(token_to_idx, token_to_char_end, full_rows)


def test_base_eval_evaluate_core_excludes_all_oversized_task_labels(monkeypatch, tmp_path):
    _write_core_bundle(
        tmp_path,
        [
            {"label": "mixed_mc", "rows": _mixed_data()},
            {
                "label": "all_oversized_mc",
                "rows": [
                    {"query": "OVERSIZED a", "choices": ["A", "B"], "gold": 0},
                    {"query": "OVERSIZED b", "choices": ["A", "B"], "gold": 1},
                ],
            },
        ],
    )
    monkeypatch.setattr(base_eval, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(base_eval, "download_file_with_lock", lambda *a, **k: None)

    out = base_eval.evaluate_core(
        _ZeroChoiceModel(),
        _MixedLengthTokenizer(),
        "cpu",
        max_per_task=-1,
    )

    assert set(out["results"]) == {"mixed_mc"}
    assert set(out["centered_results"]) == {"mixed_mc"}
    assert out["results"]["mixed_mc"] == pytest.approx(0.5)
    assert out["core_metric"] == pytest.approx(out["centered_results"]["mixed_mc"])


def test_core_target_real_skip_set_matches_context_length_filter(
    monkeypatch,
    tmp_path,
):
    rows = _mixed_data()
    _write_core_bundle(tmp_path, [{"label": "mixed_mc", "rows": rows}])
    monkeypatch.setattr(base_eval, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(base_eval, "download_file_with_lock", lambda *a, **k: None)

    tokenizer = _MixedLengthTokenizer()
    task_data = _shuffled(rows)
    expected_pairs = _expected_scored_mc_pairs(
        task_data,
        tokenizer,
        max_tokens=_ZeroChoiceModel.max_seq_len,
    )

    provider = _build_provider(rows, tokenizer)
    monkeypatch.setattr(
        base_eval,
        "build_wrapper_for_features",
        lambda *args, **kwargs: _ZeroChoiceModel(),
    )

    out = base_eval._run_core_target_real_eval(
        raw_encdec=_ZeroChoiceModel(),
        tokenizer=tokenizer,
        device="cpu",
        trc=PooledTargetRealConfig(
            name="pooled",
            cache_dir=tmp_path / "cache",
            pooling_op=PoolingOp.BUCKET_MAX,
        ),
        provider=provider,
        max_per_task=-1,
    )

    target_pairs = [(idx, task_data[idx]["gold"] == 0) for idx in provider.scored_indices]
    assert target_pairs == expected_pairs
    assert max(provider.lookup_lengths) <= _ZeroChoiceModel.max_seq_len
    assert set(out["results"]) == {"mixed_mc"}
    assert out["results"]["mixed_mc"] == pytest.approx(0.5)


def test_core_target_real_wrapper_uses_clipped_context_prefix(
    monkeypatch,
    tmp_path,
):
    rows = _mixed_data()
    _write_core_bundle(tmp_path, [{"label": "mixed_mc", "rows": rows}])
    monkeypatch.setattr(base_eval, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(base_eval, "download_file_with_lock", lambda *a, **k: None)

    tokenizer = _MixedLengthTokenizer()
    provider = _build_provider(rows, tokenizer)
    captured_contexts: list[tuple[list[int], tuple]] = []

    def _fake_wrapper(*args, **kwargs):
        return _RecordingTargetAwareZeroChoiceModel(
            kwargs["context_provider"],
            captured_contexts,
        )

    monkeypatch.setattr(base_eval, "build_wrapper_for_features", _fake_wrapper)

    base_eval._run_core_target_real_eval(
        raw_encdec=_ZeroChoiceModel(),
        tokenizer=tokenizer,
        device="cpu",
        trc=PooledTargetRealConfig(
            name="pooled",
            cache_dir=tmp_path / "cache",
            pooling_op=PoolingOp.BUCKET_MAX,
        ),
        provider=provider,
        max_per_task=-1,
    )

    assert captured_contexts
    assert provider.full_lookup_lengths
    assert provider.context_lookup_lengths
    assert max(provider.context_lookup_lengths) < max(provider.full_lookup_lengths)
    for row_tokens, segments in captured_contexts:
        assert len(segments) == 1
        assert segments[0].char_start == 0
        assert segments[0].char_end == len(tokenizer.decode(row_tokens))


def test_core_target_real_exposes_non_oversized_task_set(
    monkeypatch,
    tmp_path,
):
    rows = _mixed_data()
    _write_core_bundle(
        tmp_path,
        [
            {"label": "mixed_mc", "rows": rows},
            {
                "label": "all_oversized_mc",
                "rows": [
                    {"query": "OVERSIZED a", "choices": ["A", "B"], "gold": 0},
                    {"query": "OVERSIZED b", "choices": ["A", "B"], "gold": 1},
                ],
            },
        ],
    )
    monkeypatch.setattr(base_eval, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(base_eval, "download_file_with_lock", lambda *a, **k: None)
    monkeypatch.setattr(
        base_eval,
        "build_wrapper_for_features",
        lambda *args, **kwargs: _ZeroChoiceModel(),
    )

    tokenizer = _MixedLengthTokenizer()
    provider = _build_provider(rows, tokenizer)

    target_real = base_eval._run_core_target_real_eval(
        raw_encdec=_ZeroChoiceModel(),
        tokenizer=tokenizer,
        device="cpu",
        trc=PooledTargetRealConfig(
            name="pooled",
            cache_dir=tmp_path / "cache",
            pooling_op=PoolingOp.BUCKET_MAX,
        ),
        provider=provider,
        max_per_task=-1,
    )

    assert set(target_real["results"]) == {"mixed_mc"}
    assert set(target_real["centered_results"]) == {"mixed_mc"}
    assert target_real["core_metric"] == pytest.approx(target_real["centered_results"]["mixed_mc"])


def test_core_target_real_raises_when_all_tasks_are_oversized(monkeypatch, tmp_path):
    _write_core_bundle(
        tmp_path,
        [
            {
                "label": "all_oversized_mc",
                "rows": [
                    {"query": "OVERSIZED a", "choices": ["A", "B"], "gold": 0},
                    {"query": "OVERSIZED b", "choices": ["A", "B"], "gold": 1},
                ],
            }
        ],
    )
    monkeypatch.setattr(base_eval, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(base_eval, "download_file_with_lock", lambda *a, **k: None)
    provider = _RecordingProvider({}, {}, set())

    with pytest.raises(ValueError, match="no evaluable tasks"):
        base_eval._run_core_target_real_eval(
            raw_encdec=_ZeroChoiceModel(),
            tokenizer=_MixedLengthTokenizer(),
            device="cpu",
            trc=PooledTargetRealConfig(
                name="pooled",
                cache_dir=tmp_path / "cache",
                pooling_op=PoolingOp.BUCKET_MAX,
            ),
            provider=provider,
            max_per_task=-1,
        )

    assert provider.lookup_lengths == []
