"""Real processes must not interleave conversion publication in one directory."""

import multiprocessing
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
import pytest

from errors import DataLoadError
from scripts.preprocess.convert_sft_to_parquet import convert_mixture_to_parquet
from tests.tasks.test_convert_sft_to_parquet import (
    FakeTokenizer,
    _convert_with_fake_tokenizer,
    _make_fake_mixture,
)


class _PausedTokenizer(FakeTokenizer):
    def __init__(self, entered, release, failure):
        self.entered = entered
        self.release = release
        self.failure = failure

    def render_conversation(self, conversation):
        self.entered.set()
        if not self.release.wait(timeout=60):
            raise TimeoutError("parent did not release conversion owner")
        if self.failure:
            raise RuntimeError("injected rendering failure")
        return super().render_conversation(conversation)


def _conversion_owner(root, entered, release, fail, result):
    tokenizer = _PausedTokenizer(entered, release, fail)
    with patch("scripts.preprocess.convert_sft_to_parquet.get_tokenizer", return_value=tokenizer):
        try:
            convert_mixture_to_parquet(
                _make_fake_mixture(4), root, max_conversations=4, conversations_per_shard=10
            )
        except RuntimeError as exc:
            result.put(str(exc))
        else:
            result.put("completed")


@pytest.mark.parametrize(
    ("finish", "use_alias"),
    [("success", False), ("failure", False), ("terminate", False), ("success", True)],
)
def test_concurrent_conversion_refuses_before_work_and_releases_lock(
    tmp_path: Path, monkeypatch, finish, use_alias
):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    context = multiprocessing.get_context("spawn")
    entered, release = context.Event(), context.Event()
    result = context.Queue()
    root = tmp_path / "conversion"
    owner = context.Process(
        target=_conversion_owner, args=(root, entered, release, finish == "failure", result)
    )
    owner.start()
    try:
        assert entered.wait(timeout=60), "owner did not reach the renderer"
        before = {path.name: path.read_bytes() for path in root.iterdir()}
        contender_path = root
        if use_alias:
            contender_path = tmp_path / "same-directory"
            contender_path.symlink_to(root, target_is_directory=True)
        with patch(
            "scripts.preprocess.convert_sft_to_parquet.get_tokenizer",
            side_effect=AssertionError("contending writer started work"),
        ):
            with pytest.raises(DataLoadError, match="Another SFT conversion is already writing"):
                convert_mixture_to_parquet(
                    _make_fake_mixture(4),
                    contender_path,
                    max_conversations=4,
                    conversations_per_shard=10,
                )
        assert {path.name: path.read_bytes() for path in root.iterdir()} == before
        if finish == "terminate":
            owner.terminate()
        else:
            release.set()
        owner.join(timeout=60)
        assert not owner.is_alive()
        if finish != "terminate":
            assert owner.exitcode == 0
            assert result.get(timeout=5) == (
                "injected rendering failure" if finish == "failure" else "completed"
            )
        # Retry from a fresh process owner after normal completion, exception,
        # or OS termination. There must be no stale lock file to clean up.
        _convert_with_fake_tokenizer(
            _make_fake_mixture(4), root, max_conversations=4, conversations_per_shard=10
        )
        assert pq.read_table(root / "shard_00000.parquet")["original_index"].to_pylist() == [
            0,
            1,
            2,
            3,
        ]
        assert sorted(path.name for path in root.iterdir()) == [
            "conversion_meta.json",
            "shard_00000.parquet",
        ]
    finally:
        # A killed process may have held the Event condition lock. Never touch
        # its synchronization primitives after termination.
        if owner.is_alive():
            owner.terminate()
        owner.join(timeout=60)
        result.close()
        result.join_thread()
