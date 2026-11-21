import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nanochat.tokenizer import get_tokenizer
from scripts.chat_tui import convo
from scripts.chat_tui.convo import (
    ConvoCatalog,
    assert_renderable,
    default_val_parquet_dir,
    prefix_headroom_error,
    prefix_messages,
    verify_against_parquet,
)
from scripts.preprocess.convert_sft_to_parquet import (
    render_conversation_to_full_nanochat_text,
)
from steering.feature_paths import resolve_data_dir


class StubMixture:
    def __init__(self, conversations: list[dict]) -> None:
        self.conversations = conversations
        self.requested_indices: list[int] = []

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, index: int) -> dict:
        self.requested_indices.append(index)
        return self.conversations[index]


@pytest.fixture(scope="module")
def tokenizer():
    previous = os.environ.get("NANOCHAT_BASE_DIR")
    os.environ["NANOCHAT_BASE_DIR"] = str(_shared_nanochat_base_dir())
    try:
        yield get_tokenizer()
    finally:
        if previous is None:
            os.environ.pop("NANOCHAT_BASE_DIR", None)
        else:
            os.environ["NANOCHAT_BASE_DIR"] = previous


def _shared_nanochat_base_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "nanochat_cache"
        if (candidate / "tokenizer" / "tokenizer.pkl").is_file():
            return candidate
    return Path(__file__).resolve().parents[2] / "nanochat_cache"


def _resolve_real_parquet_dir() -> Path:
    previous = os.environ.get("NANOCHAT_BASE_DIR")
    os.environ["NANOCHAT_BASE_DIR"] = str(_shared_nanochat_base_dir())
    try:
        return resolve_data_dir(None, "standard_sft_val")
    finally:
        if previous is None:
            os.environ.pop("NANOCHAT_BASE_DIR", None)
        else:
            os.environ["NANOCHAT_BASE_DIR"] = previous


def _conversation(user: str = "Question", assistant: str = "Answer") -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def _load_stub_catalog(monkeypatch, tokenizer, conversations: list[dict]):
    mixture = StubMixture(conversations)
    monkeypatch.setattr(
        convo,
        "build_task_mixtures",
        lambda _mixture: (None, mixture),
    )
    return ConvoCatalog.load(tokenizer), mixture


def _write_probe_shard(
    parquet_dir: Path,
    *,
    original_index: int,
    text: str,
) -> None:
    parquet_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"text": [text], "original_index": [original_index]}),
        parquet_dir / "shard_00000.parquet",
    )


def test_catalog_is_lazy_and_caches_entries(monkeypatch, tokenizer) -> None:
    conversations = [
        _conversation("First question", "First answer"),
        _conversation("Second question", "Second answer"),
    ]
    first_catalog, first_mixture = _load_stub_catalog(monkeypatch, tokenizer, conversations)

    assert len(first_catalog) == 2
    assert first_mixture.requested_indices == []

    first_entry = first_catalog.entry(1)
    assert first_catalog.entry(1) is first_entry
    assert first_mixture.requested_indices == [1]

    assert first_entry.index == 1
    assert first_entry.n_user_turns == 1
    assert first_entry.skip_reason is None
    assert isinstance(first_entry.messages, tuple)


def test_real_mixture_catalog_loads_deterministically(tokenizer) -> None:
    try:
        first = ConvoCatalog.load(tokenizer)
        second = ConvoCatalog.load(tokenizer)
        indices = (0, 1, 2, 137)
        first_rows = [
            (
                first.entry(index).index,
                first.entry(index).token_count,
                first.entry(index).n_user_turns,
                first.entry(index).first_user_snippet,
            )
            for index in indices
        ]
        second_rows = [
            (
                second.entry(index).index,
                second.entry(index).token_count,
                second.entry(index).n_user_turns,
                second.entry(index).first_user_snippet,
            )
            for index in indices
        ]
    except (ImportError, ConnectionError, OSError) as exc:
        pytest.skip(f"standard SFT validation mixture is unavailable: {exc}")

    assert first_rows == second_rows


def test_load_rejects_missing_validation_mixture(monkeypatch, tokenizer) -> None:
    monkeypatch.setattr(
        convo,
        "build_task_mixtures",
        lambda _mixture: (StubMixture([]), None),
    )

    with pytest.raises(ValueError, match="no validation mixture"):
        ConvoCatalog.load(tokenizer)


def test_prefix_messages_supports_every_turn_boundary(monkeypatch, tokenizer) -> None:
    conversation = {
        "messages": [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
    }
    catalog, _ = _load_stub_catalog(monkeypatch, tokenizer, [conversation])
    entry = catalog.entry(0)

    assert prefix_messages(entry, 0) == []
    assert prefix_messages(entry, 1) == conversation["messages"][:2]
    assert prefix_messages(entry, entry.n_user_turns) == conversation["messages"]
    assert isinstance(prefix_messages(entry, 1), list)


def test_system_leading_prefixes_are_role_positioned_and_renderable(
    monkeypatch,
    tokenizer,
) -> None:
    conversation = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
    }
    catalog, _ = _load_stub_catalog(monkeypatch, tokenizer, [conversation])
    entry = catalog.entry(0)
    expected_roles = {
        0: [],
        1: ["system", "user", "assistant"],
        2: ["system", "user", "assistant", "user", "assistant"],
        3: [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
        ],
    }

    for start_user_turn, roles in expected_roles.items():
        prefix = prefix_messages(entry, start_user_turn)
        assert [message["role"] for message in prefix] == roles
        if prefix:
            rendered, _mask = tokenizer.render_conversation({"messages": prefix}, max_tokens=2**31)
            assert rendered


def test_prefix_headroom_validator_reports_exact_counts(monkeypatch, tokenizer) -> None:
    catalog, _ = _load_stub_catalog(
        monkeypatch,
        tokenizer,
        [
            {
                "messages": [
                    {"role": "user", "content": "u0"},
                    {"role": "assistant", "content": "a0"},
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ]
            }
        ],
    )
    entry = catalog.entry(0)
    prefix = prefix_messages(entry, 1)
    rendered, _mask = tokenizer.render_conversation({"messages": prefix}, max_tokens=2**31)
    prefix_tokens = len(rendered)
    required_tokens = prefix_tokens + 32

    assert (
        prefix_headroom_error(
            entry,
            1,
            tokenizer,
            sequence_len=required_tokens + 1,
            max_new_tokens_per_turn=5,
        )
        is None
    )
    error = prefix_headroom_error(
        entry,
        1,
        tokenizer,
        sequence_len=required_tokens,
        max_new_tokens_per_turn=5,
    )

    assert error is not None
    assert f"prefix uses {prefix_tokens} tokens" in error
    assert "32 reserved generation tokens" in error
    assert f"({required_tokens} total)" in error
    assert f"sequence_len is {required_tokens}" in error


def test_default_val_parquet_dir_is_standard_sft_family(monkeypatch) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(_shared_nanochat_base_dir()))

    assert default_val_parquet_dir() == resolve_data_dir(None, "standard_sft_val")


@pytest.mark.parametrize("start_user_turn", [-1, 3])
def test_prefix_messages_rejects_out_of_range_turn(
    monkeypatch, tokenizer, start_user_turn: int
) -> None:
    catalog, _ = _load_stub_catalog(
        monkeypatch,
        tokenizer,
        [
            {
                "messages": [
                    {"role": "user", "content": "u0"},
                    {"role": "assistant", "content": "a0"},
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ]
            }
        ],
    )

    with pytest.raises(ValueError, match="start_user_turn"):
        prefix_messages(catalog.entry(0), start_user_turn)


def test_prefix_messages_rejects_skipped_conversation(monkeypatch, tokenizer) -> None:
    catalog, _ = _load_stub_catalog(
        monkeypatch,
        tokenizer,
        [{"messages": [{"role": "user", "content": "No answer"}]}],
    )
    entry = catalog.entry(0)

    with pytest.raises(ValueError, match="skipped conversation.*no_assistant"):
        prefix_messages(entry, 0)


def test_converter_skip_reasons_are_surfaced_exactly(monkeypatch, tokenizer) -> None:
    catalog, _ = _load_stub_catalog(
        monkeypatch,
        tokenizer,
        [
            {"messages": [{"role": "user", "content": "No answer"}]},
            _conversation("Empty answer", " \n\t"),
        ],
    )

    no_assistant = catalog.entry(0)
    empty_response = catalog.entry(1)

    assert no_assistant.skip_reason == "no_assistant"
    assert no_assistant.messages == ()
    assert empty_response.skip_reason == "empty_response"
    assert empty_response.messages == ()


def test_token_count_uses_non_truncating_render(monkeypatch, tokenizer) -> None:
    catalog, _ = _load_stub_catalog(
        monkeypatch,
        tokenizer,
        [_conversation("repeated text " * 5000, "done")],
    )
    entry = catalog.entry(0)
    expected_ids, _ = tokenizer.render_conversation(
        {"messages": list(entry.messages)}, max_tokens=2**31
    )

    assert entry.token_count == len(expected_ids)
    assert entry.token_count > 2048


def test_assert_renderable_has_a_strict_2048_cap() -> None:
    assert_renderable(2047, source="test prefix")

    with pytest.raises(ValueError, match="2048"):
        assert_renderable(2048, source="test prefix")


def test_first_user_snippet_collapses_whitespace_then_truncates(monkeypatch, tokenizer) -> None:
    user = "  alpha\n\tbeta   " + "0123456789 " * 20
    catalog, _ = _load_stub_catalog(
        monkeypatch,
        tokenizer,
        [_conversation(user, "answer")],
    )
    entry = catalog.entry(0)

    assert entry.first_user_snippet == " ".join(user.split())[:80]
    assert len(entry.first_user_snippet) == 80


def test_verify_against_parquet_compares_converter_text(
    monkeypatch, tokenizer, tmp_path: Path
) -> None:
    conversation = _conversation("Probe question", "Probe answer")
    catalog, _ = _load_stub_catalog(monkeypatch, tokenizer, [conversation])
    entry = catalog.entry(0)
    expected, skip_reason = render_conversation_to_full_nanochat_text(tokenizer, conversation)
    assert skip_reason is None
    assert expected is not None
    _write_probe_shard(tmp_path, original_index=entry.index, text=expected)

    verify_against_parquet(catalog, entry, tokenizer, tmp_path)


def test_verify_against_parquet_rejects_missing_directory(
    monkeypatch, tokenizer, tmp_path: Path
) -> None:
    catalog, _ = _load_stub_catalog(monkeypatch, tokenizer, [_conversation()])
    entry = catalog.entry(0)

    with pytest.raises(ValueError, match=r"original_index=0"):
        verify_against_parquet(catalog, entry, tokenizer, tmp_path / "missing")


def test_verify_against_parquet_rejects_missing_row(monkeypatch, tokenizer, tmp_path: Path) -> None:
    catalog, _ = _load_stub_catalog(monkeypatch, tokenizer, [_conversation()])
    entry = catalog.entry(0)
    _write_probe_shard(tmp_path, original_index=99, text="not this row")

    with pytest.raises(ValueError, match=r"original_index=0"):
        verify_against_parquet(catalog, entry, tokenizer, tmp_path)


def test_verify_against_parquet_rejects_text_mismatch(
    monkeypatch, tokenizer, tmp_path: Path
) -> None:
    catalog, _ = _load_stub_catalog(monkeypatch, tokenizer, [_conversation()])
    entry = catalog.entry(0)
    _write_probe_shard(tmp_path, original_index=0, text="mutated parquet text")

    with pytest.raises(ValueError, match=r"original_index=0"):
        verify_against_parquet(catalog, entry, tokenizer, tmp_path)


REAL_PARQUET_DIR = _resolve_real_parquet_dir()
HAS_REAL_PARQUET = REAL_PARQUET_DIR.is_dir() and any(REAL_PARQUET_DIR.glob("shard_*.parquet"))


@pytest.fixture(scope="module")
def real_catalog(tokenizer):
    return ConvoCatalog.load(tokenizer)


@pytest.mark.skipif(not HAS_REAL_PARQUET, reason="standard_sft_val parquet is unavailable")
def test_real_catalog_entry_zero_has_sane_fields(real_catalog) -> None:
    entry = real_catalog.entry(0)

    assert entry.index == 0
    assert entry.n_user_turns >= 0
    assert entry.token_count >= 0
    assert len(entry.first_user_snippet) <= 80
    assert entry.skip_reason in {None, "no_assistant", "empty_response"}


@pytest.mark.skipif(not HAS_REAL_PARQUET, reason="standard_sft_val parquet is unavailable")
def test_real_catalog_first_kept_row_matches_parquet(
    real_catalog, tokenizer, tmp_path: Path
) -> None:
    entry = next(
        candidate
        for index in range(len(real_catalog))
        if (candidate := real_catalog.entry(index)).skip_reason is None
    )

    verify_against_parquet(real_catalog, entry, tokenizer, REAL_PARQUET_DIR)

    _write_probe_shard(
        tmp_path,
        original_index=entry.index,
        text="deliberately mutated parquet text",
    )
    with pytest.raises(ValueError, match=str(entry.index)):
        verify_against_parquet(real_catalog, entry, tokenizer, tmp_path)
