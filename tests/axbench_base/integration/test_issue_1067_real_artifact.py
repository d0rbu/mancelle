"""Read-only retained-artifact regression for authenticated Stage 1 output."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

import pytest
from safetensors import safe_open

from steering.eval.axbench_base.synth.continuation_guards import (
    ContinuationProtocolError,
    prefix_echo_text,
    prefix_suffix_echo_chars,
)
from steering.eval.axbench_base.synth.continuation_protocol import (
    build_stage1_continuation_request,
    parse_stage1_continuation_response,
)
from steering.eval.axbench_base.templates import load_template

_HEAD_COPY_SEED_INDEX = 23_654
_BRIDGE_COPY_SEED_INDEX = 11_037
_HEAD_COPY_KEY = (
    "bd6cb941ae6ee81b01191567806526363df58480fcee6a15225064302b9a4131"
)
_BRIDGE_COPY_KEY = (
    "3c66bbaa19e0fbc574c2b45ad46f3c125fc47ba9b5dd12923a404b7f57487e81"
)
_REGURGITATION_KEY = (
    "9796de71557874f1fab99c2c6509474b92c55c32df1fa1792b0636e61fa7db2e"
)
_HEAD_COPY_PREFIX_SHA256 = (
    "0cd00ba20a3448c3ba2d3cc8fe2c5b897896e1438bdd348ec799aa2f852db06b"
)
_BRIDGE_COPY_PREFIX_SHA256 = (
    "ea35eae6076573aaa22b3c0c66c7aaa10dbdca6af1aa744279710b38fe5aeb1f"
)
_RESPONSE_IDENTITIES = {
    _HEAD_COPY_KEY: (
        271,
        "d5d173788488e555616f96b3d1a5e696473bca0df9e55bb840f41cd269d37dea",
    ),
    _BRIDGE_COPY_KEY: (
        287,
        "1bd76a4768fb5538b73c16e75e518448d584b8413c6e3f98aeda97b1981521ec",
    ),
    _REGURGITATION_KEY: (
        2_521,
        "9340f106d200db84e98934a362464791275bd69a6bd4e3e05d9bab27397222dd",
    ),
}


class _CharTokenizer:
    bos_token_id = 0
    all_special_ids = [0]

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        del add_special_tokens
        return [ord(char) + 1 for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id - 1) for token_id in token_ids)


def _real_run_dir() -> tuple[Path, bool]:
    configured = os.environ.get("AXBENCH_ISSUE_1067_RUN_DIR")
    if configured:
        return Path(configured).expanduser().resolve(), True
    repo_root = Path(__file__).resolve().parents[3]
    return (
        repo_root
        / "nanochat_cache/eval/axbench_base/l22_f65k_clustered_std50",
        False,
    )


def _read_prefixes(path: Path) -> dict[int, str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        offsets = handle.get_tensor("prefix_text_offsets").tolist()
        text_bytes = (
            handle.get_tensor("prefix_text_bytes").numpy().tobytes()
        )
    return {
        seed_index: text_bytes[
            offsets[seed_index] : offsets[seed_index + 1]
        ].decode("utf-8", errors="strict")
        for seed_index in (_HEAD_COPY_SEED_INDEX, _BRIDGE_COPY_SEED_INDEX)
    }


def _read_responses(path: Path) -> dict[str, str]:
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        return {
            key: str(
                connection.execute(
                    "SELECT response FROM cache WHERE key = ?",
                    (key,),
                ).fetchone()[0]
            )
            for key in _RESPONSE_IDENTITIES
        }
    finally:
        connection.close()


def _frame(request, prefix_text: str, continuation: str) -> str:
    return (
        request.envelope.begin
        + "\n"
        + prefix_echo_text(prefix_text)
        + continuation
        + " "
        + request.envelope.end
    )


def _parse(request, prefix_text: str, continuation: str) -> str:
    return parse_stage1_continuation_response(
        response=_frame(request, prefix_text, continuation),
        envelope=request.envelope,
        prefix_text=prefix_text,
        output_length=len(continuation),
        tokenizer=_CharTokenizer(),
        subject="retained Stage 1 response",
    )


@pytest.mark.very_slow
@pytest.mark.requires_feature_data
def test_issue_1067_retained_failures_and_valid_boundary_are_exact() -> None:
    run_dir, explicitly_configured = _real_run_dir()
    seeds_path = run_dir / "seeds_train.safetensors"
    cache_path = run_dir / "synth_response_cache.sqlite"
    missing = [path for path in (seeds_path, cache_path) if not path.is_file()]
    if missing:
        if explicitly_configured:
            pytest.fail(f"Audited Issue #1067 artifacts are missing: {missing}")
        pytest.skip(
            "set AXBENCH_ISSUE_1067_RUN_DIR to the audited "
            "l22_f65k_clustered_std50 run directory"
        )

    prefixes = _read_prefixes(seeds_path)
    assert hashlib.sha256(
        prefixes[_HEAD_COPY_SEED_INDEX].encode("utf-8")
    ).hexdigest() == _HEAD_COPY_PREFIX_SHA256
    assert hashlib.sha256(
        prefixes[_BRIDGE_COPY_SEED_INDEX].encode("utf-8")
    ).hexdigest() == _BRIDGE_COPY_PREFIX_SHA256

    responses = _read_responses(cache_path)
    for key, (expected_length, expected_sha256) in (
        _RESPONSE_IDENTITIES.items()
    ):
        assert len(responses[key]) == expected_length
        assert hashlib.sha256(
            responses[key].encode("utf-8")
        ).hexdigest() == expected_sha256

    template = load_template("synth_continue_with_concept.txt")
    head_prefix = prefixes[_HEAD_COPY_SEED_INDEX]
    head_request = build_stage1_continuation_request(
        template,
        label="retained head-copy response",
        kind="STAGE1B_RICH",
        identity={"fixture": "head-copy"},
        prefix_text=head_prefix,
        description="retained artifact regression",
    )
    with pytest.raises(
        ContinuationProtocolError,
        match="copied a substantial source phrase",
    ):
        _parse(head_request, head_prefix, responses[_HEAD_COPY_KEY])
    with pytest.raises(
        ContinuationProtocolError,
        match="copied a substantial source phrase",
    ):
        _parse(head_request, head_prefix, responses[_REGURGITATION_KEY])

    bridge_prefix = prefixes[_BRIDGE_COPY_SEED_INDEX]
    bridge_response = responses[_BRIDGE_COPY_KEY]
    assert prefix_suffix_echo_chars(bridge_prefix, bridge_response) == 73
    bridge_request = build_stage1_continuation_request(
        template,
        label="retained bridge response",
        kind="STAGE1B_RICH",
        identity={"fixture": "bridge-copy"},
        prefix_text=bridge_prefix,
        description="retained artifact regression",
    )
    with pytest.raises(
        ContinuationProtocolError,
        match="repeated 73 character",
    ):
        _parse(bridge_request, bridge_prefix, bridge_response)

    novel_continuation = bridge_response[73:]
    assert _parse(
        bridge_request,
        bridge_prefix,
        novel_continuation,
    ) == novel_continuation
