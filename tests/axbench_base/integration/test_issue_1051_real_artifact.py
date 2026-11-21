"""Read-only production-artifact acceptance regression for Issue #1051.

Run against the audited Stage 0b pool and Stage 1 response cache::

    AXBENCH_ISSUE_1051_RUN_DIR=/abs/path/to/l22_f65k_clustered_std50 \
      TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
      uv run pytest \
        tests/axbench_base/integration/test_issue_1051_real_artifact.py \
        -v -m 'very_slow and requires_feature_data'

The SQLite database is opened in immutable read-only mode. The test never
constructs ``SynthResponseCache`` because its constructor is intentionally a
write-capable cache initializer.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from safetensors import safe_open

from steering.eval.axbench_base.prefix_text import window_prefix_text
from steering.eval.axbench_base.synth.continuation_guards import (
    source_phrase_echo,
    validate_no_source_echo_or_repeated_phrase,
)
from steering.eval.axbench_base.synth.llm_response_cache import make_cache_key
from steering.eval.axbench_base.synth.response_parser import get_extraction_tokenizer
from steering.eval.axbench_base.templates import _safe_substitute
from steering.eval.axbench_base.types import AxBenchBaseError

_SEED_INDEX = 20_968
_CONCEPT_ID = 0
_FEATURE_INDEX = 488
_MAX_PREFIX_CHARS = 2_048
_OUTPUT_LENGTH = 64
_SYNTH_MODEL = "inclusionai/ling-2.6-flash"
_TOKENIZER_ID = "google/gemma-3-4b-pt"
_CONCEPT_DESCRIPTION = "first name last name"
_LEGACY_BOUNDARY_PUNCT = frozenset(".,;:!?)]}\"'")
_CLUSTER_FINGERPRINT = (
    "1b4ecc4908b77eb09b8285db395f6b8dd0954304a5b0ae2e797a3d1f386bb44d"
)
_SEED_FINGERPRINT = (
    "3baacbd1fb60bb73aef45cd6d1e7e38332b206cc87b8aca3369e62646ef59b80"
)
_FULL_PREFIX_SHA256 = (
    "a5d27eae1b2d30dbbe8e15c03243eb5243551fa2b8dcc7d1b60763362934045b"
)
_PARSED_CONTINUATION_SHA256 = (
    "cadab6e1623750c1cd4b8a6b6115c8f6d5d4365ece20a6b08d808c09caa63cfc"
)


def _parse_legacy_continuation_response(
    prefix_text: str,
    response: str,
    output_length: int,
    tokenizer: Any,
) -> str:
    """Reproduce the retired unauthenticated parser for the audited artifact."""
    response = response.strip()
    stripped_prefix = prefix_text.strip()
    if stripped_prefix and response.startswith(stripped_prefix):
        tail = response[len(stripped_prefix):]
        if not tail or tail[0].isspace() or tail[0] in _LEGACY_BOUNDARY_PUNCT:
            response = tail.lstrip()
    response_tokens = tokenizer.encode(
        response,
        add_special_tokens=False,
    )
    return tokenizer.decode(response_tokens[:output_length]).strip()


_OLD_CACHE_KEYS = (
    "0f1e52ac6addce0e2dbd51926cd5af4db97c903b589cec7723b233895e5a5fee",
    "f4a5bf82747ed07919ec195ca0ac1a5bc9bc20be74a860aa18e25c4cc81e38c3",
    "4fe1b4a0f8c47089e83dff6dd73da7c1dce21bc8bcceaf0d1d51f7ea182719b6",
    "c418874a90b063b09464ad4a93e6ba870bf01918f1b1698e30f9857ef07c0ad4",
)
_RAW_RESPONSE_LENGTHS = (2_998, 2_955, 3_000, 3_089)
_RAW_RESPONSE_SHA256 = (
    "58320eb2cb9609e3468597852e31e1e24c46322fb1eec040a05fba1d66cb0ac2",
    "33feaf9eb9fd70db597c0452393f98a54c29389830a441c160be96d38b78149e",
    "f21320b42ffde5e776236a5b145e1b6c87db06d4922f9314a5b6afbe717d3c81",
    "24437e8a16072ed73223dad22c5a93aed5119403958fbd15c439b9c463977fa5",
)
_LEGACY_SYNTH_TEMPLATE = """Given the partial content:

{{PREFIX_TEXT}}

Your task is to:

1. Complete the content by adding elements that are related to '{{DESCRIPTION}}'.
2. Try to avoid copying words from the definition of '{{DESCRIPTION}}' if possible.
3. Ensure that the continuation relates to '{{DESCRIPTION}}', even if the overall meaning is not fully coherent.

**Formatting Guidelines:**

- Return only the continuation to the content.
- Write the final content in plain text.
- Do not include any additional text, explanations, or formatting.

**Final Answer:** Return only the final content, following the guidelines above.
"""


def _real_run_dir() -> tuple[Path, bool]:
    configured = os.environ.get("AXBENCH_ISSUE_1051_RUN_DIR")
    if configured:
        return Path(configured).expanduser().resolve(), True
    repo_root = Path(__file__).resolve().parents[3]
    return (
        repo_root
        / "nanochat_cache/eval/axbench_base/l22_f65k_clustered_std50",
        False,
    )


def _required_artifacts(run_dir: Path) -> dict[str, Path]:
    return {
        "seeds": run_dir / "seeds_train.safetensors",
        "cache": run_dir / "synth_response_cache.sqlite",
        "concepts": run_dir / "concepts.json",
        "cluster_provenance": (
            run_dir / "candidate_concepts.cluster_provenance.json"
        ),
    }


def _load_prefix_artifact(
    path: Path,
) -> tuple[str, list[int], dict[str, str]]:
    """Load only the 52.5 MB text column and offsets, never the 1.2 GB CSR."""
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        offsets = [
            int(value)
            for value in handle.get_tensor("prefix_text_offsets").tolist()
        ]
        text_bytes = (
            handle.get_tensor("prefix_text_bytes").numpy().tobytes()
        )
    start = offsets[_SEED_INDEX]
    end = offsets[_SEED_INDEX + 1]
    prefix_text = text_bytes[start:end].decode("utf-8", errors="strict")
    character_lengths = [
        len(text_bytes[start:end].decode("utf-8", errors="strict"))
        for start, end in itertools.pairwise(offsets)
    ]
    return prefix_text, character_lengths, metadata


def _read_cached_responses(
    path: Path,
    keys: tuple[str, ...],
) -> list[str | None]:
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        return [
            (
                None
                if (row := connection.execute(
                    "SELECT response FROM cache WHERE key = ?",
                    (key,),
                ).fetchone()) is None
                else str(row[0])
            )
            for key in keys
        ]
    finally:
        connection.close()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.very_slow
@pytest.mark.requires_feature_data
def test_issue_1051_real_artifact_reproduces_and_closes_cache_failure() -> None:
    run_dir, explicitly_configured = _real_run_dir()
    artifacts = _required_artifacts(run_dir)
    missing = [path for path in artifacts.values() if not path.is_file()]
    if missing:
        if explicitly_configured:
            pytest.fail(f"Audited Issue #1051 artifacts are missing: {missing}")
        pytest.skip(
            "set AXBENCH_ISSUE_1051_RUN_DIR to the audited "
            "l22_f65k_clustered_std50 run directory"
        )

    full_prefix, character_lengths, metadata = _load_prefix_artifact(
        artifacts["seeds"]
    )
    assert len(character_lengths) == 23_953
    assert metadata["schema_version"] == "2"
    assert metadata["split"] == "train"
    assert metadata["seed_fingerprint"] == _SEED_FINGERPRINT
    assert len(full_prefix) == 2_499
    assert _sha256_text(full_prefix) == _FULL_PREFIX_SHA256

    assert min(character_lengths) == 197
    assert max(character_lengths) == 3_323
    assert sum(character_lengths) == 52_257_907
    sorted_lengths = sorted(character_lengths)
    # These indices pin the issue's NumPy-linear p50/p90/p95/p99 values.
    # Both interpolation endpoints are equal at every asserted percentile,
    # so this remains exact without depending on a percentile library.
    assert (sorted_lengths[11_976], sorted_lengths[11_977]) == (2_293, 2_293)
    assert (sorted_lengths[21_556], sorted_lengths[21_557]) == (2_642, 2_642)
    assert (sorted_lengths[22_754], sorted_lengths[22_755]) == (2_726, 2_726)
    assert (sorted_lengths[23_712], sorted_lengths[23_713]) == (2_887, 2_887)
    assert sum(length > _MAX_PREFIX_CHARS for length in character_lengths) == (
        16_532
    )

    concepts: Any = json.loads(
        artifacts["concepts"].read_text(encoding="utf-8")
    )
    assert isinstance(concepts, list)
    concept = concepts[_CONCEPT_ID]
    assert concept["concept_id"] == _CONCEPT_ID
    assert concept["feature_index"] == _FEATURE_INDEX
    assert concept["description"] == _CONCEPT_DESCRIPTION
    cluster_provenance: Any = json.loads(
        artifacts["cluster_provenance"].read_text(encoding="utf-8")
    )
    assert (
        cluster_provenance["feature_contract_fingerprint"]
        == _CLUSTER_FINGERPRINT
    )

    # These immutable cache keys were derived from the pre-#1067 template.
    # Keep that historical request bytestring frozen here: loading the current
    # authenticated prompt would make this #1051 artifact audit self-mutating.
    template = _LEGACY_SYNTH_TEMPLATE
    full_prompt = _safe_substitute(
        template,
        PREFIX_TEXT=full_prefix,
        DESCRIPTION=_CONCEPT_DESCRIPTION,
    )
    cluster_suffix = f"_fc{_CLUSTER_FINGERPRINT}"
    computed_old_keys = tuple(
        make_cache_key(
            model=_SYNTH_MODEL,
            prompt=full_prompt,
            tag=(
                f"pos_c{_CONCEPT_ID}_s{_SEED_INDEX}_a{attempt}"
                f"{cluster_suffix}"
            ),
        )
        for attempt in range(4)
    )
    assert len(full_prompt) == 3_107
    assert computed_old_keys == _OLD_CACHE_KEYS

    cached = _read_cached_responses(artifacts["cache"], computed_old_keys)
    assert all(response is not None for response in cached)
    responses = [str(response) for response in cached]
    assert tuple(map(len, responses)) == _RAW_RESPONSE_LENGTHS
    assert tuple(map(_sha256_text, responses)) == _RAW_RESPONSE_SHA256

    tokenizer = get_extraction_tokenizer(_TOKENIZER_ID)
    parsed = [
        _parse_legacy_continuation_response(
            full_prefix,
            response,
            _OUTPUT_LENGTH,
            tokenizer,
        )
        for response in responses
    ]
    assert len(set(parsed)) == 1
    continuation = parsed[0]
    assert len(continuation) == 361
    assert len(
        tokenizer.encode(continuation, add_special_tokens=False)
    ) == _OUTPUT_LENGTH
    assert _sha256_text(continuation) == _PARSED_CONTINUATION_SHA256
    assert full_prefix.find(continuation) == 1
    assert source_phrase_echo(full_prefix, continuation) is not None
    with pytest.raises(
        AxBenchBaseError,
        match=r"copied a substantial source phrase",
    ):
        validate_no_source_echo_or_repeated_phrase(
            continuation,
            source_text=full_prefix,
            subject="Stage 1b concept_rich continuation",
        )

    bounded_prefix = window_prefix_text(
        full_prefix,
        len(full_prefix),
        _MAX_PREFIX_CHARS,
    )
    assert bounded_prefix == full_prefix[-_MAX_PREFIX_CHARS:]
    assert len(bounded_prefix) == _MAX_PREFIX_CHARS
    assert continuation not in bounded_prefix
    assert source_phrase_echo(bounded_prefix, continuation) is None
    bounded_prompt = _safe_substitute(
        template,
        PREFIX_TEXT=bounded_prefix,
        DESCRIPTION=_CONCEPT_DESCRIPTION,
    )
    bounded_keys = tuple(
        make_cache_key(
            model=_SYNTH_MODEL,
            prompt=bounded_prompt,
            tag=(
                f"pos_c{_CONCEPT_ID}_s{_SEED_INDEX}_a{attempt}"
                f"{cluster_suffix}"
            ),
        )
        for attempt in range(4)
    )
    assert len(bounded_prompt) == 2_656
    assert set(bounded_keys).isdisjoint(computed_old_keys)

    # The different request identity makes the historical full-prefix
    # head-copy unreachable after bounding. Later resumptions may legitimately
    # populate the bounded keys, so cache absence is not an immutable property.
    # Any model copy of the bounded source itself remains rejected.
    with pytest.raises(
        AxBenchBaseError,
        match=r"copied a substantial source phrase",
    ):
        validate_no_source_echo_or_repeated_phrase(
            bounded_prefix,
            source_text=bounded_prefix,
            subject="Stage 1b concept_rich continuation",
        )
