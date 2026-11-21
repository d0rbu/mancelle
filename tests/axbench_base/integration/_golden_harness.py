"""Shared golden-harness helpers for the Stage 1b/1c committed-golden tests.

This module is deliberately named with a leading underscore (and NO ``test_``
prefix) so pytest does NOT collect it as a test module. It holds the pieces that
``test_stage1b_avoid_golden.py`` and ``test_stage1c_avoiding_golden.py`` share
byte-for-byte: the deterministic extraction tokenizer + fake extractor, the fake
base model / SAE, the metadata builder, the nanochat-tokenizer stub, a unified
prompt-keyed synth client, and the NaN-aware row/schema comparator. Each test
supplies its own fully-built pyarrow schema to ``assert_rows_equal``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t

from steering.eval.axbench_base.types import TargetModelMetadata
from tests.axbench_base.stage1_protocol_helpers import frame_stage1_response

GOLDEN_N_FEATURES = 128
FC = 42  # concept feature index; must be < GOLDEN_N_FEATURES


class GoldenExtractionTokenizer:
    """Deterministic word-level extraction tokenizer for the golden harness.

    Tokenization is whitespace-split (``text.split()``), so behavior depends on
    whether a word survives as a standalone token:

    - ``FIRE`` -> a sentinel token id the fake extractor marks as active.
    - ``MERGE`` is an ordinary token in isolation, so the authenticated
      continuation parser accepts it. The avoid path joins prefix and
      continuation WITHOUT a space (``prefix_text + continuation_text``), so
      ``"prefix three" + "MERGE"`` becomes ``"prefix threeMERGE"`` and leaves
      the full-text token count equal to the prefix token count. That reaches
      the intended downstream ``NO_TOKENS`` seam branch rather than being
      rejected earlier by the continuation parser.
    """

    FIRE_ID = 7_000_001
    # Sentinel BOS id, far outside both the small sequential per-word vocab
    # range (starts at 2) and FIRE_ID -- qc.py's SAE-forward sites prepend
    # this via resolve_bos_token_id and exclude its row from the pool.
    bos_token_id = 999

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._words_by_id: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        for word in text.split():
            if word == "FIRE":
                ids.append(self.FIRE_ID)
            else:
                token_id = self._vocab.setdefault(word, len(self._vocab) + 2)
                self._words_by_id[token_id] = word
                ids.append(token_id)
        return ids

    def decode(self, token_ids: list[int]) -> str:
        words: list[str] = []
        for token_id in token_ids:
            if token_id == self.FIRE_ID:
                words.append("FIRE")
                continue
            try:
                words.append(self._words_by_id[token_id])
            except KeyError as exc:
                raise AssertionError(
                    f"golden tokenizer cannot decode unknown token id {token_id}"
                ) from exc
        return " ".join(words)


def golden_extractor(**kwargs: Any) -> list[t.Tensor]:
    out: list[t.Tensor] = []
    for tokens in kwargs["token_id_lists"]:
        acts = t.zeros(len(tokens), GOLDEN_N_FEATURES, dtype=t.float32)
        for i, tok in enumerate(tokens):
            if tok == GoldenExtractionTokenizer.FIRE_ID:
                acts[i, :] = 1.0
        out.append(acts)
    return out


class PromptKeyedSynthClient:
    """Order-independent synth stub returning an authenticated response.

    Subsumes both golden tests' stubs. ``by_prompt_substring`` maps a needle to
    either a single ``str`` (returned on every matching call) or a ``list[str]``
    (the i-th matching call to that needle returns ``seq[min(i, len-1)]``, so
    retries on the serial path can vary across attempts). Matching is by
    substring (the seed/prefix text is embedded in the prompt), so it is
    independent of selection order. The selected continuation body is framed
    from the request prompt's exact nonce markers and suffix echo before being
    returned, so the golden path exercises the production protocol parser.
    """

    def __init__(self, by_prompt_substring: dict[str, str | list[str]]) -> None:
        self._by = dict(by_prompt_substring)
        self._idx: dict[str, int] = {}
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        prompt: str,
        *,
        cache_key: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "cache_key": cache_key,
                "system_prompt": system_prompt,
            }
        )
        for needle, resp in self._by.items():
            if needle in prompt:
                if isinstance(resp, str):
                    return frame_stage1_response(prompt, resp)
                i = self._idx.get(needle, 0)
                self._idx[needle] = i + 1
                return frame_stage1_response(
                    prompt,
                    resp[min(i, len(resp) - 1)],
                )
        raise AssertionError(f"golden synth stub: no response for prompt {prompt!r}")


class StubNanochatTokenizer:
    """Reversible character tokenizer for exact labeled-pair boundaries."""

    def get_bos_token_id(self) -> int:
        return 1

    def encode(self, text: str, prepend: int | None = None) -> list[int]:
        base = [ord(c) + 2 for c in text]
        return ([prepend] if prepend is not None else []) + base

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id - 2) for token_id in token_ids)


def make_meta(sae_layer: int = 8) -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=8, n_sae_features=GOLDEN_N_FEATURES, negative_selection=None,
        feature_selection=None, per_feature_stats=None, device="cpu",
        sae_layer=sae_layer, segment_size=128,
    )


def fake_base_model() -> SimpleNamespace:
    # The REAL avoid path runs _run_sae_on_token_lists, which needs
    # base_model.cfg.n_ctx for the context-window clamp.
    return SimpleNamespace(cfg=SimpleNamespace(n_ctx=2048, model_name="gemma-golden"))


def fake_base_sae() -> SimpleNamespace:
    # _sae_hook_name(base_sae) is evaluated before the (mocked) extractor; it
    # needs base_sae.cfg.hook_name.
    return SimpleNamespace(
        cfg=SimpleNamespace(
            hook_name="blocks.8.hook_resid_post", release="rel", d_sae=GOLDEN_N_FEATURES,
        ),
    )


def assert_rows_equal(
    actual: list[dict[str, Any]],
    golden_path: Path,
    schema: pa.Schema,
    *,
    golden_row_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> None:
    golden_table = pq.read_table(golden_path)
    actual_table = pa.Table.from_pylist(actual, schema=schema)
    assert actual_table.schema.names == golden_table.schema.names
    assert (
        [str(actual_table.schema.field(n).type) for n in actual_table.schema.names]
        == [str(golden_table.schema.field(n).type) for n in golden_table.schema.names]
    )
    a_rows = actual_table.to_pylist()
    g_rows = golden_table.to_pylist()
    if golden_row_transform is not None:
        g_rows = [golden_row_transform(dict(row)) for row in g_rows]
    assert len(a_rows) == len(g_rows)
    for i, (ar, gr) in enumerate(zip(a_rows, g_rows, strict=True)):
        for key in gr:
            av, gv = ar[key], gr[key]
            if isinstance(av, float) and isinstance(gv, float):
                if math.isnan(av) and math.isnan(gv):
                    continue
            assert av == gv, f"row {i} key {key!r}: {av!r} != golden {gv!r}"
