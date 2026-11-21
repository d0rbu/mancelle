"""BOS-token guards for AxBench-BPB-Adv adversarial token views."""

from __future__ import annotations

from collections.abc import Sequence

from steering.eval.axbench_base.tokenizer_utils import assert_bos_prefix
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError


def assert_single_bos_prefix(
    tokens: Sequence[int],
    *,
    bos_id: int,
    context: str,
) -> list[int]:
    """Validate a leading single BOS and preserve the package error type."""

    token_list = [int(token_id) for token_id in tokens]
    try:
        assert_bos_prefix(token_list, int(bos_id))
    except AxBenchBaseError as exc:
        raise AxBenchBpbAdvError(f"{context}: {exc}") from exc
    return token_list


def assert_no_bos_payload(
    tokens: Sequence[int],
    *,
    bos_id: int,
    context: str,
) -> list[int]:
    """Validate that adversarial payload tokens do not contain BOS."""

    token_list = [int(token_id) for token_id in tokens]
    positions = [
        position
        for position, token_id in enumerate(token_list)
        if token_id == int(bos_id)
    ]
    if positions:
        raise AxBenchBpbAdvError(
            f"{context}: double BOS/special-token leak; BOS token id {bos_id} "
            f"appears inside the payload at position(s) {positions[:5]}"
        )
    return token_list


def assert_bos_padded_tokens(
    tokens: Sequence[int],
    *,
    bos_id: int,
    context: str,
) -> list[int]:
    """Validate ``[BOS, *payload]`` and reject BOS inside the payload."""

    token_list = assert_single_bos_prefix(tokens, bos_id=bos_id, context=context)
    assert_no_bos_payload(token_list[1:], bos_id=bos_id, context=context)
    return token_list


def bos_padded_tokens(
    payload_tokens: Sequence[int],
    *,
    bos_id: int,
    context: str,
) -> list[int]:
    """Build and validate a ``[BOS, *payload]`` token view."""

    payload = assert_no_bos_payload(
        payload_tokens,
        bos_id=bos_id,
        context=context,
    )
    padded = [int(bos_id), *payload]
    return assert_bos_padded_tokens(padded, bos_id=bos_id, context=context)


def strip_bos_padded_tokens(
    tokens: Sequence[int],
    *,
    bos_id: int,
    context: str,
) -> list[int]:
    """Validate and strip a ``[BOS, *payload]`` token view."""

    token_list = assert_bos_padded_tokens(tokens, bos_id=bos_id, context=context)
    return token_list[1:]
