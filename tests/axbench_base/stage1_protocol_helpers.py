"""Test-only helpers for deterministic authenticated Stage 1 responses."""

from __future__ import annotations

import re

TEST_STAGE1_TEMPLATE = """\
PREFIX={{PREFIX_TEXT}}
DESCRIPTION={{DESCRIPTION}}
BEGIN={{BEGIN_MARKER}}
END={{END_MARKER}}
ECHO_CHARS={{ECHO_CHARS}}
SUFFIX:
{{PREFIX_SUFFIX}}
"""

_BEGIN_RE = re.compile(r"\[\[AXBENCH_[A-Z0-9_]+_BEGIN_[0-9A-F]{24}\]\]")
_END_RE = re.compile(r"\[\[AXBENCH_[A-Z0-9_]+_END_[0-9A-F]{24}\]\]")
_ECHO_CHARS_RE = re.compile(r"(?:ECHO_CHARS=|Copy all )(\d+)")


def frame_stage1_response(prompt: str, continuation: str) -> str:
    """Wrap canned continuation text using the exact attempt prompt contract."""
    begin_match = _BEGIN_RE.search(prompt)
    end_match = _END_RE.search(prompt)
    echo_chars_match = _ECHO_CHARS_RE.search(prompt)
    if begin_match is None or end_match is None or echo_chars_match is None:
        raise AssertionError("test prompt does not contain a Stage 1 protocol contract")
    echo_chars = int(echo_chars_match.group(1))
    echo = prompt[-echo_chars:] if echo_chars else ""
    return (
        f"{begin_match.group(0)}\n"
        f"{echo}{continuation} {end_match.group(0)}"
    )


def unframe_stage1_response(
    response: str,
    *,
    begin: str,
    end: str,
    prefix_text: str,
) -> str:
    """Extract a canned body from the exact framing emitted above."""
    payload = response.removeprefix(begin + "\n")
    echo = prefix_text[-96:]
    if not payload.startswith(echo):
        raise AssertionError("test response did not repeat the expected suffix")
    payload = payload[len(echo):]
    return payload.removesuffix(" " + end)
