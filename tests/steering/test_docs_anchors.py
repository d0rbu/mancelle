"""Slice-02 audit B-13: AGENTS.md links to architecture.md anchors that exist.

Phase 0 surfaced ``AGENTS.md:37`` referencing
``docs/reference/architecture.md#tokenizer-warning`` — an anchor that did not
exist in the architecture doc. Slice-02 plan Cluster D adds a "Tokenizer
Warning" section to architecture.md so the link resolves.

This test:
1. Asserts AGENTS.md still references the anchor.
2. Asserts architecture.md contains the matching heading.
3. Future-proofs by also asserting the heading slug matches the anchor.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_MD = REPO_ROOT / "AGENTS.md"
ARCHITECTURE_MD = REPO_ROOT / "docs" / "reference" / "architecture.md"

ANCHOR = "tokenizer-warning"


def _slugify(heading: str) -> str:
    """Replicate GitHub's heading-to-anchor algorithm closely enough for our
    purposes: lower, replace whitespace with ``-``, strip non-word/non-hyphen.
    """
    s = heading.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s_]+", "-", s, flags=re.UNICODE)
    return s.strip("-")


def test_agents_md_references_tokenizer_warning_anchor() -> None:
    text = AGENTS_MD.read_text()
    assert f"architecture.md#{ANCHOR}" in text, (
        f"AGENTS.md must continue to link to "
        f"docs/reference/architecture.md#{ANCHOR}; if you removed the link, "
        f"also drop the architecture.md section."
    )


def test_architecture_md_has_tokenizer_warning_heading() -> None:
    text = ARCHITECTURE_MD.read_text()
    pattern = re.compile(r"^#{1,6} +(.*)$", flags=re.MULTILINE)
    headings = pattern.findall(text)
    slugs = [_slugify(h) for h in headings]
    assert ANCHOR in slugs, (
        f"docs/reference/architecture.md must contain a heading whose slug "
        f"resolves to ``#{ANCHOR}``. Current heading slugs: {slugs}."
    )


def test_architecture_md_tokenizer_warning_describes_bridge() -> None:
    """Sanity check: the section actually describes the cross-tokenizer
    bridge. Catches a future contributor renaming the anchor target without
    keeping the content useful."""
    text = ARCHITECTURE_MD.read_text()
    # Find the Tokenizer Warning section content.
    match = re.search(
        r"^#{1,6}\s+Tokenizer Warning\s*\n(.*?)^#{1,6}\s+",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        pytest.fail(
            "Could not locate the Tokenizer Warning section content "
            "(no following heading found?). Ensure the section exists "
            "and is followed by another heading."
        )
    body = match.group(1)
    # Required mentions: Gemma, RustBPE, char, checksum.
    for required in ("Gemma", "RustBPE", "char", "checksum"):
        assert required in body, (
            f"Tokenizer Warning section must mention {required!r}; got "
            f"body: {body!r}"
        )
