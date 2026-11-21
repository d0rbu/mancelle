"""Document-level exclusion list for autointerp positive/intruder candidates.

Some source documents contain content that is fine as corpus text but
adversarial for LLM-judge intruder-detection prompts (for example, a stray
numbered list such as ``7. Bellwether constituencies`` sitting next to a
highlighted span, which can bias a judge into echoing the embedded number as
if it were the requested example index instead of a real answer). This
module lets specific documents be excluded from ever being sampled as a
positive or intruder candidate, independent of which target feature is
being scored.

Documents are identified by ``(shard_stem, doc_id)``, matching
:class:`~steering.eval.autointerp_filter.types.SourceDocRef`'s doc-level
identity (not :class:`~steering.eval.autointerp_filter.types.TokenWindowRef`,
whose ``stable_key()`` also bakes in the scored target's feature ids and
activations, so the same document would need one blacklist entry per target
ever tested against it).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from steering.eval.autointerp_filter.strict_json import canonical_sha256
from steering.eval.autointerp_filter.types import AutoInterpConfigError

DocumentKey = tuple[str, int]

EMPTY_DOCUMENT_BLACKLIST: frozenset[DocumentKey] = frozenset()
DOCUMENT_BLACKLIST_DIGEST_METHOD = "document_blacklist_entries_v1"


@dataclass(frozen=True)
class ResolvedDocumentBlacklist:
    """One parsed blacklist snapshot and its semantic content identity."""

    configured_path: str | None
    entries: frozenset[DocumentKey]
    sha256: str

    def __post_init__(self) -> None:
        if self.configured_path is None and self.entries:
            raise AutoInterpConfigError("disabled document blacklist must have no entries")
        expected = document_blacklist_sha256(self.entries)
        if self.sha256 != expected:
            raise AutoInterpConfigError(
                "resolved document blacklist SHA-256 does not match its parsed entries: "
                f"stored={self.sha256!r}, expected={expected}"
            )


def document_key(*, shard_stem: str, doc_id: int) -> DocumentKey:
    """Canonical doc-level identity, shared by ``SourceDocRef``/``TokenWindowRef``."""
    return (shard_stem, doc_id)


def parse_document_blacklist(path: str) -> frozenset[DocumentKey]:
    """Parse a plain-text document blacklist file.

    Format: one ``<shard_stem>:<doc_id>`` entry per line. ``#`` starts a
    comment (to end of line); blank lines are skipped. Whitespace around the
    entry is stripped. Raises :class:`AutoInterpConfigError` fail-fast on any
    malformed line rather than skipping it silently.
    """
    p = Path(path)
    if not p.is_file():
        raise AutoInterpConfigError(f"document_blacklist_path not found: {path}")
    entries: set[DocumentKey] = set()
    for line_number, raw_line in enumerate(p.read_text().splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.rsplit(":", 1)
        if len(parts) != 2:
            raise AutoInterpConfigError(
                f"document_blacklist_path {path} line {line_number} is not "
                f"'<shard_stem>:<doc_id>': {raw_line!r}"
            )
        shard_stem, doc_id_text = parts
        shard_stem = shard_stem.strip()
        if not shard_stem:
            raise AutoInterpConfigError(
                f"document_blacklist_path {path} line {line_number} has an "
                f"empty shard_stem: {raw_line!r}"
            )
        try:
            doc_id = int(doc_id_text.strip())
        except ValueError as exc:
            raise AutoInterpConfigError(
                f"document_blacklist_path {path} line {line_number} has a "
                f"non-integer doc_id: {raw_line!r}"
            ) from exc
        if doc_id < 0:
            raise AutoInterpConfigError(
                f"document_blacklist_path {path} line {line_number} has a "
                f"negative doc_id: {raw_line!r}"
            )
        entries.add(document_key(shard_stem=shard_stem, doc_id=doc_id))
    return frozenset(entries)


def document_blacklist_sha256(entries: frozenset[DocumentKey]) -> str:
    """Hash the canonical parsed exclusion set, independent of file formatting."""
    return canonical_sha256(
        {
            "method": DOCUMENT_BLACKLIST_DIGEST_METHOD,
            "entries": [
                {"shard_stem": shard_stem, "doc_id": doc_id}
                for shard_stem, doc_id in sorted(entries)
            ],
        }
    )


EMPTY_DOCUMENT_BLACKLIST_SHA256 = document_blacklist_sha256(EMPTY_DOCUMENT_BLACKLIST)


def resolve_document_blacklist(path: str | None) -> ResolvedDocumentBlacklist:
    """Parse ``path`` once and bind its effective entries to a semantic digest."""
    if path is None:
        return ResolvedDocumentBlacklist(
            configured_path=None,
            entries=EMPTY_DOCUMENT_BLACKLIST,
            sha256=EMPTY_DOCUMENT_BLACKLIST_SHA256,
        )
    entries = parse_document_blacklist(path)
    return ResolvedDocumentBlacklist(
        configured_path=path,
        entries=entries,
        sha256=document_blacklist_sha256(entries),
    )
