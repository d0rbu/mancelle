from __future__ import annotations

from pathlib import Path

import pytest

from steering.eval.autointerp_filter.document_blacklist import (
    DOCUMENT_BLACKLIST_DIGEST_METHOD,
    EMPTY_DOCUMENT_BLACKLIST,
    EMPTY_DOCUMENT_BLACKLIST_SHA256,
    ResolvedDocumentBlacklist,
    document_key,
    parse_document_blacklist,
    resolve_document_blacklist,
)
from steering.eval.autointerp_filter.strict_json import canonical_sha256
from steering.eval.autointerp_filter.types import AutoInterpConfigError


def test_parse_document_blacklist_parses_entries_skips_comments_and_blanks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "blacklist.txt"
    path.write_text(
        "# a leading comment\n"
        "\n"
        "shard_00000_features:1731  # trailing comment\n"
        "   \n"
        "shard_00042_features:0\n"
    )

    result = parse_document_blacklist(str(path))

    assert result == frozenset(
        {
            document_key(shard_stem="shard_00000_features", doc_id=1731),
            document_key(shard_stem="shard_00042_features", doc_id=0),
        }
    )


def test_parse_document_blacklist_dedupes_duplicate_lines(tmp_path: Path) -> None:
    path = tmp_path / "blacklist.txt"
    path.write_text("shard_00000_features:1\nshard_00000_features:1\n")

    result = parse_document_blacklist(str(path))

    assert result == frozenset({document_key(shard_stem="shard_00000_features", doc_id=1)})


def test_parse_document_blacklist_empty_after_comments_is_empty_set_not_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "blacklist.txt"
    path.write_text("# nothing enabled right now\n\n")

    assert parse_document_blacklist(str(path)) == EMPTY_DOCUMENT_BLACKLIST


def test_parse_document_blacklist_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AutoInterpConfigError, match="not found"):
        parse_document_blacklist(str(tmp_path / "missing.txt"))


@pytest.mark.parametrize(
    "line",
    [
        "no_colon_here",
        "shard_00000_features:not_an_int",
        "shard_00000_features:-1",
        ":5",
        "   :5",
    ],
)
def test_parse_document_blacklist_rejects_malformed_lines(tmp_path: Path, line: str) -> None:
    path = tmp_path / "blacklist.txt"
    path.write_text(f"{line}\n")

    with pytest.raises(AutoInterpConfigError):
        parse_document_blacklist(str(path))


def test_parse_document_blacklist_shard_stem_may_contain_colons(tmp_path: Path) -> None:
    # Matches on the LAST colon, so a shard_stem containing ':' (unusual, but
    # not forbidden by shard-naming elsewhere) still parses correctly as long
    # as doc_id is the final ':'-delimited token.
    path = tmp_path / "blacklist.txt"
    path.write_text("weird:shard:stem:42\n")

    result = parse_document_blacklist(str(path))

    assert result == frozenset({document_key(shard_stem="weird:shard:stem", doc_id=42)})


def test_resolved_digest_hashes_canonical_parsed_entries(tmp_path: Path) -> None:
    path = tmp_path / "blacklist.txt"
    path.write_text(
        "shard_00042_features:7\nshard_00000_features:3\nshard_00042_features:7  # duplicate\n",
        encoding="utf-8",
    )

    first = resolve_document_blacklist(str(path))
    expected = canonical_sha256(
        {
            "method": DOCUMENT_BLACKLIST_DIGEST_METHOD,
            "entries": [
                {"shard_stem": "shard_00000_features", "doc_id": 3},
                {"shard_stem": "shard_00042_features", "doc_id": 7},
            ],
        }
    )
    assert first.sha256 == expected

    path.write_text(
        "# formatting, ordering, and duplicates are not semantic\n"
        "shard_00042_features:7\n"
        "shard_00000_features:3\n"
        "shard_00000_features:3\n",
        encoding="utf-8",
    )
    equivalent = resolve_document_blacklist(str(path))
    assert equivalent.entries == first.entries
    assert equivalent.sha256 == first.sha256

    path.write_text("shard_00000_features:4\n", encoding="utf-8")
    changed = resolve_document_blacklist(str(path))
    assert changed.entries != first.entries
    assert changed.sha256 != first.sha256


def test_disabled_and_configured_empty_blacklists_share_empty_set_digest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("# no exclusions\n", encoding="utf-8")

    disabled = resolve_document_blacklist(None)
    configured_empty = resolve_document_blacklist(str(path))

    assert disabled.entries == configured_empty.entries == EMPTY_DOCUMENT_BLACKLIST
    assert disabled.sha256 == configured_empty.sha256 == EMPTY_DOCUMENT_BLACKLIST_SHA256
    assert EMPTY_DOCUMENT_BLACKLIST_SHA256 == canonical_sha256(
        {"method": DOCUMENT_BLACKLIST_DIGEST_METHOD, "entries": []}
    )


def test_resolved_blacklist_rejects_digest_not_derived_from_entries() -> None:
    entries = frozenset({document_key(shard_stem="shard_00000_features", doc_id=1)})

    with pytest.raises(AutoInterpConfigError, match="does not match its parsed entries"):
        ResolvedDocumentBlacklist(
            configured_path="blacklist.txt",
            entries=entries,
            sha256="0" * 64,
        )


def test_resolved_blacklist_rejects_optional_disabled_digest() -> None:
    with pytest.raises(AutoInterpConfigError, match="does not match its parsed entries"):
        ResolvedDocumentBlacklist(
            configured_path=None,
            entries=EMPTY_DOCUMENT_BLACKLIST,
            sha256=None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        )
