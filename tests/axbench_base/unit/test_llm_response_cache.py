"""Unit tests for the per-run SQLite LLM response cache (review-fix 7).

This suite is organised into sections. The early sections (1-5) are the
original small-surface tests; sections 6-17 are the comprehensive
edge-case battery added in Batch F. Every test carries a docstring
naming the invariant or edge case it exercises.

Linux-only: a handful of tests use ``os.chmod`` to synthesise
permission-denied conditions. The repo is Linux-only so no skip logic
is added.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from pathlib import Path

import pytest

from steering.eval.axbench_base.synth.llm_response_cache import (
    SynthResponseCache,
    make_cache_key,
)


def _row_count(cache: SynthResponseCache) -> int:
    with cache._lock:
        row = cache._conn.execute("SELECT COUNT(*) FROM cache").fetchone()
    return int(row[0])


# ===========================================================================
# SECTION 1 — Original: file / parent-dir semantics
# ===========================================================================


def test_cache_creates_sqlite_file(tmp_path: Path) -> None:
    """Constructor creates the parent dir and the .sqlite file on disk."""
    path = tmp_path / "nested" / "dir" / "cache.sqlite"
    assert not path.parent.exists()
    cache = SynthResponseCache(path)
    try:
        assert path.parent.is_dir()
        assert path.exists()
    finally:
        cache.close()


# ===========================================================================
# SECTION 2 — Original: basic CRUD operations
# ===========================================================================


def test_cache_get_miss_returns_none(tmp_path: Path) -> None:
    """A key that was never set should resolve to ``None`` on get."""
    cache = SynthResponseCache(tmp_path / "c.sqlite")
    try:
        assert cache.get("missing") is None
    finally:
        cache.close()


def test_cache_set_and_get_roundtrip(tmp_path: Path) -> None:
    """Setting then getting a key should return the stored value."""
    cache = SynthResponseCache(tmp_path / "c.sqlite")
    try:
        cache.set("k1", "value-1")
        assert cache.get("k1") == "value-1"
    finally:
        cache.close()


def test_cache_set_overwrites_existing(tmp_path: Path) -> None:
    """INSERT OR REPLACE semantics: setting an existing key overwrites."""
    cache = SynthResponseCache(tmp_path / "c.sqlite")
    try:
        cache.set("k1", "first")
        cache.set("k1", "second")
        assert cache.get("k1") == "second"
        assert _row_count(cache) == 1  # not duplicated
    finally:
        cache.close()


# ===========================================================================
# SECTION 4 — Original: persistence across instances
# ===========================================================================


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    """Re-opening a cache file must return values written by a prior instance."""
    path = tmp_path / "persist.sqlite"
    first = SynthResponseCache(path)
    try:
        first.set("alpha", "beta")
    finally:
        first.close()

    second = SynthResponseCache(path)
    try:
        assert second.get("alpha") == "beta"
    finally:
        second.close()


# ===========================================================================
# SECTION 5 — Original: context-manager protocol
# ===========================================================================


def test_cache_context_manager_closes(tmp_path: Path) -> None:
    """``with`` statement should open the cache and close the connection on exit."""
    path = tmp_path / "ctx.sqlite"
    with SynthResponseCache(path) as cache:
        cache.set("x", "y")
        assert cache.get("x") == "y"
    # After __exit__, the underlying connection must be closed; any further
    # use should raise sqlite3.ProgrammingError.
    with pytest.raises(sqlite3.ProgrammingError):
        cache.get("x")


# ===========================================================================
# SECTION 6 — Original: make_cache_key determinism + disambiguation
# ===========================================================================


def test_make_cache_key_deterministic() -> None:
    """Same (model, prompt, tag) should always produce the same key."""
    k1 = make_cache_key(model="gpt-4o", prompt="hello", tag="attempt-0")
    k2 = make_cache_key(model="gpt-4o", prompt="hello", tag="attempt-0")
    assert k1 == k2


def test_make_cache_key_different_model_differs() -> None:
    """Distinct models with otherwise-identical call identity must not collide."""
    k1 = make_cache_key(model="gpt-4o", prompt="hello", tag="t")
    k2 = make_cache_key(model="gpt-5", prompt="hello", tag="t")
    assert k1 != k2


def test_make_cache_key_different_prompt_differs() -> None:
    """Distinct prompts with otherwise-identical call identity must not collide."""
    k1 = make_cache_key(model="m", prompt="question A", tag="t")
    k2 = make_cache_key(model="m", prompt="question B", tag="t")
    assert k1 != k2


def test_make_cache_key_different_tag_differs() -> None:
    """Distinct tags with otherwise-identical call identity must not collide."""
    k1 = make_cache_key(model="m", prompt="p", tag="attempt-0")
    k2 = make_cache_key(model="m", prompt="p", tag="attempt-1")
    assert k1 != k2


def test_make_cache_key_is_sha256_hex() -> None:
    """The returned key should be 64 lowercase hex chars (sha256 digest)."""
    k = make_cache_key(model="m", prompt="p", tag="t")
    assert len(k) == 64
    int(k, 16)  # parseable as hex
    # And it should match the explicit sha256 formulation.
    expected = hashlib.sha256(b"m\n---\np\n---\nt").hexdigest()
    assert k == expected


# ===========================================================================
# SECTION 7 — Batch F: Unicode, encoding, and byte-exact round-trip
# ===========================================================================


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("japanese", "こんにちは世界"),
        ("emoji", "fire ok party mix"),
        ("latin1_diacritics", "café résumé naïve Grüße"),
        ("arabic_rtl", "مرحبا بالعالم"),
        ("hebrew_rtl", "שלום עולם"),
        ("mixed_script", "Hello 世界 مرحبا שלום"),
        ("zero_width_joiners", "e\u0301 a\u0301 o\u0301"),  # decomposed é á ó
        ("astral_plane", "H E L L O"),  # outside the BMP
        ("combining_marks_stack", "a\u0300\u0301\u0302\u0303\u0304"),
        ("bom_in_middle", "before\ufeffafter"),
        ("soft_hyphen", "co\u00adoperate"),
    ],
)
def test_cache_unicode_value_byte_exact_roundtrip(
    tmp_path: Path, label: str, text: str
) -> None:
    """Non-ASCII values must round-trip byte-for-byte identical."""
    cache = SynthResponseCache(tmp_path / f"{label}.sqlite")
    try:
        cache.set("k", text)
        recovered = cache.get("k")
        assert recovered == text
        assert isinstance(recovered, str)
        assert recovered is not None
        assert recovered.encode("utf-8") == text.encode("utf-8")
    finally:
        cache.close()


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("japanese", "鍵"),
        ("emoji_key", "fire key"),
        ("astral", "K E Y"),
        ("mixed", "key_مفتاح_键"),
        ("combining", "ke\u0301y"),
    ],
)
def test_cache_unicode_key_roundtrip(
    tmp_path: Path, label: str, text: str
) -> None:
    """Non-ASCII keys must be usable for get/set/contains equally."""
    cache = SynthResponseCache(tmp_path / f"k_{label}.sqlite")
    try:
        cache.set(text, "payload")
        assert cache.get(text) == "payload"
        assert _row_count(cache) == 1
    finally:
        cache.close()


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("null_byte_only", "\x00"),
        ("null_byte_middle", "before\x00after"),
        ("soh_stx", "\x01\x02\x03"),
        ("tab", "col1\tcol2\tcol3"),
        ("lf", "line1\nline2"),
        ("crlf", "line1\r\nline2"),
        ("mixed_control", "a\x00b\tc\nd\r\ne"),
    ],
)
def test_cache_control_characters_roundtrip(
    tmp_path: Path, label: str, payload: str
) -> None:
    """Control bytes including NULL must round-trip intact (SQLite TEXT allows them)."""
    cache = SynthResponseCache(tmp_path / f"{label}.sqlite")
    try:
        cache.set("k", payload)
        got = cache.get("k")
        assert got == payload
        assert got is not None
        assert len(got) == len(payload)
        assert got.encode("utf-8") == payload.encode("utf-8")
    finally:
        cache.close()


def test_cache_unicode_key_and_value_combined(tmp_path: Path) -> None:
    """A key and a value that both contain mixed scripts must round-trip together."""
    key = "鍵_fire_مفتاح"
    value = "返答_party_إجابة\nmultiline\x00with\x01controls"
    cache = SynthResponseCache(tmp_path / "mixed.sqlite")
    try:
        cache.set(key, value)
        assert cache.get(key) == value
    finally:
        cache.close()


# ===========================================================================
# SECTION 8 — Batch F: size edges
# ===========================================================================


def test_cache_empty_string_value_is_not_treated_as_miss(tmp_path: Path) -> None:
    """A stored empty-string value must be returned as '', not None."""
    cache = SynthResponseCache(tmp_path / "c.sqlite")
    try:
        cache.set("k", "")
        got = cache.get("k")
        assert got == ""
        assert got is not None
        assert _row_count(cache) == 1
    finally:
        cache.close()


def test_cache_empty_string_key(tmp_path: Path) -> None:
    """Empty string is a valid SQLite TEXT value for a PRIMARY KEY column."""
    cache = SynthResponseCache(tmp_path / "c.sqlite")
    try:
        cache.set("", "value-for-empty-key")
        assert cache.get("") == "value-for-empty-key"
        assert _row_count(cache) == 1
    finally:
        cache.close()


def test_cache_single_character_key_and_value(tmp_path: Path) -> None:
    """One-character keys and values must round-trip without special handling."""
    cache = SynthResponseCache(tmp_path / "c.sqlite")
    try:
        cache.set("a", "b")
        assert cache.get("a") == "b"
    finally:
        cache.close()


def test_cache_very_large_value_100kb(tmp_path: Path) -> None:
    """A 100KB value must round-trip without truncation."""
    cache = SynthResponseCache(tmp_path / "big.sqlite")
    try:
        payload = "x" * 100_000
        cache.set("big", payload)
        got = cache.get("big")
        assert got == payload
        assert got is not None
        assert len(got) == 100_000
    finally:
        cache.close()


def test_cache_very_large_key_several_kb(tmp_path: Path) -> None:
    """A multi-KB key must still round-trip and be hashable by SQLite."""
    cache = SynthResponseCache(tmp_path / "bigkey.sqlite")
    try:
        key = "k" * 8192
        cache.set(key, "v")
        assert cache.get(key) == "v"
    finally:
        cache.close()


def test_cache_multiline_value_preserves_structure(tmp_path: Path) -> None:
    """Multi-line values (JSON, source code) must preserve exact whitespace."""
    cache = SynthResponseCache(tmp_path / "c.sqlite")
    try:
        payload = (
            '{\n'
            '  "name": "test",\n'
            '  "body": "def f():\\n    return 1",\n'
            '  "list": [1, 2, 3]\n'
            '}'
        )
        cache.set("json", payload)
        assert cache.get("json") == payload
    finally:
        cache.close()


def test_cache_python_source_value_roundtrip(tmp_path: Path) -> None:
    """A block of Python source must round-trip identically (indentation intact)."""
    cache = SynthResponseCache(tmp_path / "c.sqlite")
    try:
        src = (
            "def greeter(name):\n"
            "    # a comment\n"
            "    message = f'Hello, {name}!'\n"
            "    return message\n"
            "\n"
            "print(greeter('world'))\n"
        )
        cache.set("src", src)
        assert cache.get("src") == src
    finally:
        cache.close()


# ===========================================================================
# SECTION 9 — Batch F: SQL-injection-shaped inputs (parameterised queries
# mean these are inert, but we prove it end-to-end)
# ===========================================================================


SQL_INJECTION_PAYLOADS = [
    "'; DROP TABLE cache; --",
    "'; DELETE FROM cache; --",
    "' OR '1'='1",
    "' OR 1=1 --",
    "'); DROP TABLE cache; --",
    "' UNION SELECT * FROM cache --",
    "%_;--/* */`\"'",  # SQLite meta-character soup
    "key\" AND \"1\"=\"1",
]


@pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
def test_cache_injection_payload_as_key_is_inert(
    tmp_path: Path, payload: str
) -> None:
    """SQL-shaped strings used as KEYS must be treated as literal data."""
    cache = SynthResponseCache(tmp_path / "inj.sqlite")
    try:
        # Pre-seed a canary row; any injection that DELETE/DROPs would erase it.
        cache.set("canary", "still-here")
        # Miss on the injection payload must return None (not raise).
        assert cache.get(payload) is None
        # And it must not have affected the canary.
        assert cache.get("canary") == "still-here"
        # Now store the injection string as a literal key + retrieve it.
        cache.set(payload, "stored")
        assert cache.get(payload) == "stored"
        # Table still contains both literal keys.
        assert _row_count(cache) == 2
    finally:
        cache.close()


@pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
def test_cache_injection_payload_as_value_is_inert(
    tmp_path: Path, payload: str
) -> None:
    """SQL-shaped strings used as VALUES must round-trip literally."""
    cache = SynthResponseCache(tmp_path / "inj_val.sqlite")
    try:
        cache.set("canary", "still-here")
        cache.set("k", payload)
        assert cache.get("k") == payload
        assert cache.get("canary") == "still-here"
        assert _row_count(cache) == 2
    finally:
        cache.close()


def test_cache_injection_cannot_drop_cache_table(tmp_path: Path) -> None:
    """After a best-effort injection attempt, the 'cache' table must still exist."""
    path = tmp_path / "inj_drop.sqlite"
    cache = SynthResponseCache(path)
    try:
        cache.set("canary", "v")
        # Throw every injection we have at the cache as a value, as a key, and
        # through both cache operations.
        for payload in SQL_INJECTION_PAYLOADS:
            cache.set(payload, payload)
            _ = cache.get(payload)
    finally:
        cache.close()

    # Re-open with a raw sqlite3 connection and verify the schema survived.
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='cache'"
        ).fetchone()
        assert row is not None, "injection destroyed the cache table"
        # And the canary data is still there.
        canary = conn.execute(
            "SELECT response FROM cache WHERE key=?", ("canary",)
        ).fetchone()
        assert canary is not None
        assert canary[0] == "v"
    finally:
        conn.close()


@pytest.mark.parametrize(
    "meta_char",
    ["%", "_", ";", "--", "/*", "*/", "'", '"', "`", "\\"],
)
def test_cache_meta_char_key_is_literal(tmp_path: Path, meta_char: str) -> None:
    """Each SQL meta character used as a key must be stored/retrieved literally."""
    cache = SynthResponseCache(tmp_path / "meta.sqlite")
    try:
        cache.set(meta_char, f"value-for-{meta_char}")
        assert cache.get(meta_char) == f"value-for-{meta_char}"
        # Meta chars should not pattern-match other keys (no LIKE-style behavior).
        cache.set("plain", "plain-value")
        assert cache.get(meta_char) == f"value-for-{meta_char}"
        assert cache.get("plain") == "plain-value"
        assert _row_count(cache) == 2
    finally:
        cache.close()


# ===========================================================================
# SECTION 10 — Batch F: persistence and reopening
# ===========================================================================


def test_cache_persistence_without_explicit_close(tmp_path: Path) -> None:
    """A set() must commit immediately; opening a second instance sees the row
    even if the first is not yet closed.
    """
    path = tmp_path / "noclose.sqlite"
    first = SynthResponseCache(path)
    try:
        first.set("k", "immediate")
        second = SynthResponseCache(path)
        try:
            assert second.get("k") == "immediate"
        finally:
            second.close()
    finally:
        first.close()


def test_cache_reopen_overwrites_with_new_value(tmp_path: Path) -> None:
    """Old value in a persisted file is cleanly replaced by a new instance."""
    path = tmp_path / "over.sqlite"
    first = SynthResponseCache(path)
    try:
        first.set("k", "first")
    finally:
        first.close()

    second = SynthResponseCache(path)
    try:
        assert second.get("k") == "first"
        second.set("k", "second")
    finally:
        second.close()

    third = SynthResponseCache(path)
    try:
        assert third.get("k") == "second"
        assert _row_count(third) == 1
    finally:
        third.close()


def test_cache_reopen_preserves_schema(tmp_path: Path) -> None:
    """Re-opening an existing DB must NOT clobber the 'cache' table."""
    path = tmp_path / "schema.sqlite"
    first = SynthResponseCache(path)
    try:
        first.set("alpha", "1")
        first.set("beta", "2")
    finally:
        first.close()

    second = SynthResponseCache(path)
    try:
        assert second.get("alpha") == "1"
        assert second.get("beta") == "2"
        assert _row_count(second) == 2
    finally:
        second.close()


def test_cache_mixed_open_close_cycles_survive(tmp_path: Path) -> None:
    """100 writes spread across 3 open/close cycles all survive to the end."""
    path = tmp_path / "cycles.sqlite"

    # Cycle 1: write entries 0-33.
    cache = SynthResponseCache(path)
    try:
        for i in range(34):
            cache.set(f"k{i}", f"v{i}")
    finally:
        cache.close()

    # Cycle 2: write entries 34-66.
    cache = SynthResponseCache(path)
    try:
        for i in range(34, 67):
            cache.set(f"k{i}", f"v{i}")
    finally:
        cache.close()

    # Cycle 3: write entries 67-99.
    cache = SynthResponseCache(path)
    try:
        for i in range(67, 100):
            cache.set(f"k{i}", f"v{i}")
    finally:
        cache.close()

    # Final check.
    cache = SynthResponseCache(path)
    try:
        assert _row_count(cache) == 100
        for i in range(100):
            assert cache.get(f"k{i}") == f"v{i}"
    finally:
        cache.close()


# ===========================================================================
# SECTION 11 — Batch F: post-close behaviour
# ===========================================================================


def test_cache_get_after_close_raises(tmp_path: Path) -> None:
    """get() after close() must raise sqlite3.ProgrammingError."""
    cache = SynthResponseCache(tmp_path / "c.sqlite")
    cache.close()
    with pytest.raises(sqlite3.ProgrammingError):
        cache.get("k")


def test_cache_set_after_close_raises(tmp_path: Path) -> None:
    """set() after close() must raise sqlite3.ProgrammingError."""
    cache = SynthResponseCache(tmp_path / "c.sqlite")
    cache.close()
    with pytest.raises(sqlite3.ProgrammingError):
        cache.set("k", "v")


def test_cache_double_close_is_idempotent(tmp_path: Path) -> None:
    """Calling close() twice must NOT raise — sqlite3.Connection.close is idempotent."""
    cache = SynthResponseCache(tmp_path / "c.sqlite")
    cache.close()
    cache.close()  # second close is a no-op


def test_cache_context_manager_then_explicit_close_is_idempotent(
    tmp_path: Path,
) -> None:
    """After ``with`` exits, calling close() again must still be a no-op."""
    path = tmp_path / "c.sqlite"
    with SynthResponseCache(path) as cache:
        cache.set("k", "v")
    # __exit__ already called close(); a subsequent close() must not raise.
    cache.close()


# ===========================================================================
# SECTION 12 — Batch F: constructor and path edge cases
# ===========================================================================


def test_cache_deeply_nested_parent_dir_is_created(tmp_path: Path) -> None:
    """mkdir(parents=True) must construct a 4+-level parent chain."""
    path = tmp_path / "a" / "b" / "c" / "d" / "cache.sqlite"
    assert not (tmp_path / "a").exists()
    cache = SynthResponseCache(path)
    try:
        assert path.parent.is_dir()
        assert path.exists()
    finally:
        cache.close()


def test_cache_parent_dir_already_exists_is_noop(tmp_path: Path) -> None:
    """exist_ok=True: if the parent already exists, construction must not raise."""
    parent = tmp_path / "existing"
    parent.mkdir()
    assert parent.is_dir()
    cache = SynthResponseCache(parent / "c.sqlite")
    try:
        cache.set("k", "v")
        assert cache.get("k") == "v"
    finally:
        cache.close()


def test_cache_construct_over_existing_file_preserves_data(tmp_path: Path) -> None:
    """Re-construction on an already-initialised DB file preserves prior rows."""
    path = tmp_path / "existing.sqlite"
    first = SynthResponseCache(path)
    try:
        first.set("k", "v")
    finally:
        first.close()

    assert path.exists()
    second = SynthResponseCache(path)
    try:
        assert second.get("k") == "v"
    finally:
        second.close()


# ===========================================================================
# SECTION 13 — Batch F: error-path tests
# ===========================================================================


def test_cache_construct_on_corrupted_db_raises_database_error(
    tmp_path: Path,
) -> None:
    """Opening a non-SQLite file with garbage content must raise sqlite3.DatabaseError.

    Note: sqlite3 is lenient about empty / very-short bytes (they look like a
    zero-page DB), so we write a chunk of garbage bigger than the SQLite
    header and then trigger the header check by running a query.
    """
    path = tmp_path / "corrupt.sqlite"
    path.write_bytes(b"this is not a valid sqlite db file " * 64)
    with pytest.raises(sqlite3.DatabaseError):
        SynthResponseCache(path)


def test_cache_construct_on_readonly_parent_dir_raises(tmp_path: Path) -> None:
    """A read-only parent directory must cause construction to raise OSError.

    Linux-only assumption: os.chmod(0o555) on a directory disables writes
    for the owner (effective uid will not be root in CI).
    """
    parent = tmp_path / "readonly"
    parent.mkdir()
    os.chmod(parent, 0o555)
    try:
        with pytest.raises((OSError, sqlite3.OperationalError)):
            SynthResponseCache(parent / "c.sqlite")
    finally:
        # Restore so pytest can clean up tmp_path.
        os.chmod(parent, 0o755)


def test_cache_set_on_readonly_file_raises(tmp_path: Path) -> None:
    """SQLite must raise OperationalError when the DB file is read-only."""
    path = tmp_path / "ro.sqlite"
    cache = SynthResponseCache(path)
    try:
        cache.set("k", "v")
    finally:
        cache.close()

    os.chmod(path, 0o444)
    try:
        reopened = SynthResponseCache(path)
        try:
            # Reads should still work.
            assert reopened.get("k") == "v"
            with pytest.raises(sqlite3.OperationalError):
                reopened.set("k2", "v2")
        finally:
            reopened.close()
    finally:
        os.chmod(path, 0o644)


# ===========================================================================
# SECTION 14 — Batch F: concurrency sanity (single-process, two connections)
# ===========================================================================


def test_two_cache_instances_same_path_see_each_others_writes(
    tmp_path: Path,
) -> None:
    """Two SynthResponseCache instances on one path must interleave reads/writes.

    This is the 'immediately-committed' contract: every set() calls commit(),
    so a sibling connection should observe the write on its next query.
    """
    path = tmp_path / "shared.sqlite"
    a = SynthResponseCache(path)
    b = SynthResponseCache(path)
    try:
        # A writes, B reads.
        a.set("from_a", "first")
        assert b.get("from_a") == "first"

        # B writes, A reads.
        b.set("from_b", "second")
        assert a.get("from_b") == "second"

        # Final state: union of both keys visible from both instances.
        assert a.get("from_a") == "first"
        assert a.get("from_b") == "second"
        assert b.get("from_a") == "first"
        assert b.get("from_b") == "second"
        assert _row_count(a) == 2
        assert _row_count(b) == 2
    finally:
        a.close()
        b.close()


def test_two_cache_instances_see_overwrite(tmp_path: Path) -> None:
    """An overwrite through one connection must be visible from the other."""
    path = tmp_path / "over.sqlite"
    a = SynthResponseCache(path)
    b = SynthResponseCache(path)
    try:
        a.set("k", "v1")
        assert b.get("k") == "v1"
        b.set("k", "v2")
        assert a.get("k") == "v2"
        assert _row_count(a) == 1
        assert _row_count(b) == 1
    finally:
        a.close()
        b.close()


# ===========================================================================
# SECTION 15 — Batch F: stress / performance sanity
# ===========================================================================


def test_cache_insert_and_readback_many_entries(tmp_path: Path) -> None:
    """1k set/get round-trips must succeed, with a loose wall-clock ceiling.

    The cache commits every ``set()`` synchronously (that is its durability
    contract), so each write pays an fsync. On the repo's CI host 1k commits
    take O(a few seconds) — we set the ceiling to 60 s to be robust against
    slow disks, since this is a pathological-regression sanity test, not a
    benchmark. A regression that made every set() rewrite the whole table
    would blow past minutes.
    """
    n = 1_000
    cache = SynthResponseCache(tmp_path / "stress.sqlite")
    try:
        start = time.perf_counter()
        for i in range(n):
            cache.set(f"k{i}", f"v{i}")
        write_dur = time.perf_counter() - start

        start = time.perf_counter()
        for i in range(n):
            assert cache.get(f"k{i}") == f"v{i}"
        read_dur = time.perf_counter() - start

        assert _row_count(cache) == n
        assert write_dur < 60.0, f"write took {write_dur:.2f}s"
        assert read_dur < 10.0, f"read took {read_dur:.2f}s"
    finally:
        cache.close()


def test_cache_2k_mixed_gets(tmp_path: Path) -> None:
    """After 2k inserts, present and absent key lookups remain consistent."""
    cache = SynthResponseCache(tmp_path / "mix.sqlite")
    try:
        n = 2_000
        for i in range(n):
            cache.set(f"k{i}", f"v{i}")
        assert _row_count(cache) == n
        # Spot-check every 7th key for its correct value.
        for i in range(0, n, 7):
            assert cache.get(f"k{i}") == f"v{i}"
        # Negative spot-check: keys never inserted are absent.
        for i in range(n, n + 50):
            assert cache.get(f"k{i}") is None
    finally:
        cache.close()


# ===========================================================================
# SECTION 16 — Batch F: make_cache_key edges
# ===========================================================================


def test_make_cache_key_all_empty_inputs_still_returns_hex() -> None:
    """Empty model/prompt/tag still produces a 64-char sha256 hex string."""
    k = make_cache_key(model="", prompt="", tag="")
    assert len(k) == 64
    int(k, 16)
    # It must equal sha256 of the literal "\n---\n\n---\n" payload.
    expected = hashlib.sha256(b"\n---\n\n---\n").hexdigest()
    assert k == expected


def test_make_cache_key_megabyte_prompt_still_returns_hex() -> None:
    """A 1 MB prompt must still hash cleanly to a 64-char hex string."""
    prompt = "x" * 1_000_000
    k = make_cache_key(model="m", prompt=prompt, tag="t")
    assert len(k) == 64
    int(k, 16)


def test_make_cache_key_unicode_in_all_fields_is_hex() -> None:
    """Unicode in all three fields must not blow up UTF-8 encoding."""
    k = make_cache_key(
        model="モデル",
        prompt="こんにちは fire",
        tag="試行",
    )
    assert len(k) == 64
    int(k, 16)


def test_make_cache_key_field_permutation_differs() -> None:
    """Swapping which field holds which string must produce distinct hashes."""
    k1 = make_cache_key(model="A", prompt="B", tag="C")
    k2 = make_cache_key(model="C", prompt="B", tag="A")
    k3 = make_cache_key(model="B", prompt="A", tag="C")
    # All three must be distinct.
    assert k1 != k2
    assert k1 != k3
    assert k2 != k3


def test_make_cache_key_separator_aware_disambiguation() -> None:
    """A naive construction of model/prompt/tag that embeds the separator in
    the model field still disambiguates correctly because of the *additional*
    separator in the payload.

    Example: model="a\\n---\\nb" with empty prompt/tag yields
    "a\\n---\\nb\\n---\\n\\n---\\n", which is DIFFERENT from
    model="a", prompt="b", tag="" (which yields "a\\n---\\nb\\n---\\n").
    """
    k_split = make_cache_key(model="a", prompt="b", tag="")
    k_joined = make_cache_key(model="a\n---\nb", prompt="", tag="")
    assert k_split != k_joined


def test_make_cache_key_known_collision_is_documented() -> None:
    """KNOWN LIMITATION: the separator ``\\n---\\n`` is not escape-safe.

    Two different (model, prompt, tag) triples whose concatenation with the
    separator yields the same byte string will hash to the same key. This
    test pins that behaviour so any future refactor of ``make_cache_key`` to
    use a safer construction (e.g. length-prefixed or JSON-encoded payload)
    is a visible breaking change and will fail loudly.

    Concretely:
        model="x", prompt="y\\n---\\nz", tag=""
        model="x", prompt="y", tag="z\\n---\\n"
    both produce the payload "x\\n---\\ny\\n---\\nz\\n---\\n".
    """
    k1 = make_cache_key(model="x", prompt="y\n---\nz", tag="")
    k2 = make_cache_key(model="x", prompt="y", tag="z\n---\n")
    assert k1 == k2  # documented collision


def test_make_cache_key_control_chars_roundtrip_to_hex() -> None:
    """Control characters in any field must not break the hash function."""
    k = make_cache_key(
        model="m\x00m",
        prompt="p\x01\x02p",
        tag="t\tt\nt",
    )
    assert len(k) == 64
    int(k, 16)


def test_make_cache_key_is_usable_as_cache_key(tmp_path: Path) -> None:
    """The full pipeline: make_cache_key() → SynthResponseCache.get/set."""
    cache = SynthResponseCache(tmp_path / "pipe.sqlite")
    try:
        key = make_cache_key(model="gpt-4o", prompt="hello", tag="attempt-0")
        cache.set(key, "response body")
        assert cache.get(key) == "response body"
        # Same identity recomputes the same key.
        key2 = make_cache_key(model="gpt-4o", prompt="hello", tag="attempt-0")
        assert cache.get(key2) == "response body"
    finally:
        cache.close()


# ===========================================================================
# SECTION 17 — Batch F: INSERT OR REPLACE + commit-on-write semantics
# ===========================================================================


def test_cache_ten_way_overwrite_chain_last_value_wins(tmp_path: Path) -> None:
    """Writing 10 different values to the same key leaves only the last one."""
    cache = SynthResponseCache(tmp_path / "c.sqlite")
    try:
        for i in range(10):
            cache.set("k", f"v{i}")
        assert _row_count(cache) == 1
        assert cache.get("k") == "v9"
    finally:
        cache.close()


def test_cache_multiple_keys_only_matching_key_is_overwritten(
    tmp_path: Path,
) -> None:
    """Overwriting one key must not disturb other keys in the table."""
    cache = SynthResponseCache(tmp_path / "c.sqlite")
    try:
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        cache.set("b", "2-new")  # overwrite only b
        assert cache.get("a") == "1"
        assert cache.get("b") == "2-new"
        assert cache.get("c") == "3"
        assert _row_count(cache) == 3
    finally:
        cache.close()


def test_cache_commit_is_immediate_without_close(tmp_path: Path) -> None:
    """set() must COMMIT without needing close(); a second connection sees it."""
    path = tmp_path / "commit.sqlite"
    writer = SynthResponseCache(path)
    try:
        writer.set("k", "committed")
        # Use a raw sqlite3 connection to bypass any code path inside the cache.
        raw = sqlite3.connect(str(path))
        try:
            row = raw.execute(
                "SELECT response FROM cache WHERE key=?", ("k",)
            ).fetchone()
            assert row is not None
            assert row[0] == "committed"
        finally:
            raw.close()
    finally:
        writer.close()


def test_cache_commit_is_immediate_after_overwrite(tmp_path: Path) -> None:
    """A second set() must also be immediately committed and visible externally."""
    path = tmp_path / "commit2.sqlite"
    writer = SynthResponseCache(path)
    try:
        writer.set("k", "v1")
        writer.set("k", "v2")
        raw = sqlite3.connect(str(path))
        try:
            row = raw.execute(
                "SELECT response FROM cache WHERE key=?", ("k",)
            ).fetchone()
            assert row is not None
            assert row[0] == "v2"
        finally:
            raw.close()
    finally:
        writer.close()
