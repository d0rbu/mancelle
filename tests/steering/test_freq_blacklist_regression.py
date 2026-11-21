"""Regression: our wiring never perturbs PR 471's I1/I5 + golden + audit.

Mirrors PR 471's frozen-golden invariant EXACTLY (tests/steering/
test_selection_hash_feature_mask.py): same representative payload + positional
signature compute_selection_hash(pos, neg, delta, num_buckets, seed,
ddp_rank, feature_mask=...). PR 471 pins the golden RELATIVELY (no hardcoded
hex): hash-with-feature_mask=None == hash-without. Our added invariant: our
no-op wiring must leave cfg.feature_mask None so the resolved-canonical hash
also equals that golden.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import torch as t
from omegaconf import OmegaConf

from steering.conditional import freq_blacklist as fb
from steering.conditional.feature_mask import canonical_dict, resolve_feature_mask
from steering.packed_batch_io import compute_selection_hash
from tests.steering._freq_blacklist_fixtures import write_stats

POS = {"threshold_type": "segment_topk", "threshold": 2048}


def _legacy() -> str:
    return compute_selection_hash(POS, None, None, 10, 42, 0)


def test_pr471_relative_golden_holds_at_head():
    golden = _legacy()
    assert (
        compute_selection_hash(POS, None, None, 10, 42, 0, feature_mask=None)
        == golden
    )
    assert compute_selection_hash(POS, None, None, 10, 42, 0) == golden


def test_disabled_wiring_byte_identical_I1(tmp_path):
    cfg = OmegaConf.create(
        {"feature_mask_freq_min": None, "feature_mask_freq_max": None, "feature_mask": None}
    )
    fb.maybe_apply_frequency_blacklist(cfg, str(tmp_path), 16384)
    h = compute_selection_hash(
        POS, None, None, 10, 42, 0,
        feature_mask=canonical_dict(resolve_feature_mask(cfg, 16384)),
    )
    assert h == _legacy()


def test_disabled_wiring_literal_cfg_identity(tmp_path):
    before = {"feature_mask_freq_min": None, "feature_mask_freq_max": None, "feature_mask": None}
    cfg = OmegaConf.create(before)
    fb.maybe_apply_frequency_blacklist(cfg, str(tmp_path), 16384)
    assert OmegaConf.to_container(cfg, resolve=True) == before


def test_empty_noop_byte_identical_I1(tmp_path):
    write_stats(
        tmp_path, n_segments=100, counts={1: 1, 2: 2}, n_sae_features=16384
    )
    cfg = OmegaConf.create(
        {"feature_mask_freq_min": None, "feature_mask_freq_max": 0.9, "feature_mask": None}
    )
    fb.maybe_apply_frequency_blacklist(cfg, str(tmp_path), 16384)
    h = compute_selection_hash(
        POS, None, None, 10, 42, 0,
        feature_mask=canonical_dict(resolve_feature_mask(cfg, 16384)),
    )
    assert h == _legacy()


def test_user_mask_literal_cfg_identity(tmp_path):
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16384)
    before = {
        "feature_mask_freq_min": None, "feature_mask_freq_max": 0.4,
        "feature_mask": {"mode": "blacklist", "ids": [99]},
    }
    cfg = OmegaConf.create(before)
    fb.maybe_apply_frequency_blacklist(cfg, str(tmp_path), 16384)
    assert OmegaConf.to_container(cfg, resolve=True) == before


def test_applied_changes_hash(tmp_path):
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16384)
    cfg = OmegaConf.create(
        {"feature_mask_freq_min": None, "feature_mask_freq_max": 0.4, "feature_mask": None}
    )
    fb.maybe_apply_frequency_blacklist(cfg, str(tmp_path), 16384)
    h = compute_selection_hash(
        POS, None, None, 10, 42, 0,
        feature_mask=canonical_dict(resolve_feature_mask(cfg, 16384)),
    )
    assert h != _legacy()


def test_disabled_wiring_rng_state_bit_identical(tmp_path):
    """Mirror PR 471's I5 pattern: a no-op call must not consume RNG."""
    cfg = OmegaConf.create(
        {"feature_mask_freq_min": None, "feature_mask_freq_max": None, "feature_mask": None}
    )
    t.manual_seed(7)
    pre = t.get_rng_state()
    fb.maybe_apply_frequency_blacklist(cfg, str(tmp_path), 16384)
    assert t.equal(t.get_rng_state(), pre)


def test_blacklist_txt_full_byte_golden(tmp_path):
    write_stats(tmp_path, n_segments=10, counts={1: 8, 2: 1}, n_sae_features=16)
    out = fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4)
    assert out is not None
    txt = out[0].read_text()
    expected = (
        "# activation-frequency blacklist (generator v1, source: "
        "feature_statistics.safetensors)\n"
        "# freq_min: null  (blacklist iff count <= freq_min * n_segments)\n"
        "# freq_max: 0.4  (blacklist iff count >  freq_max * n_segments)\n"
        "# n_sae_features: 16\n"
        "# n_segments: 10\n"
        "# n_active_features: 2\n"
        "# stats_generated_at: 2026-05-19T00:00:00+00:00\n"
        "# n_blacklisted: 1\n"
        "1\n"
    )
    assert txt == expected
    assert (
        hashlib.sha256(txt.encode()).hexdigest()
        == hashlib.sha256(expected.encode()).hexdigest()
    )


def test_meta_json_schema(tmp_path):
    write_stats(tmp_path, n_segments=10, counts={1: 8, 2: 1}, n_sae_features=16)
    fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4)
    _txt, meta_path = fb.blacklist_paths(tmp_path, freq_min=None, freq_max=0.4)
    meta = json.loads(meta_path.read_text())
    assert meta["generator_version"] == fb.GENERATOR_VERSION == 1
    assert meta["source"] == "feature_statistics.safetensors"
    assert meta["freq_min"] == "null"
    assert meta["freq_max"] == repr(0.4)
    assert meta["n_sae_features"] == 16
    assert meta["n_segments"] == 10
    assert meta["n_active_features"] == 2
    assert meta["stats_generated_at"] == "2026-05-19T00:00:00+00:00"
    assert meta["n_blacklisted"] == 1
    assert meta["empty"] is False
    assert "generated_at" in meta  # volatile, not byte-pinned


def test_version_bump_rewrites(tmp_path, monkeypatch):
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16)
    fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4)
    txt, _meta = fb.blacklist_paths(tmp_path, freq_min=None, freq_max=0.4)
    first = txt.read_text()
    monkeypatch.setattr(fb, "GENERATOR_VERSION", fb.GENERATOR_VERSION + 1)
    # header embeds the version -> content differs -> rewrite
    fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4)
    assert txt.read_text() != first
    assert "generator v2" in txt.read_text()  # GENERATOR_VERSION (1) + 1


def test_default_change_audit():
    # (a) no keys ⇒ current defaults (min=1e-5, max=0.5) in effect (#702
    # raised freq_min from 0.0; this assertion exists specifically to catch
    # the next silent default drift the same way it caught this one)
    assert fb.resolve_blacklist_bounds(OmegaConf.create({})) == fb.FrequencyBounds(
        min=1e-5, max=0.5
    )
    # (b) both bounds explicit null ⇒ fully disabled
    assert (
        fb.resolve_blacklist_bounds(
            OmegaConf.create(
                {"feature_mask_freq_min": None, "feature_mask_freq_max": None}
            )
        )
        is None
    )
    # (c) Non-tautological audit: no tests/ file may BOTH obtain a selection hash AND drive
    # a training script end-to-end. A change to the default band changes the resolved id set,
    # which changes compute_selection_hash AND the bl_/wl_ checkpoint leaf — so such a test
    # would go silently stale.
    #
    # Matched on CALLS (AST), not text. The grep version false-positived on PROSE: a docstring
    # merely EXPLAINING why the mask mode affects `compute_selection_hash`, and a NOTE
    # mentioning `conditional_encdec_sft.main(`, made an innocent file an "offender". A text
    # scan on a rule this important is worse than useless — every false positive is an
    # invitation to weaken it.
    #
    # But the FIRST AST rewrite was strictly WEAKER than the grep, in three ways it took a
    # second review to catch, and all three are fixed here:
    #   * it silently DROPPED both `main()` entrypoints from the driver set;
    #   * it deleted the grep's non-vacuity canary, so a broken scan would pass silently;
    #   * it scanned a CWD-relative `Path("tests")`, so it passed VACUOUSLY from any other
    #     working directory.
    # A weaker replacement for a guard is worse than no replacement, because it still reads
    # like one. `TestTheAuditHasTeeth` below now proves each property instead of asserting it.
    suspects, offenders = _audit_tests_tree()

    assert suspects, (
        "the audit found NO test calling compute_selection_hash. That is not possible in "
        "this repo — the scan is broken (wrong root? renamed function?) and every assertion "
        "below it is vacuous. This canary existed in the grep version and was dropped in the "
        "first AST rewrite."
    )
    assert not offenders, (
        f"Full-training-script golden tests found: {sorted(offenders)}. Each must pin the "
        "band explicitly (feature_mask_freq_min / _max) or set BOTH to null — otherwise a "
        "default-band change silently invalidates the hash it asserts."
    )


# The functions whose RESULT depends on the resolved blacklist id set...
HASHERS = frozenset({"compute_selection_hash"})
# ...and the entrypoints that drive a real training run end-to-end. `main` is qualified,
# because a bare `main(` is far too common to match on.
DRIVERS = frozenset(
    {
        "build_shared_state",
        "train_one_config",
        "conditional_encdec_train.main",
        "conditional_encdec_sft.main",
    }
)
# Anchored to the REPO, not the CWD. `Path("tests")` made the whole audit pass vacuously from
# any directory that has no tests/ subtree — which is to say, it was one `cd` from useless.
TESTS_ROOT = Path(__file__).resolve().parents[2] / "tests"


def _alias_map(tree: ast.AST) -> dict[str, str]:
    """Local name -> the FULL dotted path it was imported from.

    ``ImportFrom`` must be qualified with ``node.module``. An earlier version wrote
    ``out[a.asname or a.name] = a.name``, dropping the module — so the MOST IDIOMATIC driver
    import there is::

        from scripts.train.conditional_encdec_sft import main
        main([])

    resolved to the bare name ``"main"``, which matches no driver, and the offender sailed
    straight through. That is the third hole found in this matcher; each one made it weaker
    than the grep it replaced. Hence ``TestTheAuditHasTeeth``, which drives the REAL import
    forms rather than ones convenient to the implementation.
    """
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                # Only an `as` binding needs substituting. A plain `import a.b.c` binds `a`,
                # and any call through it (`a.b.c.main()`) is ALREADY fully dotted — mapping
                # `a -> a.b.c` here would splice the tail on twice.
                if a.asname:
                    out[a.asname] = a.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for a in node.names:
                out[a.asname or a.name] = f"{module}.{a.name}" if module else a.name
    return out


def _imported_roots(tree: ast.AST) -> set[str]:
    """Root names bound by a plain ``import a.b.c`` — i.e. ``{"a"}``.

    A call whose head is one of these is a MODULE path and already fully qualified. Any other
    dotted head (``self.``, ``obj.``, a fixture) is not, and must not be resolved as if it
    were: that is what turns an innocent ``self.train_one_config()`` into an offender.
    """
    return {
        a.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for a in node.names
        if not a.asname
    }


def _dotted_call_target(fn: ast.expr) -> str | None:
    """The full dotted path of a call target, or ``None`` if it is not a static name chain.

    Walks the WHOLE ``Attribute`` chain. The previous version only resolved
    ``Attribute(value=Name)``, one level deep — so ``scripts.train.conditional_encdec_sft.main()``
    (whose ``func.value`` is itself an ``Attribute``) collapsed to the bare name ``"main"``,
    which ``DRIVERS`` deliberately excludes. The offender walked straight through. That was the
    FOURTH hole in this matcher, and every one of them made it weaker than the grep it replaced.
    """
    parts: list[str] = []
    node: ast.expr = fn
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None  # f()(), d["k"].main(), ... — no static name to resolve
    parts.append(node.id)
    return ".".join(reversed(parts))


def _name_forms(dotted: str) -> set[str]:
    """Every dotted SUFFIX of a name, so DRIVERS/HASHERS can be written at any qualification.

    `scripts.train.conditional_encdec_sft.main` yields {full, `conditional_encdec_sft.main`,
    `main`}. Without the middle form, `DRIVERS = {"conditional_encdec_sft.main"}` matched
    NOTHING — the fix for the last hole introduced the next one.
    """
    parts = dotted.split(".")
    return {".".join(parts[i:]) for i in range(len(parts))}


def _called_names(tree: ast.AST) -> set[str]:
    """Every function actually CALLED, resolved to a fully-qualified dotted name.

    A name is only emitted when it can be traced to an IMPORT — either an alias
    (``from x import f`` / ``import x.y as z``) or a plain ``import a.b.c`` whose root heads
    the call. A bare name that was never imported is a LOCAL definition, and a dotted head
    that was never imported is an attribute of some object; neither is the driver ``DRIVERS``
    means, and emitting them anyway is what let a file with its own ``def train_one_config``
    be reported as an offender for calling it.

    Both directions matter, and this matcher has been wrong in both: it MISSED real offenders
    (four times) and, once the misses were patched, FLAGGED innocent ones. ``TestTheAuditHasTeeth``
    drives every real import/call form rather than the ones convenient to this implementation.
    """
    aliases = _alias_map(tree)
    roots = _imported_roots(tree)
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted_call_target(node.func)
        if dotted is None:
            continue
        head, _, rest = dotted.partition(".")
        if head in aliases:
            dotted = f"{aliases[head]}.{rest}" if rest else aliases[head]
        elif head not in roots:
            continue  # a local def, or a method on some object — not an imported driver
        out.update(_name_forms(dotted))
    return out


def _audit_tests_tree(root: Path | None = None) -> tuple[set[str], set[str]]:
    """(files that obtain a selection hash, of those the ones that also drive training)."""
    root = TESTS_ROOT if root is None else root
    suspects: set[str] = set()
    offenders: set[str] = set()
    for path in root.rglob("*.py"):
        if "freq_blacklist" in path.name:
            continue
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except SyntaxError:
            continue
        called = _called_names(tree)
        if not (called & HASHERS):
            continue
        rel = str(path.relative_to(root.parent))
        suspects.add(rel)
        if called & DRIVERS:
            offenders.add(rel)
    return suspects, offenders


class TestTheAuditHasTeeth:
    """The audit above passes. These prove that is because nothing offends — not because the
    matcher is broken. Every one of them corresponds to a way the first AST rewrite was
    silently weaker than the grep it replaced."""

    def _plant(self, root: Path, name: str, body: str) -> None:
        (root / name).write_text(body)

    def test_it_catches_a_direct_call(self, tmp_path):
        root = tmp_path / "tests"
        root.mkdir()
        self._plant(root, "test_x.py",
            "from steering.packed_batch_io import compute_selection_hash\n"
            "from scripts.train.conditional_encdec_train import train_one_config\n"
            "def test_a():\n"
            "    train_one_config(cfg)\n"
            "    assert compute_selection_hash(a) == 'x'\n")
        suspects, offenders = _audit_tests_tree(root)
        assert suspects and offenders

    def test_it_catches_an_ALIASED_import(self, tmp_path):
        """`import ... as sft` then `sft.main(...)` — invisible to the first AST version."""
        root = tmp_path / "tests"
        root.mkdir()
        self._plant(root, "test_y.py",
            "import scripts.train.conditional_encdec_sft as sft\n"
            "from steering.packed_batch_io import compute_selection_hash as csh\n"
            "def test_b():\n"
            "    sft.main([])\n"
            "    assert csh(a) == 'x'\n")
        suspects, offenders = _audit_tests_tree(root)
        assert suspects, "an aliased hasher must still make the file a suspect"
        assert offenders, "an aliased main() driver must still make it an offender"

    def test_it_catches_a_FULLY_DOTTED_call_after_a_plain_import(self, tmp_path):
        """The FOURTH hole: `import a.b.c` + `a.b.c.main([])`.

        The call's `func.value` is itself an `Attribute`, not a `Name`, so the one-level-deep
        resolver collapsed the whole chain to the bare name "main" — which DRIVERS excludes on
        purpose ("a bare `main(` is far too common"). The offender walked straight through, and
        the plain grep this replaced had matched it as a substring.
        """
        root = tmp_path / "tests"
        root.mkdir()
        self._plant(root, "test_z.py",
            "import scripts.train.conditional_encdec_sft\n"
            "from steering.packed_batch_io import compute_selection_hash\n"
            "def test_c():\n"
            "    scripts.train.conditional_encdec_sft.main([])\n"
            "    assert compute_selection_hash(a) == 'x'\n")
        suspects, offenders = _audit_tests_tree(root)
        assert suspects, "the hasher must make the file a suspect"
        assert offenders, (
            "a fully-dotted `scripts.train.conditional_encdec_sft.main([])` is a driver call "
            "and must be caught — the whole Attribute chain has to be walked, not one level"
        )

    def test_it_does_NOT_flag_a_LOCAL_function_that_shares_a_driver_name(self, tmp_path):
        """The other direction, and just as fatal: crying wolf.

        `_called_names` used to emit every bare called name, and DRIVERS holds the UNQUALIFIED
        `train_one_config` / `build_shared_state`. So a file defining its own helper by that
        name — and never touching a training script — was reported as a full-training-script
        golden test. A guard that fails on innocent files gets deleted, and then it guards
        nothing.
        """
        root = tmp_path / "tests"
        root.mkdir()
        self._plant(root, "test_local.py",
            "from steering.packed_batch_io import compute_selection_hash\n"
            "def train_one_config(cfg):\n"
            "    return cfg\n"
            "def test_d():\n"
            "    train_one_config({})\n"
            "    assert compute_selection_hash(a) == 'x'\n")
        suspects, offenders = _audit_tests_tree(root)
        assert suspects, "it does obtain a selection hash, so it is a suspect"
        assert not offenders, (
            "a LOCAL def train_one_config is not the training driver — flagging it makes the "
            f"audit cry wolf. offenders={offenders}"
        )

    def test_it_does_NOT_flag_a_METHOD_that_shares_a_driver_name(self, tmp_path):
        """Same hole, reached through an attribute rather than a bare name."""
        root = tmp_path / "tests"
        root.mkdir()
        self._plant(root, "test_method.py",
            "from steering.packed_batch_io import compute_selection_hash\n"
            "class Harness:\n"
            "    def train_one_config(self, cfg):\n"
            "        return cfg\n"
            "def test_e(h):\n"
            "    h.train_one_config({})\n"
            "    assert compute_selection_hash(a) == 'x'\n")
        suspects, offenders = _audit_tests_tree(root)
        assert suspects
        assert not offenders, f"h.train_one_config() is a method call, not a driver: {offenders}"

    def test_it_catches_the_main_ENTRYPOINTS_the_first_rewrite_dropped(self, tmp_path):
        root = tmp_path / "tests"
        root.mkdir()
        self._plant(root, "test_z.py",
            "from steering.packed_batch_io import compute_selection_hash\n"
            "from scripts.train import conditional_encdec_train\n"
            "def test_c():\n"
            "    conditional_encdec_train.main([])\n"
            "    assert compute_selection_hash(a) == 'x'\n")
        _, offenders = _audit_tests_tree(root)
        assert offenders, "conditional_encdec_train.main() is a driver and was dropped"

    def test_it_catches_the_IDIOMATIC_from_import_of_main(self, tmp_path):
        """`from scripts.train.conditional_encdec_sft import main` + `main([])`.

        The single most natural way to write the offender, and it EVADED the matcher until a
        third review caught it: the alias map dropped `node.module`, so `main` resolved to
        `main` and matched no driver.
        """
        root = tmp_path / "tests"
        root.mkdir()
        self._plant(root, "test_i.py",
            "from scripts.train.conditional_encdec_sft import main\n"
            "from steering.packed_batch_io import compute_selection_hash\n"
            "def test_e():\n"
            "    main([])\n"
            "    assert compute_selection_hash(a) == 'x'\n")
        suspects, offenders = _audit_tests_tree(root)
        assert suspects and offenders, "the idiomatic `from ... import main` form must be caught"

    def test_a_call_inside_a_LOCAL_HELPER_is_still_caught(self, tmp_path):
        """`ast.walk` sees every Call in the file, whichever function encloses it — so routing
        the driver through a local helper does NOT evade the matcher.

        Written first as a "known limit" test asserting the opposite. It failed, which is how I
        learned the limit I was about to document did not exist. Pinning a limitation you have
        not verified is how a guard acquires imaginary holes.
        """
        root = tmp_path / "tests"
        root.mkdir()
        self._plant(root, "test_h.py",
            "from scripts.train.conditional_encdec_train import train_one_config\n"
            "from steering.packed_batch_io import compute_selection_hash\n"
            "def _drive(cfg):\n"
            "    train_one_config(cfg)\n"
            "def test_f():\n"
            "    _drive(cfg)\n"
            "    assert compute_selection_hash(a) == 'x'\n")
        _, offenders = _audit_tests_tree(root)
        assert offenders, "a driver call inside a local helper is still a direct Call node"

    def test_the_REAL_known_limit_dynamic_dispatch(self, tmp_path):
        """What the matcher genuinely cannot see: a driver reached by NAME at runtime
        (`importlib`, `getattr`). Static call-graph matching one level deep is the contract.

        Pinned so the limit is a known property, not a surprise. Nobody should mistake this
        audit for exhaustive — but it is now strictly stronger than the grep it replaced,
        which this file's other five teeth tests prove.
        """
        root = tmp_path / "tests"
        root.mkdir()
        self._plant(root, "test_dyn.py",
            "import importlib\n"
            "from steering.packed_batch_io import compute_selection_hash\n"
            "def test_g():\n"
            "    getattr(importlib.import_module('scripts.train.conditional_encdec_train'), "
            "'train_one_config')(cfg)\n"
            "    assert compute_selection_hash(a) == 'x'\n")
        _, offenders = _audit_tests_tree(root)
        assert not offenders, (
            "KNOWN LIMIT (stated, not hidden): a driver invoked via importlib/getattr is not "
            "a static Call to a resolvable name and is not flagged."
        )

    def test_PROSE_is_not_an_offender(self, tmp_path):
        """The false positive that made the grep untenable."""
        root = tmp_path / "tests"
        root.mkdir()
        self._plant(root, "test_p.py",
            '"""Explains that the mode changes compute_selection_hash and the leaf."""\n'
            "# NOTE: unreachable through conditional_encdec_sft.main(...)\n"
            "def test_d():\n"
            "    assert True\n")
        suspects, offenders = _audit_tests_tree(root)
        assert not suspects and not offenders

    def test_the_scan_is_anchored_to_the_REPO_not_the_CWD(self, tmp_path, monkeypatch):
        """The first AST rewrite used a CWD-relative Path("tests"), so it passed VACUOUSLY
        from any other directory. One `cd` from useless."""
        monkeypatch.chdir(tmp_path)  # a directory with NO tests/ subtree
        suspects, _ = _audit_tests_tree()
        assert suspects, (
            "the audit found nothing after chdir — it is scanning the CWD, not the repo, "
            "so it would pass vacuously wherever pytest happens to be invoked from"
        )

    def test_the_NON_VACUITY_canary_is_real(self):
        """The repo really does contain files calling compute_selection_hash. If this ever
        goes empty, the audit is broken, not clean."""
        suspects, _ = _audit_tests_tree()
        assert len(suspects) >= 1


def test_explicit_freq_min_zero_still_kills_dead_only(tmp_path):
    """#702: DEFAULT_FREQ_MIN moved 0.0 -> 1e-5, but an EXPLICIT freq_min=0.0
    must still reproduce pre-#702 behavior byte-for-byte (kill count==0 dead
    features only), independent of whatever DEFAULT_FREQ_MIN currently is.
    n_segments=200_000 is deliberately large enough that count=2 WOULD be
    killed under the new 1e-5 default (1e-5 * 200_000 = 2.0, and 2 <= 2.0),
    but must NOT be killed when freq_min=0.0 is passed explicitly (2 > 0) --
    this is what actually discriminates "explicit 0.0" from "the new
    default," which a small-n_segments fixture cannot do."""
    write_stats(
        tmp_path, n_segments=200_000, counts={2: 2, 3: 50}, n_sae_features=8,
    )
    res = fb.compute_activation_frequency_blacklist(
        tmp_path, 8, freq_min=0.0, freq_max=None,
    )
    assert 2 not in res.ids  # count=2 > 0 -> survives under explicit 0.0
    assert 3 not in res.ids  # count=50 > 0 -> survives
    assert set(res.ids) == {0, 1, 4, 5, 6, 7}  # only the dead (count==0) fids


def test_explicit_freq_min_zero_differs_from_new_default_at_same_corpus(tmp_path):
    """Companion to the above: the SAME corpus under the NEW default (1e-5,
    passed explicitly to isolate compute_activation_frequency_blacklist from
    resolve_blacklist_bounds) DOES kill count=2, proving the two are
    genuinely different behaviors and the test above isn't vacuous."""
    write_stats(
        tmp_path, n_segments=200_000, counts={2: 2, 3: 50}, n_sae_features=8,
    )
    res = fb.compute_activation_frequency_blacklist(
        tmp_path, 8, freq_min=1e-5, freq_max=None,
    )
    assert 2 in res.ids  # count=2 <= 1e-5 * 200_000 = 2.0 -> killed
    assert 3 not in res.ids  # count=50 > 2.0 -> survives
