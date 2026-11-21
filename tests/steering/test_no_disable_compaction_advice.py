"""Nothing in the codebase may tell an operator to DISABLE COMPACTION (#695).

The owner's rule, verbatim:

    "Don't disable compaction. Even if it would cause a NotImplementedError for our
     feature prediction-head configs. This will be addressed in the next PR. Generally
     we should be okay with hitting NotImplementedErrors until we wrap up the final PR
     for this."

Compaction is ALWAYS ON whenever a ``feature_mask`` resolves. The prediction head now
uses the effective vocabulary, while compact self-conditioned generation and several
eval entry points remain intentionally unsupported and raise ``NotImplementedError``.
Universal feature-GCG supports compacted checkpoints.

What is NOT fine is the *remedy*. Nulling ``feature_mask_freq_min`` /
``feature_mask_freq_max`` silently opts a run OUT of compaction: the run looks fine and
simply forfeits the saving. An honest crash beats an invisible regression. Commit
``3473412d`` deliberately DELETED that advice — and a later fix commit RE-ADDED it to
``fp_head_unavailable_error``. This file exists so that cannot happen a third time.

**This scanner was already too weak once.** Its first version had two detectors: the
literal ``feature_mask_freq_min: null`` token, and the word "disabl*" next to
"compaction"/"band". Two live sites walked straight through it and shipped —

* ``configuration.md:501``  "Set **both** bounds to `null` to opt out."
  — never names the knobs, never says "disable".
* ``configuration.md:446``  "Set `feature_prediction: null` (the default), or disable the
  mask." — says "disable", but of *the mask*, which was not in the alternation. Disabling
  the mask IS what disables compaction.

So the detectors below match on the SHAPE of the advice, not on one spelling of it:
an imperative verb (``set``/``leave``/``pin``…) aimed at the band/bounds/knobs/mask and
landing on ``null``; a disabling verb (``disable``/``turn off``/``remove``/``drop``…)
aimed at compaction/the band/the mask; the verb ``null`` itself; and "opt out" in a
compaction context. ``TestTheGuardHasTeeth`` pins all of this against the VERBATIM text
of the two findings that got past. Anything weaker is theatre.

Text is normalised (markdown ``*`` and backticks stripped) before matching, so
``**both** bounds to `null``` reads as ``both bounds to null`` — and, just as importantly,
so does the *prohibition* ``Do **not**``, which the char-window excuse could not see
through before.

Strictness is graded:

* **The recipe** (``feature_mask_freq_min: null``) is a FINGERPRINT. It is not how anyone
  documents a nullable knob — the schema docs write
  ``feature_mask_freq_min: 1e-5  # null or float in [0, 1)`` — so a literal ``: null`` on a
  band knob only ever appears in prescriptive advice. Allow-list only, no excuses.
* **The prose** is excused when the ADVICE UNIT it sits in forbids the remedy, because
  "do not disable compaction to dodge it" is exactly what we DO want the codebase to say.
  The unit is the enclosing markdown element (bullet / table row / blockquote line /
  paragraph) or, in code, the enclosing blank-line-delimited block — NOT a blind ±120-char
  window. A window straddles into the NEXT bullet, so a prohibition one bullet away
  silently excused an offence in this one (finding 9 sits two bullets from a "rather
  than"); and a window is anchored to characters, so it grew and shrank with unrelated
  edits. The unit is the thing a reader actually reads as one instruction.

Two files carry allow-listed SNIPPETS. In each, nulling the band is a genuine REQUIREMENT,
not a dodge — a fact about what the operator can even compute, not a way to skip compaction:

* ``freq_blacklist.py`` — the band cannot be computed without
  ``feature_statistics.safetensors``. With no statistics artifact there is nothing to
  threshold. Nulling it is the honest fix.
* ``feature_vocab.py`` — resuming a PRE-#695 checkpoint whose encoder is *physically*
  uncompacted. A resolving mask would size the tables differently and the load would
  fail. Running it uncompacted is what the checkpoint IS.

**The excuse is pinned to the exact snippets, never to the file.** A whole-file excuse is a
blind spot the size of the file, and round 4 found precisely that: ``freq_blacklist.py`` was
allow-listed for the missing-statistics carve-out above, and a SECOND error message in the
same file — the stats WIDTH-MISMATCH one, where the statistics exist and are merely the wrong
ones — sat behind the same excuse while prescribing the forbidden remedy. It shipped.

So ``ALLOWED_ADVICE`` maps each file to the exact strings it is excused FOR, and
``test_the_allow_list_has_no_stale_entries`` asserts EQUALITY: new advice in an allow-listed
file still fails, and an excuse whose text is gone fails as a dead entry.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from errors import FeatureMaskError
from steering.conditional.encoder_decoder.model import fp_head_unavailable_error
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.freq_blacklist import compute_activation_frequency_blacklist
from tests.steering._freq_blacklist_fixtures import write_stats

REPO_ROOT = Path(__file__).resolve().parents[2]

SCANNED_ROOTS = ("steering", "scripts", "configs", "docs", "nanochat")
SCANNED_SUFFIXES = (".py", ".yaml", ".yml", ".md")

# ...PLUS the repo's top-level files, which the roots above miss entirely. `AGENTS.md` is the
# single highest-leverage place the forbidden advice could land — it is what every agent reads
# first, and this PR edits its feature_mask bullets — and it was never scanned. `README.md` and
# `errors.py` were equally invisible. Scanned as a FLAT glob, not an rglob: a recursive walk of
# the repo root would descend into .venv/, .git/ and out/.
SCANNED_TOP_LEVEL = True

# Nulling the band is a REQUIREMENT here, not a dodge. See the module docstring.
#
# Pinned to the EXACT excused snippets, not to the file. A whole-file excuse is a blind spot
# the size of the file, and round 4 found exactly that: `freq_blacklist.py` was allow-listed
# for its missing-statistics carve-out, and a SECOND error message in the same file — the
# width-mismatch one, where the statistics exist and are merely the wrong ones — quietly
# prescribed the forbidden remedy behind the same excuse. Any snippet not listed here is a
# NEW offence and fails, even in an allow-listed file.
ALLOWED_ADVICE: dict[str, frozenset[str]] = {
    # The missing-statistics carve-out: with no statistics there is nothing to compute a band
    # FROM, so nulling it is the genuine remedy. Plus the schema's own descriptive prose.
    "steering/conditional/freq_blacklist.py": frozenset(
        {
            "feature_mask_freq_min: null",
            "feature_mask_freq_max: null",
            "set both bounds to null",
            # The missing-statistics error, and ONLY that one: with no statistics artifact
            # there is nothing to threshold, so nulling the band is the genuine remedy. The
            # WIDTH-MISMATCH error one screen below used to say the very same sentence while
            # the statistics existed and were merely the WRONG ones — a snippet set cannot
            # tell two identical phrases apart, so that message is pinned separately and
            # behaviourally by ``TestTheWidthMismatchErrorDoesNotPrescribeTheRemedy``.
            "disable the auto activation-frequency blacklist",
            "skip that bound",
            "skips that bound",
        }
    ),
    # The legacy-checkpoint error, which names the knobs in order to FORBID them — a pre-#695
    # checkpoint is PHYSICALLY uncompacted, so it is the one case where no mask may resolve.
    # FINGERPRINT takes no excuses by design, so the prohibition still lands here.
    "steering/conditional/feature_vocab.py": frozenset(
        {"feature_mask_freq_min: null", "feature_mask_freq_max: null"}
    ),
}
ALLOWED = frozenset(ALLOWED_ADVICE)


# Markdown emphasis and code spans are noise. `**both** bounds to `null`` must read as
# "both bounds to null" — and `Do **not**` must read as "do not", which is what the old
# raw-text excuse could not see. Underscores are NOT stripped: they are inside every
# identifier that matters (feature_mask_freq_min).
def _normalize(text: str) -> str:
    return text.replace("*", "").replace("`", "")


# What the advice is aimed at. "mask" is here because finding 10 aimed at the mask, and
# disabling the mask IS what disables compaction.
_TARGET = (
    r"(?:encoder\s+)?(?:frequency[-\s])?"
    r"(?:compaction|bands?|bounds?|knobs?|masks?|feature_masks?|feature_mask_freq_(?:min|max))\b"
)
_DET = r"(?:the|this|that|these|those|both|either|its|their|your|our|any|a|one)\s+"

# The FREQUENCY blacklist — the thing the band actually PRODUCES, so "disable the blacklist" IS
# "null the band", said in the words an operator would naturally reach for. Round 4 found that
# exact sentence shipped in `freq_blacklist.py` while this scanner reported the tree clean.
#
# It must be QUALIFIED. A bare `blacklists?` in `_TARGET` sweeps up an unrelated subsystem —
# `steering/eval/autointerp_filter/document_blacklist.py` talks about a "disabled document
# blacklist", which has nothing to do with feature masks — and biting the codebase's own
# vocabulary is how a guard gets weakened back out again. So the noun only counts when the
# words before it say WHICH blacklist: auto / activation / frequency / feature.
_FREQ_BLACKLIST = (
    r"(?:auto|automatic|activation|frequency|feature)[\w\s-]{0,24}?(?:black|white)lists?\b"
)

# Verbs strong enough to be unambiguous. The broad set below reads adjectives and nouns as
# verbs once the object can be a bare noun phrase — "a non-EMPTY computed blacklist", "the
# SKIP would mask", "DISABLES the upper bound" (schema prose) — so anything whose object is a
# blacklist or a pronoun has to come from this shorter, verb-only list.
_STRONG_DISABLE_VERB = (
    r"(?:disabl\w*|turn(?:s|ing)?\s+off|switch(?:es|ing)?\s+off|unset\w*"
    r"|null(?:s|ing|ed)?|remov\w*|delet\w*|drop(?:s|ping)?)"
)

# Context tight enough to be an ANTECEDENT. The loose CONTEXT below includes `blacklist`
# itself, which would make a blacklist-gated detector self-satisfying, and matches so widely
# that "omit it" in an unrelated eval script reads as an offence.
TIGHT_CONTEXT = re.compile(
    r"compact\w*|feature_mask|freq_(?:min|max)|frequency[-\s]band", re.IGNORECASE
)

# A `null` (or `disabled`) that is a YAML/dict VALUE is not a verb. Without this, the gap
# in `feature_ids: null\nfeature_mask: null` reads as the phrase "null feature_mask" —
# two unrelated keys, one bogus offence. Prose is unaffected: "to null the band", "or
# disable the mask" are never preceded by a colon.
_NOT_A_VALUE = r"(?<!:)(?<!: )"

# 1. THE FINGERPRINT — a band knob, literally set to null. Not how anyone documents a
#    nullable knob, so it only ever appears in prescriptive advice. NO excuses; allow-list
#    only. (`feature_mask_freq_max: 0.99` is a pin, not a null, and is not this.)
FINGERPRINT = re.compile(
    r"feature_mask_freq_(?:min|max)\s*(?::\s*null|[^\n]{0,45}?\bto\s+null)",
    re.IGNORECASE,
)

# 2. Disabling verbs aimed at compaction / the band / THE MASK. Finding 10 hid in the gap
#    here: the old alternation was (compaction|band), so "or disable the mask" — the same
#    instruction, one noun over — scanned clean.
DISABLE = re.compile(
    rf"{_NOT_A_VALUE}\b(?:disabl\w*|turn(?:s|ing)?\s+off|switch(?:es|ing)?\s+off|remov\w*"
    rf"|delet\w*|drop(?:s|ping)?|unset\w*|omit(?:s|ting)?|skip(?:s|ping)?|clear(?:s|ing)?"
    rf"|wipe[sd]?|wiping|strip(?:s|ping)?|empt(?:y|ies|ying)|blank\w*)"
    rf"\s+(?:out\s+)?(?:{_DET})?{_TARGET}",
    re.IGNORECASE,
)

# 2b. The same instruction, aimed at the frequency BLACKLIST — the noun the band produces.
DISABLE_BLACKLIST = re.compile(
    rf"{_NOT_A_VALUE}\b{_STRONG_DISABLE_VERB}\s+(?:out\s+)?(?:{_DET})?{_FREQ_BLACKLIST}",
    re.IGNORECASE,
)

# 2c. The same instruction, aimed at a PRONOUN: "Widen the mask, or disable it." The shortest
#     way to say the forbidden thing, and it names no noun at all — so no noun alternation can
#     ever reach it. Only a tight compaction context in the same advice unit can, which is also
#     what stops it biting an unrelated "the progress bar is noisy — disable it".
DISABLE_PRONOUN = re.compile(
    rf"{_NOT_A_VALUE}\b{_STRONG_DISABLE_VERB}\s+(?:it|them)\b",
    re.IGNORECASE,
)

# 3. The verb "null", aimed at the band: "null the frequency band", "nulling these bounds".
NULL_VERB = re.compile(
    rf"{_NOT_A_VALUE}\bnull(?:s|ing|ed)?\s+(?:out\s+)?(?:{_DET})?(?:{_TARGET}|them\b)",
    re.IGNORECASE,
)

# 4. An imperative aimed at the band/mask and landing on null. Finding 9 hid here:
#    "Set **both** bounds to `null`" names no knob, so the fingerprint never saw it.
#    The IMPERATIVE is what separates prescriptive advice from the schema's own
#    descriptive "Either bound can be `null` to skip", which must survive.
SET_TO_NULL = re.compile(
    r"\b(?:set|sets|setting|pin|pins|pinning|leave|leaves|leaving|configure|configures"
    r"|configuring|pass|passes|passing|put|puts|revert|reverts|reverting|make|makes|making)\b"
    r"[^.;\n]{0,45}?\b(?:bounds?|bands?|knobs?|masks?|feature_mask\w*)\b"
    r"[^.;\n]{0,30}?\bto\s+null\b",
    re.IGNORECASE,
)

# 5. "opt out" — finding 9's actual words ("Set both bounds to `null` to opt out"), which
#    name neither a knob nor "disable". Gated on a compaction CONTEXT so that opting out of
#    something unrelated is not swept up.
OPT_OUT = re.compile(r"\bopt(?:s|ed|ing)?\b[^.;\n]{0,25}?\bout\b", re.IGNORECASE)
CONTEXT = re.compile(
    r"compact\w*|feature_mask|frequency[-\s]band|freq_(?:min|max)|blacklist|whitelist",
    re.IGNORECASE,
)

# (label, pattern, needs a compaction context, may be excused by a prohibition)
# (label, pattern, the context gate this detector needs — None for "no gate", may be excused)
DETECTORS: tuple[tuple[str, re.Pattern[str], re.Pattern[str] | None, bool], ...] = (
    ("the band-nulling recipe", FINGERPRINT, None, False),
    ("disabling the mask / band / compaction", DISABLE, None, True),
    ("disabling the frequency blacklist", DISABLE_BLACKLIST, None, True),
    ("disabling it / them", DISABLE_PRONOUN, TIGHT_CONTEXT, True),
    ("nulling the band", NULL_VERB, None, True),
    ("setting the band / mask to null", SET_TO_NULL, None, True),
    ("opting out of compaction", OPT_OUT, CONTEXT, True),
)

# A hit is excused when its own ADVICE UNIT forbids the remedy rather than offering it:
# "do not disable compaction to dodge it" is exactly what we DO want the codebase to say.
#
# Two rules keep the excuse from becoming a loophole:
#
# 1. It must PRECEDE the advice it excuses (enforced in ``offending_advice``). "rather than
#    nulling their band" negates the nulling; "Set both bounds to null to opt out. Do this
#    rather than fighting the loader." does not — the same words, after the fact, excusing
#    advice they have nothing to do with. Round 4 verified that the second form scanned CLEAN.
# 2. A bare "never" is not a prohibition. "Compaction never applies without a mask. Set both
#    bounds to null to opt out." would otherwise excuse itself with a purely descriptive
#    sentence. Only "never <verb>" forbids anything.
PROHIBITIONS = (
    "do not",
    "don't",
    "never null",
    "never disable",
    "never set",
    "never reach",
    "not a fix",
    "not a workaround",
    "does not fix",
    "fixes nothing",
    "not optional",
    "no knob",
    "rather than",
    "instead of",
    "take the crash",
    "must not",
    "worse than a crash",
)

# The band knobs must not be named as the way out of a compaction NotImplementedError.
BAND_KNOBS = ("feature_mask_freq_min", "feature_mask_freq_max")

# A new markdown ELEMENT — bullet, numbered item, table row, heading, blockquote line, or
# blank. Everything else is a continuation of the element above (a wrapped line), so a
# wrapped bullet stays ONE unit while two sibling bullets stay TWO. Matched on RAW lines,
# before `*` is stripped, so `* item` still reads as a bullet.
_MD_ELEMENT = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|\||#{1,6}\s|>)")


def _units(text: str, *, markdown: bool) -> list[str]:
    """The spans a reader reads as ONE instruction.

    The excuse is scoped to these, not to a ±120-char window. A window straddles the
    boundary into the NEXT bullet — which is how finding 9 came within one bullet of being
    excused by a "rather than" that had nothing to do with it — and it drifts with every
    unrelated edit nearby. In markdown a unit is a markdown element; in code it is a
    blank-line-delimited block, which is exactly one docstring paragraph or one error
    message.
    """
    units: list[list[str]] = []
    for line in text.splitlines():
        blank = not line.strip()
        starts = blank or (markdown and _MD_ELEMENT.match(line) is not None)
        if starts or not units:
            units.append([line])
        else:
            units[-1].append(line)
    return ["\n".join(u) for u in units]


def offending_advice(text: str, *, markdown: bool = True, excusable: bool = True) -> list[str]:
    """Every unexcused piece of disable-compaction advice in ``text``."""
    found: list[str] = []
    for unit in _units(text, markdown=markdown):
        norm = _normalize(unit)
        low = norm.lower()
        starts = [low.find(p) for p in PROHIBITIONS if p in low]
        # Where the unit first turns against the remedy. A prohibition only negates advice it
        # PRECEDES — see the note on PROHIBITIONS.
        forbids_from = min(starts) if starts else None
        for _label, pattern, gate, det_excusable in DETECTORS:
            if gate is not None and not gate.search(norm):
                continue
            for m in pattern.finditer(norm):
                excused_here = (
                    det_excusable
                    and excusable
                    and forbids_from is not None
                    and forbids_from < m.start()
                )
                if not excused_here:
                    found.append(" ".join(m.group(0).split()))
    return sorted(set(found))


# The two findings the guard SHIPPED PAST, verbatim (configuration.md:501 and :446 as they
# stood). These are the teeth-check: a scanner that cannot flag the text that just walked
# through it is theatre.
FINDING_9 = (
    "- **Compaction:** a non-empty computed blacklist compacts the encoder (#695), which "
    "forbids `feature_prediction`, `feature_gcg`, self-conditioned generation and "
    "legacy-masked resume. Set **both** bounds to `null` to opt out."
)
FINDING_10 = (
    "- A `feature_mask` (including the one the default frequency band auto-produces) is "
    "**incompatible with `feature_prediction`** → `NotImplementedError`. Set "
    "`feature_prediction: null` (the default), or disable the mask."
)

# Round 4: TWO MORE, both shipped in PRODUCTION while this scanner reported the tree clean.
# Same failure as finding 10 — the fix for that one added `masks?` to the target alternation
# and stopped there, so the very noun the frequency band PRODUCES (`blacklist`) still walked
# through, as did the pronoun. Twice now the alternation has been patched to exactly the word
# that was caught and no further; hence `_ADJ` and the pronoun detector, which close the SHAPE
# rather than the instance.
FINDING_R4_BLACKLIST = (
    "different SAE — point feature_dir at the matching statistics, "
    "or disable the auto activation-frequency blacklist."
)
FINDING_R4_PRONOUN = (
    "feature_mask (mode=blacklist) allows zero features out of 65536 — the encoder would "
    "have no embedding rows. Widen the mask, or disable it."
)

# Prose that names the knobs in order to FORBID them. It must survive the scanner — the
# codebase is SUPPOSED to say this — so every one of these is a false-positive check.
PROHIBITION_SITES = (
    "Do *not* null the frequency band to dodge it: that forfeits compaction silently "
    "instead of fixing anything.",
    "The FP-head ablation configs deliberately hit that `NotImplementedError` rather than "
    "nulling their band, which would silently opt them out of compaction.",
    "Do **not** reach for the frequency-band knobs to get around this. Nulling them does "
    "not fix the gap — it silently opts the run out of compaction, so the run looks fine "
    "and merely forfeits the saving.",
    "> **Encoder compaction (#695) is not optional.** There is no knob that turns that "
    "off, and **nulling these bounds is not a workaround for it**.",
)


def _scanned_files() -> list[Path]:
    files = [
        p
        for root in SCANNED_ROOTS
        for p in (REPO_ROOT / root).rglob("*")
        if p.is_file() and p.suffix in SCANNED_SUFFIXES
    ]
    if SCANNED_TOP_LEVEL:
        files += [p for p in REPO_ROOT.glob("*") if p.is_file() and p.suffix in SCANNED_SUFFIXES]
    assert len(files) > 100, f"the scan matched only {len(files)} files — the roots are wrong"
    return sorted(files)


def _scan_file(path: Path) -> list[str]:
    return offending_advice(
        path.read_text(encoding="utf-8", errors="ignore"), markdown=path.suffix == ".md"
    )


def _scan() -> dict[str, list[str]]:
    """Every unexcused piece of advice, per file.

    In an allow-listed file this subtracts only the snippets that file is allow-listed FOR —
    so a NEW offence there is still an offence. Excusing the whole file is what let the
    width-mismatch error prescribe the forbidden remedy for two rounds.
    """
    offenders: dict[str, list[str]] = {}
    for path in _scanned_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        excused = ALLOWED_ADVICE.get(rel, frozenset())
        found = [f for f in _scan_file(path) if f.lower() not in {e.lower() for e in excused}]
        if found:
            offenders[rel] = found
    return offenders


def _report(offenders: dict[str, list[str]]) -> str:
    return (
        "these files prescribe DISABLING COMPACTION — the one remedy the owner explicitly "
        "forbade. Nulling the frequency band (or the mask, which is the same thing) "
        "silently opts a run out of compaction: the run looks fine and simply forfeits the "
        "saving. An honest NotImplementedError is the intended behaviour until the "
        "follow-up PRs land. Rewrite the remedy (or, if the mention is a PROHIBITION, "
        "phrase it as one):\n" + "\n".join(f"  {f}: {v}" for f, v in sorted(offenders.items()))
    )


class TestTheGuardHasTeeth:
    """The guard must flag the advice that ALREADY walked straight past it.

    Findings 9 and 10 sat in ``docs/reference/configuration.md`` while this very file
    scanned that very file and reported it clean. Every detector below is pinned against
    the verbatim text it failed to catch.
    """

    @pytest.mark.parametrize(
        ("finding", "text"),
        [
            ("9", FINDING_9),
            ("10", FINDING_10),
            ("r4-blacklist", FINDING_R4_BLACKLIST),
            ("r4-pronoun", FINDING_R4_PRONOUN),
        ],
    )
    def test_it_flags_the_advice_that_shipped_past_it(self, finding: str, text: str) -> None:
        assert offending_advice(text), (
            f"FINDING {finding} walked past this scanner once already. It is the "
            "forbidden remedy in prose — 'set both bounds to null to opt out' / 'or "
            "disable the mask' / 'or disable the auto activation-frequency blacklist' / "
            "'Widen the mask, or disable it' — and the scanner still reports it clean. A "
            "guard that cannot catch the thing that just got past it is theatre."
        )

    def test_the_scanner_would_have_caught_finding_9s_verb(self) -> None:
        """Not just "some detector fires" — the RIGHT one, on the words that got past."""
        assert "Set both bounds to null" in offending_advice(FINDING_9)
        assert "opt out" in offending_advice(FINDING_9)

    def test_the_scanner_would_have_caught_finding_10s_verb(self) -> None:
        assert "disable the mask" in offending_advice(FINDING_10)

    def test_the_scanner_would_have_caught_the_BLACKLIST_noun(self) -> None:
        """The noun the frequency band actually PRODUCES. Twice the alternation has been
        patched to exactly the word that was caught and no further; this pins the one that
        makes the whole class of advice sayable."""
        assert "disable the auto activation-frequency blacklist" in offending_advice(
            FINDING_R4_BLACKLIST
        )

    def test_the_scanner_would_have_caught_the_PRONOUN(self) -> None:
        """ "or disable it" is the shortest way to say the forbidden thing, and it names no
        noun at all — so it can only be caught by requiring a compaction context."""
        assert "disable it" in offending_advice(FINDING_R4_PRONOUN)

    def test_the_pronoun_detector_needs_a_compaction_CONTEXT(self) -> None:
        """...which also means it must NOT bite an unrelated "disable it"."""
        assert not offending_advice("The progress bar is noisy in CI. Disable it with --quiet.")

    @pytest.mark.parametrize(
        "text",
        [
            # The excuse must PRECEDE the advice. These append a prohibition-shaped phrase
            # AFTER it, which excused the whole unit — verified clean by round 4.
            FINDING_9 + " Do this rather than fighting the loader.",
            FINDING_10 + " Disable the mask instead of retraining from scratch.",
            # A bare "never" used descriptively is not a prohibition of anything.
            "- Compaction never applies without a mask. Set both bounds to null to opt out.",
        ],
    )
    def test_a_prohibition_AFTER_the_advice_does_not_excuse_it(self, text: str) -> None:
        """The excuse is the loophole, and it was wide open.

        ``prohibited`` was computed once per unit and short-circuited every excusable detector
        in it, so ANY of the generic phrases ("rather than", "instead of", "never") appearing
        ANYWHERE in the unit — including after the advice, about something else entirely —
        turned a live offence into a clean scan. It now has to come FIRST, and a bare "never"
        no longer counts.
        """
        assert offending_advice(text), f"a trailing excuse laundered the advice: {text!r}"

    def test_AGENTS_md_is_actually_scanned(self) -> None:
        """The single highest-leverage place for this advice to land, and it was out of scope.

        ``SCANNED_ROOTS`` covered steering/scripts/configs/docs — so ``AGENTS.md`` (which every
        agent reads first, and whose feature_mask bullets THIS PR edits), ``README.md``,
        ``errors.py`` and all of ``nanochat/`` were never scanned at all.
        """
        scanned = {p.relative_to(REPO_ROOT).as_posix() for p in _scanned_files()}
        assert "AGENTS.md" in scanned
        assert any(p.startswith("nanochat/") for p in scanned)

    @pytest.mark.parametrize(
        "text",
        [
            # A DIFFERENT blacklist. steering/eval/autointerp_filter/document_blacklist.py is
            # an unrelated subsystem, and a guard that bites the codebase's own vocabulary is
            # a guard someone will weaken back out again.
            "Falls back to a disabled document blacklist when no artifact is present.",
            "Empty blacklist — every feature survives the filter.",
            "a non-empty computed blacklist compacts the encoder",
            # Descriptive schema prose for a nullable knob, not a prescription.
            "upper_bound: null  # null disables the upper bound",
            "The skip would mask a real error, so it is not taken.",
            "This omits it from the manifest.",
        ],
    )
    def test_it_does_NOT_bite_the_codebases_own_vocabulary(self, text: str) -> None:
        """Over-correcting is the same bug in the other direction.

        A first attempt at this fix put a bare ``blacklists?`` in ``_TARGET`` and allowed an
        adjective run between the determiner and the noun. It flagged SEVEN unrelated sites: a
        different subsystem's blacklist, "empty" and "skip" read as verbs, schema docs for an
        unrelated bound. A guard that cries wolf is a guard someone turns off — which is
        exactly the failure this file exists to prevent.
        """
        assert not offending_advice(text), f"false positive on: {text!r}"

    @pytest.mark.parametrize("text", PROHIBITION_SITES)
    def test_it_does_NOT_flag_prose_that_FORBIDS_the_remedy(self, text: str) -> None:
        """Teeth are worthless if they bite the codebase's own warnings."""
        assert not offending_advice(text), (
            "this text names the knobs in order to FORBID them — exactly what the codebase "
            f"is supposed to say — and the scanner flagged it anyway: {text!r}"
        )


class TestNoFilePrescribesIt:
    def test_no_file_prescribes_disabling_compaction(self) -> None:
        offenders = _scan()
        assert not offenders, _report(offenders)

    def test_the_allow_list_has_no_stale_entries(self) -> None:
        """A file that no longer needs the excuse must lose it — a stale entry is a hole.

        This is what makes the allow-list self-checking: it does not merely permit two
        files, it asserts they still contain the text they are permitted FOR. The day
        ``feature_vocab.py``'s legacy-resume error stops naming the band, its entry stops
        excusing anything and starts hiding the next regression instead.
        """
        for rel, excused in sorted(ALLOWED_ADVICE.items()):
            path = REPO_ROOT / rel
            assert path.is_file(), f"allow-listed file does not exist: {rel}"
            found = {f.lower() for f in _scan_file(path)}
            expected = {e.lower() for e in excused}
            assert found == expected, (
                f"{rel}'s allow-list no longer matches what is in the file.\n"
                f"  NEW, unexcused advice: {sorted(found - expected) or 'none'}\n"
                f"  dead entries:          {sorted(expected - found) or 'none'}\n"
                "The allow-list is pinned to exact snippets on purpose. New advice in an "
                "allow-listed file is still advice — a whole-file excuse is how the "
                "width-mismatch error prescribed the forbidden remedy for two rounds. If the "
                "new text is a legitimate carve-out, add that snippet and say why."
            )


class TestTheFpHeadErrorDoesNotDodge:
    """A compact model may own an FP head; head absence is no longer a gap."""

    def _error(self) -> RuntimeError:
        class _Model:
            feature_vocab = FeatureVocab(
                n_sae_features=64, original_ids=(0, 3, 7, 11)
            )

        return fp_head_unavailable_error(
            _Model(),  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            requested="self-conditioned generation",
            uncompacted="this checkpoint was built without feature_prediction",
        )

    def test_it_does_NOT_tell_the_operator_to_null_the_frequency_band(self) -> None:
        msg = str(self._error())
        for knob in BAND_KNOBS:
            assert knob not in msg, (
                f"the FP-head error names {knob!r} — i.e. it hands the operator the "
                "band-nulling recipe, which silently opts the run OUT of compaction. That "
                "is the exact advice commit 3473412d deleted and the owner forbade."
            )
        assert not offending_advice(msg, markdown=False)

    def test_compaction_no_longer_changes_the_missing_head_error(self) -> None:
        err = self._error()
        assert type(err) is RuntimeError
        assert str(err) == "this checkpoint was built without feature_prediction"


class TestTheTrainingConfigsDoNotDodge:
    """FINDING 4. The freq-band comment block, copy-pasted between the two configs."""

    CONFIGS = ("conditional_encdec_train.yaml", "conditional_encdec_sft.yaml")

    @pytest.mark.parametrize("name", CONFIGS)
    def test_the_band_comment_does_not_offer_nulling_as_a_way_out(self, name: str) -> None:
        text = (REPO_ROOT / "configs" / "training" / name).read_text(encoding="utf-8")
        assert not offending_advice(text, markdown=False), (
            f"{name} still says nulling the band disables compaction. It does — and that is "
            "precisely why the operator must not be told to do it."
        )

    @pytest.mark.parametrize("name", CONFIGS)
    def test_the_band_is_still_SET_so_the_run_stays_compaction_eligible(self, name: str) -> None:
        """Purging the advice must not quietly purge the band with it."""
        cfg = yaml.safe_load((REPO_ROOT / "configs" / "training" / name).read_text())
        assert cfg["feature_mask_freq_min"] is not None
        assert cfg["feature_mask_freq_max"] is not None

    @pytest.mark.parametrize("name", CONFIGS)
    def test_the_comment_still_TELLS_the_operator_compaction_is_always_on(self, name: str) -> None:
        text = (REPO_ROOT / "configs" / "training" / name).read_text(encoding="utf-8").lower()
        assert "compaction" in text, f"{name} must still explain that a mask compacts the encoder"


class TestTheWidthMismatchErrorDoesNotPrescribeTheRemedy:
    """The one message a snippet allow-list cannot police, pinned by BEHAVIOUR.

    ``freq_blacklist.py`` holds two errors. The **missing-statistics** one legitimately says
    "disable the auto activation-frequency blacklist" — with no statistics artifact there is
    nothing to threshold, so nulling the band really is the remedy — and that sentence is
    allow-listed. The **width-mismatch** one, one screen below, used to say *the very same
    sentence* while the statistics existed and were merely the WRONG ones. It shipped, behind
    the whole-file excuse.

    ``ALLOWED_ADVICE`` is a set of snippets, so it cannot tell two identical phrases apart:
    re-adding that sentence to the width-mismatch error would not change the set. Only calling
    the code and reading the message it actually raises can catch that, so this does.
    """

    def _raise_width_mismatch(self, tmp_path: Path) -> str:
        write_stats(tmp_path, n_segments=100, counts={0: 50}, n_sae_features=16)
        with pytest.raises(FeatureMaskError) as exc:
            compute_activation_frequency_blacklist(
                tmp_path, 64, freq_min=1e-5, freq_max=0.5
            )  # statistics say 16 features, the config says 64
        return str(exc.value)

    def test_it_names_the_REAL_fixes(self, tmp_path: Path) -> None:
        msg = self._raise_width_mismatch(tmp_path)
        assert "feature_dir" in msg, "the fix is to point feature_dir at the right statistics"
        assert "sae.width" in msg, "...or to correct sae.width. Those are the only two."

    def test_it_does_NOT_offer_the_forbidden_remedy(self, tmp_path: Path) -> None:
        msg = self._raise_width_mismatch(tmp_path)
        offences = offending_advice(msg, markdown=False)
        assert not offences, (
            "the width-mismatch error prescribes disabling compaction. The statistics EXIST "
            f"here and are simply the wrong ones — nulling the band would not read them, it "
            f"would drop the mask entirely. offences={offences}"
        )

    def test_it_FORBIDS_the_band_knobs_by_name(self, tmp_path: Path) -> None:
        """Naming the knobs in order to forbid them is exactly what the codebase SHOULD do."""
        msg = self._raise_width_mismatch(tmp_path)
        low = msg.lower()
        assert "do not null" in low
        assert "feature_mask_freq_min" in msg and "feature_mask_freq_max" in msg


def test_the_eval_conditioning_deferral_message_is_not_disable_advice() -> None:
    """The #695 eval-conditioning deferral message (the NotImplementedError raised when an eval
    tries to load a COMPACTED checkpoint) must never read as disable-compaction advice: it points
    the operator at ``dump_feature_mask`` to RECOVER the mask, never at nulling/disabling a band or
    the mask. Pins the real message through the same detector the file-scanner uses."""
    with pytest.raises(NotImplementedError) as exc_info:
        EvalConditioning.deferred("base_eval").assert_loadable(
            FeatureVocab(n_sae_features=64, original_ids=(0, 3)),
            checkpoint_dir="/some/checkpoint",
        )
    msg = str(exc_info.value)
    assert offending_advice(msg, markdown=False) == [], (
        f"the eval-conditioning deferral message reads as disable-compaction advice: {msg!r}"
    )
