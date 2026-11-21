"""Tests for ``scripts.analysis.dump_feature_mask`` (issue #695 spec §4.9.4).

The tool is the operator's escape hatch. Evaluating a COMPACTED checkpoint requires
re-specifying the EXACT training mask, and the default frequency blacklist lives in a
*generated* ``.txt`` under ``{feature_dir}/blacklist/`` — a file a ``feature_dir`` move
or a statistics recompute would destroy. The checkpoint already carries the truth (in
``meta.feature_mask_canonical``, and once compacted in the ``encoder.feature_vocab_ids``
buffer), so the tool recovers it from the checkpoint ALONE and prints it paste-ready.

Every checkpoint below is a REAL one on disk — a real ``torch.save``-ed state dict plus a
real ``meta_*.json``. ``torch.load`` is never mocked: the entire point of the tool is that
it reads files, and a mocked loader would test nothing.

The load-bearing property, pinned by the round-trip tests: **what the tool prints, fed
back through the production resolver (``resolve_feature_mask`` →
``FeatureVocab.from_mask``), reproduces the checkpoint's vocabulary exactly.**
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import pytest
import torch as t
import yaml

from errors import FeatureMaskError, FeatureVocabError
from scripts.analysis.dump_feature_mask import (
    FORMAT_TXT,
    FORMAT_YAML,
    _read_state_dict,
    load_dumped_mask,
    main,
    render,
)
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    append_mask_leaf,
    feature_mask_dirname,
    resolve_feature_mask,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.manual_feature_ids import parse_feature_ids_txt

# Sentinel: "derive the meta feature_vocab fingerprint from the weights" (what the
# training path does). Distinct from ``None``, which means "omit the key entirely".
_AUTO = object()

N_SAE = 8
STEP = 5


def _write_checkpoint(
    checkpoint_dir: Path,
    step: int = STEP,
    *,
    n_sae: int = N_SAE,
    vocab_ids: Sequence[int] | None = None,
    meta_mask: dict | None = None,
    meta_vocab: object = _AUTO,
    n_embd: int = 4,
    write_model: bool = True,
    write_meta: bool = True,
) -> Path:
    """Build a real encoder-decoder checkpoint on disk.

    ``vocab_ids=None`` ⇒ a LEGACY (uncompacted) checkpoint: a full ``n_sae``-row
    ``feature_embed`` and no vocab buffers, exactly as every checkpoint written before
    #695. Otherwise ⇒ a COMPACTED checkpoint: ``V = len(vocab_ids)`` rows plus the two
    persistent buffers that make the vocabulary recoverable from the file alone.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    compacted = vocab_ids is not None
    rows = len(vocab_ids) if vocab_ids is not None else n_sae

    if write_model:
        state: dict[str, t.Tensor] = {
            # A deterministic, non-degenerate table: row j is filled with j, so a
            # mis-indexed row would be visible rather than accidentally correct.
            "encoder.feature_embed.weight": t.arange(rows, dtype=t.float32)
            .unsqueeze(1)
            .expand(rows, n_embd)
            .contiguous(),
            "encoder.magnitude_embed.weight": t.zeros(4, n_embd),
        }
        if vocab_ids is not None:
            state["encoder.feature_vocab_ids"] = t.tensor(list(vocab_ids), dtype=t.int64)
            state["encoder.feature_vocab_n_sae"] = t.tensor(n_sae, dtype=t.int64)
        t.save(state, checkpoint_dir / f"model_{step:06d}.pt")

    if write_meta:
        meta: dict = {"step": step, "model_config": {"n_sae_features": n_sae}}
        if meta_mask is not None:
            meta["feature_mask_canonical"] = meta_mask
        if meta_vocab is _AUTO:
            if compacted:
                assert vocab_ids is not None
                meta["feature_vocab"] = FeatureVocab(
                    n_sae_features=n_sae, original_ids=tuple(vocab_ids)
                ).metadata_dict()
        elif meta_vocab is not None:
            meta["feature_vocab"] = meta_vocab
        (checkpoint_dir / f"meta_{step:06d}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    return checkpoint_dir


def _blacklist_vocab(n_sae: int, blacklisted: Sequence[int]) -> list[int]:
    """The allowed universe of a blacklist — i.e. what the compact table must hold."""
    return [i for i in range(n_sae) if i not in set(blacklisted)]


def _resolved_vocab_from_yaml(rendered: str, n_sae: int) -> FeatureVocab | None:
    """Round-trip the emitted YAML through the PRODUCTION resolver.

    ``yaml.safe_load`` ⇒ ``resolve_feature_mask`` ⇒ ``FeatureVocab.from_mask`` is exactly
    the path a training/eval run takes with the pasted block, so this proves the paste
    actually works — not merely that our own renderer agrees with our own parser.
    """
    parsed = yaml.safe_load(rendered)
    mask = resolve_feature_mask(parsed, n_sae)
    return FeatureVocab.from_mask(mask, n_sae)


# ===========================================================================
# Compacted + meta mask — the happy path, both formats
# ===========================================================================


class TestCompactedWithMetaMask:
    def test_yaml_emits_the_mask_not_the_vocab(self, tmp_path: Path) -> None:
        """A BLACKLIST checkpoint must emit the blacklist, never its allowed universe.

        The distinction is the whole ballgame: for n_sae=8 and a blacklist of {2, 7} the
        vocab is {0,1,3,4,5,6}. Emitting the vocab ids under ``mode: blacklist`` would
        invert the mask and silently mask out everything the run was trained on.
        """
        _write_checkpoint(
            tmp_path,
            vocab_ids=_blacklist_vocab(N_SAE, [2, 7]),
            meta_mask={"mode": "blacklist", "ids": [2, 7]},
        )
        out = render(load_dumped_mask(tmp_path), FORMAT_YAML)

        assert "feature_mask:\n  mode: blacklist\n  ids: [2, 7]\n" in out
        assert yaml.safe_load(out) == {"feature_mask": {"mode": "blacklist", "ids": [2, 7]}}

    def test_yaml_round_trips_to_the_checkpoint_vocab(self, tmp_path: Path) -> None:
        vocab_ids = _blacklist_vocab(N_SAE, [2, 7])
        _write_checkpoint(
            tmp_path, vocab_ids=vocab_ids, meta_mask={"mode": "blacklist", "ids": [2, 7]}
        )
        dumped = load_dumped_mask(tmp_path)

        assert _resolved_vocab_from_yaml(render(dumped, FORMAT_YAML), N_SAE) == dumped.vocab
        assert dumped.vocab == FeatureVocab(
            n_sae_features=N_SAE, original_ids=tuple(vocab_ids)
        )

    def test_whitelist_mode_round_trips(self, tmp_path: Path) -> None:
        _write_checkpoint(
            tmp_path, vocab_ids=[1, 4, 6], meta_mask={"mode": "whitelist", "ids": [1, 4, 6]}
        )
        dumped = load_dumped_mask(tmp_path)
        out = render(dumped, FORMAT_YAML)

        assert yaml.safe_load(out) == {
            "feature_mask": {"mode": "whitelist", "ids": [1, 4, 6]}
        }
        assert _resolved_vocab_from_yaml(out, N_SAE) == dumped.vocab

    def test_yaml_ids_are_plain_ints(self, tmp_path: Path) -> None:
        """No tensor / numpy reprs may leak into the pasted block."""
        _write_checkpoint(
            tmp_path,
            vocab_ids=_blacklist_vocab(N_SAE, [3]),
            meta_mask={"mode": "blacklist", "ids": [3]},
        )
        out = render(load_dumped_mask(tmp_path), FORMAT_YAML)

        ids = yaml.safe_load(out)["feature_mask"]["ids"]
        assert all(type(i) is int for i in ids)
        assert "tensor(" not in out and "!!python" not in out

    def test_txt_is_parseable_by_the_production_txt_parser(self, tmp_path: Path) -> None:
        _write_checkpoint(
            tmp_path,
            vocab_ids=_blacklist_vocab(N_SAE, [2, 7]),
            meta_mask={"mode": "blacklist", "ids": [2, 7]},
        )
        out = render(load_dumped_mask(tmp_path), FORMAT_TXT)

        ids_file = tmp_path / "ids.txt"
        ids_file.write_text(out, encoding="utf-8")
        assert parse_feature_ids_txt(ids_file) == [2, 7]

    def test_txt_header_names_the_mode(self, tmp_path: Path) -> None:
        """The txt carries ids ONLY — ``feature_mask.path`` still needs an explicit mode.

        If the header omitted it, an operator would have to guess, and guessing wrong
        inverts the mask.
        """
        _write_checkpoint(
            tmp_path,
            vocab_ids=_blacklist_vocab(N_SAE, [2, 7]),
            meta_mask={"mode": "blacklist", "ids": [2, 7]},
        )
        out = render(load_dumped_mask(tmp_path), FORMAT_TXT)

        header = [ln for ln in out.splitlines() if ln.startswith("#")]
        assert any("mode: blacklist" in ln for ln in header)
        assert any("path" in ln for ln in header)

    def test_txt_round_trips_through_feature_mask_path(self, tmp_path: Path) -> None:
        """The full documented `feature_mask.path` route reproduces the vocab."""
        vocab_ids = _blacklist_vocab(N_SAE, [2, 7])
        _write_checkpoint(
            tmp_path, vocab_ids=vocab_ids, meta_mask={"mode": "blacklist", "ids": [2, 7]}
        )
        dumped = load_dumped_mask(tmp_path)

        ids_file = tmp_path / "mask_ids.txt"
        ids_file.write_text(render(dumped, FORMAT_TXT), encoding="utf-8")
        mask = resolve_feature_mask(
            {"feature_mask": {"mode": "blacklist", "path": str(ids_file)}}, N_SAE
        )
        assert FeatureVocab.from_mask(mask, N_SAE) == dumped.vocab

    def test_source_is_the_meta_and_provenance_is_in_the_output(self, tmp_path: Path) -> None:
        _write_checkpoint(
            tmp_path,
            vocab_ids=_blacklist_vocab(N_SAE, [2, 7]),
            meta_mask={"mode": "blacklist", "ids": [2, 7]},
        )
        dumped = load_dumped_mask(tmp_path)
        out = render(dumped, FORMAT_YAML)

        assert dumped.source == "meta"
        assert "feature_mask_canonical" in out
        assert str(tmp_path) in out
        assert f"step {STEP}" in out

    def test_unsorted_meta_ids_are_canonicalised(self, tmp_path: Path) -> None:
        """A hand-edited meta with unsorted/duplicate ids still emits the canonical set."""
        _write_checkpoint(
            tmp_path,
            vocab_ids=_blacklist_vocab(N_SAE, [1, 5]),
            meta_mask={"mode": "blacklist", "ids": [5, 1, 5]},
        )
        out = render(load_dumped_mask(tmp_path), FORMAT_YAML)
        assert yaml.safe_load(out) == {"feature_mask": {"mode": "blacklist", "ids": [1, 5]}}


# ===========================================================================
# Compacted, meta mask DISAGREES with the weights — must raise
# ===========================================================================


class TestMetaVocabDisagreement:
    def test_meta_mask_that_does_not_reproduce_the_vocab_raises(self, tmp_path: Path) -> None:
        """Weights say the vocab excludes {2, 7}; meta says the blacklist is {3}.

        One of the two is wrong and there is no way to tell which — so the tool must
        refuse to emit either. Silently trusting the meta would hand the operator a mask
        that reinterprets every embedding row.
        """
        _write_checkpoint(
            tmp_path,
            vocab_ids=_blacklist_vocab(N_SAE, [2, 7]),
            meta_mask={"mode": "blacklist", "ids": [3]},
            meta_vocab=None,
        )
        with pytest.raises(FeatureVocabError, match="does not reproduce"):
            load_dumped_mask(tmp_path)

    def test_disagreement_message_names_both_sizes(self, tmp_path: Path) -> None:
        _write_checkpoint(
            tmp_path,
            vocab_ids=[1, 4, 6],
            meta_mask={"mode": "whitelist", "ids": [1, 4]},
            meta_vocab=None,
        )
        with pytest.raises(FeatureVocabError) as exc:
            load_dumped_mask(tmp_path)

        msg = str(exc.value)
        assert "2" in msg and "3" in msg  # mask allows V'=2, weights carry V=3
        assert str(tmp_path) in msg

    def test_message_marks_an_empty_difference_direction_as_none(self, tmp_path: Path) -> None:
        """A strict-subset mask disagrees in ONE direction only; the other must read
        cleanly rather than as an empty list."""
        _write_checkpoint(
            tmp_path,
            vocab_ids=[1, 4, 6],
            meta_mask={"mode": "whitelist", "ids": [1, 4]},
            meta_vocab=None,
        )
        with pytest.raises(FeatureVocabError) as exc:
            load_dumped_mask(tmp_path)

        msg = str(exc.value)
        assert "ABSENT from the weights: none" in msg
        assert "EXCLUDED by the mask: 1 feature(s) [6]" in msg

    def test_vocab_buffer_row_count_mismatch_raises(self, tmp_path: Path) -> None:
        """``feature_vocab_ids`` and the table's row count must agree (PR-A contract)."""
        _write_checkpoint(tmp_path, vocab_ids=[1, 4, 6], meta_mask=None, meta_vocab=None)
        model_path = tmp_path / f"model_{STEP:06d}.pt"
        state = t.load(model_path, map_location="cpu", weights_only=True)
        state["encoder.feature_embed.weight"] = state["encoder.feature_embed.weight"][:2]
        t.save(state, model_path)

        with pytest.raises(FeatureVocabError, match="rows"):
            load_dumped_mask(tmp_path)

    def test_agreeing_meta_fingerprint_passes(self, tmp_path: Path) -> None:
        """The real training-path meta shape (mask + matching vocab mirror) must be accepted."""
        _write_checkpoint(
            tmp_path,
            vocab_ids=[1, 4, 6],
            meta_mask={"mode": "whitelist", "ids": [1, 4, 6]},
            meta_vocab=FeatureVocab(
                n_sae_features=N_SAE, original_ids=(1, 4, 6)
            ).metadata_dict(),
        )
        dumped = load_dumped_mask(tmp_path)
        assert dumped.mask == FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 4, 6))

    @pytest.mark.parametrize("missing", ["mode", "ids"])
    def test_meta_mask_missing_a_required_key_raises(self, tmp_path: Path, missing: str) -> None:
        block = {"mode": "whitelist", "ids": [1, 4, 6]}
        del block[missing]
        _write_checkpoint(tmp_path, vocab_ids=[1, 4, 6], meta_mask=block)

        with pytest.raises(FeatureMaskError, match="mode.*ids|meta_000005.json"):
            load_dumped_mask(tmp_path)

    def test_explicit_null_meta_mask_is_treated_as_no_mask(self, tmp_path: Path) -> None:
        """``"feature_mask_canonical": null`` (an unmasked run's meta) is not a corrupt block."""
        _write_checkpoint(tmp_path, vocab_ids=None, meta_mask=None)
        meta_path = tmp_path / f"meta_{STEP:06d}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["feature_mask_canonical"] = None
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        dumped = load_dumped_mask(tmp_path)
        assert dumped.mask is None
        assert dumped.source == "none"

    def test_same_size_different_ids_raises(self, tmp_path: Path) -> None:
        """The sizes agree, so ``load_state_dict`` would pass — only the ids catch this."""
        _write_checkpoint(
            tmp_path,
            vocab_ids=[1, 4, 6],
            meta_mask={"mode": "whitelist", "ids": [1, 4, 5]},
            meta_vocab=None,
        )
        with pytest.raises(FeatureVocabError, match="does not reproduce"):
            load_dumped_mask(tmp_path)

    def test_same_size_message_names_the_differing_ids(self, tmp_path: Path) -> None:
        """Counts alone are USELESS when V agrees — "allows 3, weights carry 3" reads as a
        contradiction. The message must name the features that actually disagree.

        This is the subtlest corruption the tool can see (matching sizes ⇒ the weights
        would load cleanly), so it is the one that most needs an actionable message.
        """
        _write_checkpoint(
            tmp_path,
            vocab_ids=[1, 4, 6],
            meta_mask={"mode": "whitelist", "ids": [1, 4, 5]},
            meta_vocab=None,
        )
        with pytest.raises(FeatureVocabError) as exc:
            load_dumped_mask(tmp_path)

        msg = str(exc.value)
        # 5 is allowed by the mask but has no row; 6 has a row but the mask excludes it.
        assert "ABSENT from the weights: 1 feature(s) [5]" in msg
        assert "EXCLUDED by the mask: 1 feature(s) [6]" in msg

    def test_large_id_difference_is_truncated_but_counted(self, tmp_path: Path) -> None:
        """A 65k dictionary can disagree on tens of thousands of ids — preview, never dump.

        The full count is always stated, so truncation hides nothing.
        """
        # Weights hold {0..39}; the mask allows {24..63}. Same V=40, but 24 ids disagree
        # in each direction.
        _write_checkpoint(
            tmp_path,
            n_sae=64,
            vocab_ids=list(range(40)),
            meta_mask={"mode": "whitelist", "ids": list(range(24, 64))},
            meta_vocab=None,
        )
        with pytest.raises(FeatureVocabError) as exc:
            load_dumped_mask(tmp_path)

        msg = str(exc.value)
        assert "ABSENT from the weights: 24 feature(s) [40, 41, 42, 43, 44, 45, 46, 47, ..." in msg
        assert "(+16 more)" in msg  # preview truncated ...
        assert "EXCLUDED by the mask: 24 feature(s)" in msg  # ... but the true count survives

    def test_mask_allowing_every_feature_cannot_have_compacted(self, tmp_path: Path) -> None:
        """A whitelist of ALL ids has an identity allowed universe — it compacts nothing."""
        _write_checkpoint(
            tmp_path,
            vocab_ids=[1, 4, 6],
            meta_mask={"mode": "whitelist", "ids": list(range(N_SAE))},
            meta_vocab=None,
        )
        with pytest.raises(FeatureVocabError, match="all|every|identity"):
            load_dumped_mask(tmp_path)

    def test_meta_vocab_fingerprint_disagreeing_with_buffers_raises(
        self, tmp_path: Path
    ) -> None:
        """meta's ``feature_vocab`` mirror is cross-checked against the load-bearing buffers."""
        _write_checkpoint(
            tmp_path,
            vocab_ids=[1, 4, 6],
            meta_mask={"mode": "whitelist", "ids": [1, 4, 6]},
            meta_vocab=FeatureVocab(
                n_sae_features=N_SAE, original_ids=(1, 4, 5)
            ).metadata_dict(),
        )
        with pytest.raises(FeatureVocabError, match="feature_vocab"):
            load_dumped_mask(tmp_path)

    def test_meta_claims_compacted_but_weights_are_not_raises(self, tmp_path: Path) -> None:
        _write_checkpoint(
            tmp_path,
            vocab_ids=None,
            meta_mask={"mode": "blacklist", "ids": [2]},
            meta_vocab=FeatureVocab(
                n_sae_features=N_SAE, original_ids=(1, 4, 6)
            ).metadata_dict(),
        )
        with pytest.raises(FeatureVocabError, match="feature_vocab"):
            load_dumped_mask(tmp_path)

    def test_meta_mask_id_out_of_range_for_the_checkpoint_raises(self, tmp_path: Path) -> None:
        _write_checkpoint(
            tmp_path,
            vocab_ids=[1, 4, 6],
            meta_mask={"mode": "whitelist", "ids": [1, 4, 99]},
            meta_vocab=None,
        )
        with pytest.raises(FeatureMaskError, match="99"):
            load_dumped_mask(tmp_path)

    def test_corrupt_meta_mask_block_names_the_meta_file(self, tmp_path: Path) -> None:
        _write_checkpoint(
            tmp_path, vocab_ids=[1, 4, 6], meta_mask={"mode": "nonsense", "ids": [1]}
        )
        with pytest.raises(FeatureMaskError, match="meta_000005.json"):
            load_dumped_mask(tmp_path)

    def test_meta_mask_with_non_int_ids_raises(self, tmp_path: Path) -> None:
        _write_checkpoint(
            tmp_path, vocab_ids=[1, 4, 6], meta_mask={"mode": "whitelist", "ids": ["1"]}
        )
        with pytest.raises(FeatureMaskError, match="meta_000005.json"):
            load_dumped_mask(tmp_path)

    def test_meta_mask_that_is_not_a_mapping_raises(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, vocab_ids=[1, 4, 6], meta_mask=None)
        meta_path = tmp_path / f"meta_{STEP:06d}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["feature_mask_canonical"] = [1, 4, 6]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        with pytest.raises(FeatureMaskError, match="mapping"):
            load_dumped_mask(tmp_path)

    def test_half_a_vocab_in_the_state_dict_raises(self, tmp_path: Path) -> None:
        """One vocab buffer without the other is a corrupt checkpoint (PR-A contract)."""
        _write_checkpoint(tmp_path, vocab_ids=[1, 4, 6], meta_mask=None, meta_vocab=None)
        model_path = tmp_path / f"model_{STEP:06d}.pt"
        state = t.load(model_path, map_location="cpu", weights_only=True)
        del state["encoder.feature_vocab_n_sae"]
        t.save(state, model_path)

        with pytest.raises(FeatureVocabError, match="feature_vocab_n_sae"):
            load_dumped_mask(tmp_path)


# ===========================================================================
# Compacted, NO meta mask — reconstruct a whitelist from the vocab
# ===========================================================================


def _write_under_its_real_leaf(root: Path, mask: FeatureMask, n_sae: int = N_SAE) -> Path:
    """Write a compacted checkpoint where PRODUCTION writes it: under append_mask_leaf().

    The leaf name — ``bl_<16hex>`` / ``wl_<16hex>`` — is the ONLY on-disk record of the
    mask's MODE once the meta is thinned, and the mode is what makes the recovered config
    address THIS checkpoint rather than a different one.
    """
    vocab = FeatureVocab.from_mask(mask, n_sae)
    assert vocab is not None, "the fixture must actually compact something"
    leaf = Path(append_mask_leaf(str(root), mask))
    _write_checkpoint(leaf, vocab_ids=list(vocab.original_ids), meta_mask=None, n_sae=n_sae)
    return leaf


class TestReconstructedFromVocab:
    """With no recorded mask, the mask is reconstructed from the weights — MODE included.

    The vocabulary alone gives the allowed UNIVERSE but not the mode, and that is not a
    cosmetic gap: ``FeatureMask.dirname()`` hashes ``(mode, ids)``. A whitelist of the vocab
    and a blacklist of its complement describe the SAME universe yet hash to DIFFERENT
    checkpoint leaves. Emitting the wrong mode hands the operator a config addressing a
    directory with no checkpoint in it — the recovery loop would not close.

    So the mode is recovered from the leaf directory's own name, and PROVEN by round-tripping
    the hash. These tests used to assert the opposite (that a whitelist of the universe was
    "exact, sufficient"); they were pinning the bug.
    """

    def test_a_BLACKLIST_run_recovers_a_BLACKLIST(self, tmp_path: Path) -> None:
        """The DEFAULT recipe is a blacklist. This is the case that was broken."""
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 7))
        leaf = _write_under_its_real_leaf(tmp_path, mask)

        dumped = load_dumped_mask(leaf)

        assert dumped.source == "vocab"
        assert dumped.mask == mask
        assert feature_mask_dirname(dumped.mask) == leaf.name

    def test_a_WHITELIST_run_recovers_a_WHITELIST(self, tmp_path: Path) -> None:
        mask = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 4, 6))
        leaf = _write_under_its_real_leaf(tmp_path, mask)

        dumped = load_dumped_mask(leaf)

        assert dumped.source == "vocab"
        assert dumped.mask == mask
        assert feature_mask_dirname(dumped.mask) == leaf.name

    def test_the_recovered_mask_ADDRESSES_the_checkpoint_it_came_from(
        self, tmp_path: Path
    ) -> None:
        """The property that makes the tool worth anything."""
        for mask in (
            FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 7)),
            FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 4, 6)),
        ):
            leaf = _write_under_its_real_leaf(tmp_path / mask.mode.value, mask)
            recovered = load_dumped_mask(leaf).mask
            assert feature_mask_dirname(recovered) == leaf.name

    def test_output_says_it_was_reconstructed(self, tmp_path: Path) -> None:
        """The operator must see the mask came from the WEIGHTS, not the meta."""
        leaf = _write_under_its_real_leaf(
            tmp_path, FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 7))
        )
        dumped = load_dumped_mask(leaf)

        for out in (render(dumped, FORMAT_YAML), render(dumped, FORMAT_TXT)):
            comments = "\n".join(ln for ln in out.splitlines() if ln.startswith("#"))
            assert "RECONSTRUCTED" in comments
            assert "feature_vocab_ids" in comments

    def test_the_reconstruction_round_trips_to_the_same_vocab(self, tmp_path: Path) -> None:
        """Pasting it must rebuild the same encoder."""
        leaf = _write_under_its_real_leaf(
            tmp_path, FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 7))
        )
        dumped = load_dumped_mask(leaf)
        assert _resolved_vocab_from_yaml(render(dumped, FORMAT_YAML), N_SAE) == dumped.vocab

    def test_txt_reconstruction_lists_the_MASK_ids(self, tmp_path: Path) -> None:
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 7))
        leaf = _write_under_its_real_leaf(tmp_path, mask)
        out = render(load_dumped_mask(leaf), FORMAT_TXT)

        ids_file = tmp_path / "ids.txt"
        ids_file.write_text(out, encoding="utf-8")
        assert parse_feature_ids_txt(ids_file) == [2, 7]

    def test_a_checkpoint_NOT_under_a_mask_leaf_REFUSES_to_guess(self, tmp_path: Path) -> None:
        """Without the meta AND without the leaf, the mode is genuinely unknowable. Emitting
        a plausible guess would silently hand back a config for a different checkpoint."""
        _write_checkpoint(tmp_path, vocab_ids=[1, 4, 6], meta_mask=None)

        with pytest.raises(FeatureMaskError, match="(?i)mode"):
            load_dumped_mask(tmp_path)


# ===========================================================================
# Legacy (uncompacted) checkpoints
# ===========================================================================


class TestLegacyUncompacted:
    def test_legacy_with_a_meta_mask_emits_it(self, tmp_path: Path) -> None:
        """No vocab buffers ⇒ nothing to cross-check against ⇒ emit the meta mask as-is."""
        _write_checkpoint(tmp_path, vocab_ids=None, meta_mask={"mode": "blacklist", "ids": [2, 7]})
        dumped = load_dumped_mask(tmp_path)
        out = render(dumped, FORMAT_YAML)

        assert dumped.vocab is None
        assert dumped.source == "meta"
        assert dumped.n_sae_features == N_SAE
        assert yaml.safe_load(out) == {"feature_mask": {"mode": "blacklist", "ids": [2, 7]}}

    def test_legacy_output_says_the_encoder_is_uncompacted(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, vocab_ids=None, meta_mask={"mode": "blacklist", "ids": [2]})
        out = render(load_dumped_mask(tmp_path), FORMAT_YAML)
        assert "UNCOMPACTED" in out

    def test_legacy_mask_is_not_cross_checked_against_a_vocab(self, tmp_path: Path) -> None:
        """A legacy checkpoint has a row for EVERY feature, so any mask is structurally
        safe and there is no vocabulary to contradict. It must not raise."""
        _write_checkpoint(
            tmp_path, vocab_ids=None, meta_mask={"mode": "whitelist", "ids": [0, 1, 2]}
        )
        dumped = load_dumped_mask(tmp_path)
        assert dumped.mask == FeatureMask(mode=FeatureMaskMode.whitelist, ids=(0, 1, 2))

    def test_no_mask_at_all_is_a_clean_exit(self, tmp_path: Path, capsys) -> None:
        """The GENUINE no-mask case: no meta mask, an uncompacted encoder, and — the part
        that makes it genuine — NO mask leaf anywhere on the path. ``tmp_path`` is
        ``/tmp/pytest-*/test_*``, so nothing on it looks like ``bl_``/``wl_<16hex>``."""
        _write_checkpoint(tmp_path, vocab_ids=None, meta_mask=None)
        dumped = load_dumped_mask(tmp_path)

        assert dumped.mask is None
        assert dumped.vocab is None
        assert dumped.source == "none"

        assert main([str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "No feature_mask" in out
        assert "feature_mask:" not in out  # nothing to paste — must not emit a block

    def test_no_mask_output_is_inert_in_both_formats(self, tmp_path: Path) -> None:
        """A redirected "no mask" dump must still be a VALID (empty) artifact, not junk."""
        _write_checkpoint(tmp_path, vocab_ids=None, meta_mask=None)
        dumped = load_dumped_mask(tmp_path)

        assert yaml.safe_load(render(dumped, FORMAT_YAML)) is None

        ids_file = tmp_path / "ids.txt"
        ids_file.write_text(render(dumped, FORMAT_TXT), encoding="utf-8")
        assert parse_feature_ids_txt(ids_file) == []


# ===========================================================================
# FINDING 3 — "no mask" is a CLAIM, and the path can contradict it
# ===========================================================================


class TestUncompactedCheckpointUnderAMaskLeaf:
    """An UNCOMPACTED encoder + a meta with no ``feature_mask_canonical`` used to be
    reported as "No feature_mask is configured" — unconditionally.

    But an uncompacted encoder is NOT evidence of "no mask". Pre-#695 checkpoints were
    trained WITH masks and are physically uncompacted, and SFT off a legacy base inherits
    that. The mask leaf on the checkpoint's own PATH is positive evidence that a mask WAS
    configured, and it directly contradicts the report.

    Worse, the report is not merely wrong, it is *harmful* in the same way FINDING 7's
    remedy was: "leave feature_mask unset (null)" sends the operator to a config whose
    ``checkpoint_dir`` has no ``bl_``/``wl_`` leaf — a different directory from the one the
    checkpoint they just pointed at lives in.

    Nothing can be recovered here: the leaf is a one-way hash (mode + ids), and an
    uncompacted table carries no id list. So there is no config to emit, and guessing one is
    the bug. Fail loudly and say exactly what is known.
    """

    def _uncompacted_under_leaf(self, root: Path, mask: FeatureMask) -> Path:
        """A legacy masked checkpoint, where production wrote it: full-width tables, no
        vocab buffers, no meta mask — under its real ``append_mask_leaf`` directory."""
        leaf = Path(append_mask_leaf(str(root), mask))
        _write_checkpoint(leaf, vocab_ids=None, meta_mask=None)
        return leaf

    def test_a_mask_leaf_on_the_path_is_NOT_reported_as_NO_MASK(self, tmp_path: Path) -> None:
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 7))
        leaf = self._uncompacted_under_leaf(tmp_path, mask)

        with pytest.raises(FeatureMaskError) as exc:
            load_dumped_mask(leaf)

        msg = str(exc.value)
        assert leaf.name in msg, "must quote the leaf that contradicts the 'no mask' story"
        assert "feature_mask_canonical" in msg, "must say what the meta is missing"

    def test_it_says_the_ids_are_UNRECOVERABLE_rather_than_guessing(
        self, tmp_path: Path
    ) -> None:
        """The leaf hashes (mode, ids) one-way and the weights are full width, so neither
        source holds the id list. There is genuinely nothing to emit."""
        leaf = self._uncompacted_under_leaf(
            tmp_path, FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 4, 6))
        )
        with pytest.raises(FeatureMaskError) as exc:
            load_dumped_mask(leaf)

        low = str(exc.value).lower()
        assert "uncompacted" in low, "must state the encoder shape it actually found"
        assert re.search(r"cannot|unrecoverab|not recoverab|no way", low)

    def test_a_WHITELIST_leaf_is_caught_too(self, tmp_path: Path) -> None:
        leaf = self._uncompacted_under_leaf(
            tmp_path, FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 4, 6))
        )
        with pytest.raises(FeatureMaskError):
            load_dumped_mask(leaf)

    def test_a_leaf_on_an_ANCESTOR_is_caught_too(self, tmp_path: Path) -> None:
        """Production nests the gradient/SFT phase BELOW the mask leaf, so the leaf is
        usually not the basename. Matching only the basename would miss every SFT
        checkpoint — exactly the mistake ``_mask_from_vocab_and_leaf`` already had to fix."""
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 7))
        leaf = Path(append_mask_leaf(str(tmp_path), mask))
        nested = leaf / "chat_sft_conditioned_checkpoints"
        _write_checkpoint(nested, vocab_ids=None, meta_mask=None)

        with pytest.raises(FeatureMaskError) as exc:
            load_dumped_mask(nested)
        assert leaf.name in str(exc.value)

    def test_a_legacy_checkpoint_WITH_a_meta_mask_under_its_leaf_still_WORKS(
        self, tmp_path: Path
    ) -> None:
        """The REAL pre-#695 production state, and it must keep working: the meta records
        the mask, so nothing is missing and nothing is inconsistent. The new guard must fire
        only when the meta records NOTHING."""
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 7))
        leaf = Path(append_mask_leaf(str(tmp_path), mask))
        _write_checkpoint(leaf, vocab_ids=None, meta_mask={"mode": "blacklist", "ids": [2, 7]})

        dumped = load_dumped_mask(leaf)

        assert dumped.source == "meta"
        assert dumped.mask == mask
        assert dumped.vocab is None  # still uncompacted — that is fine and expected
        assert "UNCOMPACTED" in render(dumped, FORMAT_YAML)

    def test_the_GENUINE_no_mask_case_is_UNTOUCHED(self, tmp_path: Path) -> None:
        """The regression guard for the fix: no leaf on the path ⇒ still a clean 'no mask'."""
        _write_checkpoint(tmp_path, vocab_ids=None, meta_mask=None)
        dumped = load_dumped_mask(tmp_path)
        assert dumped.source == "none"
        assert dumped.mask is None
        assert "No feature_mask" in render(dumped, FORMAT_YAML)

    def test_a_directory_merely_STARTING_with_bl_is_not_a_mask_leaf(
        self, tmp_path: Path
    ) -> None:
        """The detector keys on the real ``{bl|wl}_{16hex}`` shape, not on a prefix. A
        checkpoint under ``bl_experiments/`` has no mask leaf and must stay a clean exit."""
        d = tmp_path / "bl_experiments" / "wl_notahash"
        _write_checkpoint(d, vocab_ids=None, meta_mask=None)
        assert load_dumped_mask(d).source == "none"


# ===========================================================================
# Step resolution + CLI surface
# ===========================================================================


class TestStepResolution:
    def test_defaults_to_the_latest_step(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, step=5, vocab_ids=[1], meta_mask={"mode": "whitelist", "ids": [1]})
        _write_checkpoint(
            tmp_path, step=10, vocab_ids=[2, 3], meta_mask={"mode": "whitelist", "ids": [2, 3]}
        )
        dumped = load_dumped_mask(tmp_path)

        assert dumped.step == 10
        assert dumped.mask == FeatureMask(mode=FeatureMaskMode.whitelist, ids=(2, 3))

    def test_explicit_step_selects_that_step(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, step=5, vocab_ids=[1], meta_mask={"mode": "whitelist", "ids": [1]})
        _write_checkpoint(
            tmp_path, step=10, vocab_ids=[2, 3], meta_mask={"mode": "whitelist", "ids": [2, 3]}
        )
        dumped = load_dumped_mask(tmp_path, step=5)

        assert dumped.step == 5
        assert dumped.mask == FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1,))

    def test_explicit_step_via_cli(self, tmp_path: Path, capsys) -> None:
        _write_checkpoint(tmp_path, step=5, vocab_ids=[1], meta_mask={"mode": "whitelist", "ids": [1]})
        _write_checkpoint(
            tmp_path, step=10, vocab_ids=[2, 3], meta_mask={"mode": "whitelist", "ids": [2, 3]}
        )

        assert main([str(tmp_path), "--step", "5"]) == 0
        assert yaml.safe_load(capsys.readouterr().out) == {
            "feature_mask": {"mode": "whitelist", "ids": [1]}
        }

    def test_step_zero_is_not_confused_with_the_latest(self, tmp_path: Path) -> None:
        """``--step 0`` is FALSY. A ``if step:`` fallback would silently dump the latest
        checkpoint instead — handing the operator the wrong mask without a word."""
        _write_checkpoint(tmp_path, step=0, vocab_ids=[1], meta_mask={"mode": "whitelist", "ids": [1]})
        _write_checkpoint(
            tmp_path, step=10, vocab_ids=[2, 3], meta_mask={"mode": "whitelist", "ids": [2, 3]}
        )
        dumped = load_dumped_mask(tmp_path, step=0)

        assert dumped.step == 0
        assert dumped.mask == FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1,))

    def test_missing_step_raises(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, step=5, vocab_ids=[1], meta_mask={"mode": "whitelist", "ids": [1]})
        with pytest.raises(FileNotFoundError, match="model_000007.pt"):
            load_dumped_mask(tmp_path, step=7)

    def test_missing_checkpoint_dir_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope"
        with pytest.raises(FileNotFoundError, match="nope"):
            load_dumped_mask(missing)

    def test_empty_checkpoint_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_dumped_mask(tmp_path)

    def test_missing_meta_raises(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, vocab_ids=[1, 4, 6], write_meta=False)
        with pytest.raises(FileNotFoundError, match="meta_000005.json"):
            load_dumped_mask(tmp_path)


class TestCli:
    def test_yaml_is_the_default_format(self, tmp_path: Path, capsys) -> None:
        _write_checkpoint(
            tmp_path,
            vocab_ids=_blacklist_vocab(N_SAE, [2, 7]),
            meta_mask={"mode": "blacklist", "ids": [2, 7]},
        )
        assert main([str(tmp_path)]) == 0
        assert yaml.safe_load(capsys.readouterr().out) == {
            "feature_mask": {"mode": "blacklist", "ids": [2, 7]}
        }

    def test_txt_format_flag(self, tmp_path: Path, capsys, tmp_path_factory) -> None:
        _write_checkpoint(
            tmp_path,
            vocab_ids=_blacklist_vocab(N_SAE, [2, 7]),
            meta_mask={"mode": "blacklist", "ids": [2, 7]},
        )
        assert main([str(tmp_path), "--format", "txt"]) == 0

        ids_file = tmp_path_factory.mktemp("txt") / "ids.txt"
        ids_file.write_text(capsys.readouterr().out, encoding="utf-8")
        assert parse_feature_ids_txt(ids_file) == [2, 7]

    def test_unknown_format_is_rejected(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, vocab_ids=[1], meta_mask={"mode": "whitelist", "ids": [1]})
        with pytest.raises(SystemExit) as exc:
            main([str(tmp_path), "--format", "json"])
        assert exc.value.code == 2

    def test_render_rejects_an_unknown_format(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, vocab_ids=[1], meta_mask={"mode": "whitelist", "ids": [1]})
        with pytest.raises(ValueError, match="json"):
            render(load_dumped_mask(tmp_path), "json")

    def test_cli_failure_propagates_loudly(self, tmp_path: Path) -> None:
        """No silent fallback: a meta/weights disagreement must NOT print a mask."""
        _write_checkpoint(
            tmp_path,
            vocab_ids=[1, 4, 6],
            meta_mask={"mode": "whitelist", "ids": [1, 4]},
            meta_vocab=None,
        )
        with pytest.raises(FeatureVocabError):
            main([str(tmp_path)])

    def test_stdout_carries_the_artifact_alone(self, tmp_path: Path, capsys) -> None:
        """``dump_feature_mask ... > mask.yaml`` must produce a VALID yaml file.

        Log lines go to stderr (loguru); stdout is the artifact and nothing else. If a
        status line leaked into stdout the redirect would silently produce an unparseable
        config — the exact failure this tool exists to prevent.
        """
        _write_checkpoint(
            tmp_path,
            vocab_ids=_blacklist_vocab(N_SAE, [2, 7]),
            meta_mask={"mode": "blacklist", "ids": [2, 7]},
        )
        assert main([str(tmp_path)]) == 0

        out = capsys.readouterr().out
        assert "INFO" not in out
        assert all(ln.startswith("#") or ln[:1] in ("f", " ") for ln in out.splitlines())
        assert yaml.safe_load(out) == {"feature_mask": {"mode": "blacklist", "ids": [2, 7]}}

    def test_render_defaults_to_yaml(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, vocab_ids=[1], meta_mask={"mode": "whitelist", "ids": [1]})
        dumped = load_dumped_mask(tmp_path)
        assert render(dumped) == render(dumped, FORMAT_YAML)

    def test_output_is_deterministic(self, tmp_path: Path) -> None:
        _write_checkpoint(
            tmp_path,
            vocab_ids=_blacklist_vocab(N_SAE, [2, 7]),
            meta_mask={"mode": "blacklist", "ids": [2, 7]},
        )
        first = render(load_dumped_mask(tmp_path), FORMAT_YAML)
        second = render(load_dumped_mask(tmp_path), FORMAT_YAML)
        assert first == second

    def test_provenance_names_the_rerun_command(self, tmp_path: Path) -> None:
        """A dumped file must record how to regenerate itself, pinned to an EXPLICIT step —
        so re-running it later against a further-trained dir cannot silently change."""
        _write_checkpoint(tmp_path, vocab_ids=[1], meta_mask={"mode": "whitelist", "ids": [1]})
        out = render(load_dumped_mask(tmp_path), FORMAT_YAML)

        assert (
            f"scripts.analysis.dump_feature_mask {tmp_path} --step {STEP}" in out
        )


class TestOriginalWidthIsPreserved:
    def test_n_sae_features_is_the_original_width_not_v(self, tmp_path: Path) -> None:
        """§4.2: ``n_sae_features`` keeps its ORIGINAL-space meaning under compaction.

        Reporting V here would be a silent corruption: every downstream bounds-check
        (``validate_feature_dict``, ``eval_adapter``, ...) reads it to validate ORIGINAL
        feature ids, and a compacted checkpoint's row count is not that number.
        """
        _write_checkpoint(tmp_path, vocab_ids=[1, 4, 6], meta_mask={"mode": "whitelist", "ids": [1, 4, 6]})
        dumped = load_dumped_mask(tmp_path)

        assert dumped.n_sae_features == N_SAE  # 8, the SAE width
        assert dumped.vocab is not None
        assert dumped.vocab.size == 3  # V — the compact row count
        assert dumped.vocab.n_sae_features == N_SAE

    def test_emitted_ids_are_original_space(self, tmp_path: Path) -> None:
        """The emitted mask must speak ORIGINAL SAE ids, never compact row indices.

        The vocab is {1, 4, 6}; the compact rows are {0, 1, 2}. Emitting the row indices
        would be a mask that silently means entirely different features.
        """
        leaf = _write_under_its_real_leaf(
            tmp_path, FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 4, 6))
        )
        out = render(load_dumped_mask(leaf), FORMAT_YAML)

        assert yaml.safe_load(out)["feature_mask"]["ids"] == [1, 4, 6]


# ===========================================================================
# The weights are READ, not MATERIALIZED (review finding 14)
# ===========================================================================


class TestTheCheckpointIsNotMaterialized:
    """This tool needs exactly three numbers out of a multi-GB file.

    ``encoder.feature_vocab_ids``, ``encoder.feature_vocab_n_sae`` and
    ``encoder.feature_embed.weight.shape[0]`` — a few hundred bytes. Loading eagerly
    deserialises every parameter of a ~660M-param checkpoint (GBs of host RAM, over
    ~5-9 MB/s remote mounts) to read them. The module docstring even markets the meta
    mirror as existing "so tooling can compare a checkpoint's vocabulary without loading
    the weights"; ``mmap=True`` is what makes that true of the weights path too.

    Measured on a 256 MB checkpoint: 245.8 MB peak RSS eager vs 1.6 MB mmap'd, for
    byte-identical values.
    """

    def test_read_state_dict_memory_maps_the_file(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, vocab_ids=[1, 4, 6], meta_mask=None)

        with patch.object(t, "load", wraps=t.load) as spy:
            state = _read_state_dict(tmp_path, STEP)

        assert spy.call_count == 1
        kwargs = spy.call_args.kwargs
        assert kwargs["mmap"] is True, (
            "_read_state_dict deserialises the whole checkpoint to read two small "
            "buffers; it must pass mmap=True"
        )
        assert kwargs["weights_only"] is True  # unchanged: still no arbitrary unpickling
        assert kwargs["map_location"] == "cpu"
        # The spy WRAPS the real torch.load, so this is a genuine read off disk.
        assert state["encoder.feature_vocab_ids"].tolist() == [1, 4, 6]

    def test_load_dumped_mask_never_loads_the_weights_eagerly(self, tmp_path: Path) -> None:
        """The public entry point, not just the private helper."""
        _write_checkpoint(
            tmp_path,
            vocab_ids=_blacklist_vocab(N_SAE, [2, 7]),
            meta_mask={"mode": "blacklist", "ids": [2, 7]},
        )

        with patch.object(t, "load", wraps=t.load) as spy:
            dumped = load_dumped_mask(tmp_path)

        assert spy.call_args_list, "load_dumped_mask did not read the weights at all"
        for call in spy.call_args_list:
            assert call.kwargs.get("mmap") is True
        assert dumped.mask == FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 7))

    def test_the_mmapped_load_returns_exactly_what_an_eager_load_would(
        self, tmp_path: Path
    ) -> None:
        """``mmap=True`` is a laziness change, NOT a semantics change.

        Same keys, same dtypes, same values, bit for bit — otherwise the optimisation
        would be reinterpreting the very buffers the tool exists to report.
        """
        _write_checkpoint(tmp_path, vocab_ids=[1, 4, 6], meta_mask=None)

        mapped = _read_state_dict(tmp_path, STEP)
        eager = t.load(
            tmp_path / f"model_{STEP:06d}.pt", map_location="cpu", weights_only=True
        )

        assert set(mapped) == set(eager)
        for key in eager:
            assert mapped[key].dtype == eager[key].dtype
            assert t.equal(mapped[key], eager[key]), key

    def test_a_legacy_uncompacted_checkpoint_is_mmapped_too(self, tmp_path: Path) -> None:
        """The legacy path reads ``feature_embed.weight.shape[0]`` — a SHAPE, which needs
        no page of the tensor faulted in at all."""
        _write_checkpoint(
            tmp_path, vocab_ids=None, meta_mask={"mode": "blacklist", "ids": [2, 7]}
        )

        with patch.object(t, "load", wraps=t.load) as spy:
            dumped = load_dumped_mask(tmp_path)

        assert all(c.kwargs.get("mmap") is True for c in spy.call_args_list)
        assert dumped.n_sae_features == N_SAE
        assert dumped.vocab is None


# ===========================================================================
# The mask leaf is an ANCESTOR — production NESTS below it
# ===========================================================================

# Nothing in production stops at the mask leaf. ``conditional_encdec_train.py`` composes
#     append_gradient_leaf(append_mask_leaf(.../s{seg}/{fs}, mask), fs)
# and ``conditional_encdec_sft.py`` composes
#     _build_sft_checkpoint_dir(append_mask_leaf(pretrain_dir, mask), dataset)
#         -> .../{fs}/bl_<16hex>/{dataset}_sft_conditioned_checkpoints
# so for EVERY SFT checkpoint — and every gradient-ranked pretrain — the ``bl_``/``wl_``
# leaf is a PARENT (or, for SFT off a gradient-ranked base, a GRANDPARENT) of the directory
# that actually holds ``model_*.pt``. Matching only ``checkpoint_dir.name`` dead-ended with
# "cannot determine the MODE" on exactly the checkpoints #695's guards send operators here
# to recover. The depth is not fixed, so the search cannot be depth-limited either.
SFT_LEAF = "smoltalk_sft_conditioned_checkpoints"
GRAD_LEAF = "grad_topk_mf8"


def _write_under_nested_leaf(
    root: Path, mask: FeatureMask, *below: str, n_sae: int = N_SAE
) -> Path:
    """A compacted checkpoint nested BELOW its mask leaf, the way production writes one."""
    vocab = FeatureVocab.from_mask(mask, n_sae)
    assert vocab is not None, "the fixture must actually compact something"
    ckpt = Path(append_mask_leaf(str(root), mask)).joinpath(*below)
    _write_checkpoint(ckpt, vocab_ids=list(vocab.original_ids), meta_mask=None, n_sae=n_sae)
    return ckpt


class TestTheMaskLeafIsAnAncestor:
    """The mode is recovered from the mask leaf wherever it sits ON THE PATH — and only
    from there. A hash match is a 64-bit commitment to ``(mode, ids)``, so finding one
    PROVES the mode; finding none means it is genuinely unknowable and we refuse."""

    def test_an_SFT_checkpoint_recovers_its_BLACKLIST(self, tmp_path: Path) -> None:
        """THE case the #695 error messages send operators to. The directory the operator
        passes is the SFT leaf; the mask leaf is its parent."""
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 7))
        ckpt = _write_under_nested_leaf(tmp_path / "d12" / "s128" / "t50", mask, SFT_LEAF)
        assert ckpt.name == SFT_LEAF, "the fixture must NOT be a flat mask-leaf directory"

        dumped = load_dumped_mask(ckpt)

        assert dumped.source == "vocab"
        assert dumped.mask == mask
        assert dumped.mask.mode == FeatureMaskMode.blacklist

    def test_a_gradient_ranked_pretrain_checkpoint_recovers_its_mask(
        self, tmp_path: Path
    ) -> None:
        """``append_gradient_leaf`` nests BELOW ``append_mask_leaf`` in the train script, so
        even a PRETRAIN checkpoint need not be named ``bl_``/``wl_``."""
        mask = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 4, 6))
        ckpt = _write_under_nested_leaf(tmp_path / "d12" / "s128", mask, GRAD_LEAF)

        assert load_dumped_mask(ckpt).mask == mask

    def test_the_leaf_may_be_a_GRANDparent(self, tmp_path: Path) -> None:
        """SFT off a gradient-ranked pretrain (the ``base_checkpoint_dir`` override branch):
        the mask leaf is TWO levels up. Depth is not fixed."""
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 7))
        ckpt = _write_under_nested_leaf(tmp_path, mask, GRAD_LEAF, SFT_LEAF)

        assert load_dumped_mask(ckpt).mask == mask

    def test_the_recovered_mask_ADDRESSES_the_leaf_it_was_found_under(
        self, tmp_path: Path
    ) -> None:
        """The property that makes the tool worth anything, restated for nesting: the
        emitted mask must hash to the leaf this checkpoint actually lives under."""
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 7))
        ckpt = _write_under_nested_leaf(tmp_path, mask, SFT_LEAF)

        recovered = load_dumped_mask(ckpt).mask

        assert feature_mask_dirname(recovered) == ckpt.parent.name

    def test_a_trailing_slash_does_not_hide_the_leaf(self, tmp_path: Path) -> None:
        """Shell tab-completion appends one."""
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 7))
        ckpt = _write_under_nested_leaf(tmp_path, mask, SFT_LEAF)

        assert load_dumped_mask(f"{ckpt}/").mask == mask

    def test_a_SYMLINK_into_the_leaf_resolves_to_it(self, tmp_path: Path) -> None:
        """Operators symlink checkpoints onto scratch disks. The LINK's ancestors carry no
        mask leaf; its TARGET's do, and the target is where the weights actually are."""
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 7))
        ckpt = _write_under_nested_leaf(tmp_path / "real", mask, SFT_LEAF)
        link = tmp_path / "latest"
        link.symlink_to(ckpt, target_is_directory=True)

        assert load_dumped_mask(link).mask == mask

    def test_a_checkpoint_COPIED_OUT_of_its_leaf_still_REFUSES_to_guess(
        self, tmp_path: Path
    ) -> None:
        """Searching ancestors must not become a licence to guess. A copy carries no mask
        leaf anywhere above it, so the mode is genuinely gone — and guessing it is the very
        bug this machinery exists to prevent."""
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 7))
        ckpt = _write_under_nested_leaf(tmp_path / "real", mask, SFT_LEAF)
        copied = tmp_path / "copied_elsewhere"
        shutil.copytree(ckpt, copied)

        with pytest.raises(FeatureMaskError, match="(?i)mode"):
            load_dumped_mask(copied)

    def test_BOTH_candidate_leaves_on_one_path_is_AMBIGUOUS_and_raises(
        self, tmp_path: Path
    ) -> None:
        """A blacklist leaf AND a whitelist leaf for the same universe cannot both be the
        mode this checkpoint was trained with. Production never writes that, so it means the
        tree was assembled by hand — refuse rather than let ancestor ORDER pick a winner."""
        whitelist = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 4, 6))
        blacklist = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(0, 2, 3, 5, 7))
        ckpt = (
            tmp_path
            / feature_mask_dirname(blacklist)
            / feature_mask_dirname(whitelist)
            / SFT_LEAF
        )
        _write_checkpoint(ckpt, vocab_ids=[1, 4, 6], meta_mask=None)

        with pytest.raises(FeatureMaskError, match="(?i)ambiguous"):
            load_dumped_mask(ckpt)

    def test_the_refusal_names_BOTH_candidate_leaves(self, tmp_path: Path) -> None:
        """The operator is told exactly which two directories could have held it, so they
        can go and look."""
        _write_checkpoint(tmp_path, vocab_ids=[1, 4, 6], meta_mask=None)

        with pytest.raises(FeatureMaskError) as exc:
            load_dumped_mask(tmp_path)

        message = str(exc.value)
        for candidate in (
            FeatureMask(mode=FeatureMaskMode.blacklist, ids=(0, 2, 3, 5, 7)),
            FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 4, 6)),
        ):
            assert feature_mask_dirname(candidate) in message


# ===========================================================================
# The provenance header must describe what was ACTUALLY emitted
# ===========================================================================


class TestProvenanceTellsTheTruth:
    """The ``#`` header is the part the operator READS before pasting. It hard-coded
    "Emitted as an explicit whitelist of the checkpoint's N vocab ids" — printed directly
    above a body that (correctly) says ``mode: blacklist``. The artifact contradicted
    itself, and it contradicted itself about the one field that decides which directory the
    pasted config addresses."""

    @pytest.mark.parametrize(
        ("mask", "the_other_mode"),
        [
            (FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 7)), "whitelist"),
            (FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 4, 6)), "blacklist"),
        ],
    )
    def test_the_header_agrees_with_the_body(
        self, tmp_path: Path, mask: FeatureMask, the_other_mode: str
    ) -> None:
        leaf = _write_under_its_real_leaf(tmp_path, mask)
        out = render(load_dumped_mask(leaf), FORMAT_YAML)

        header = "\n".join(ln for ln in out.splitlines() if ln.startswith("#"))
        body = yaml.safe_load(out)["feature_mask"]

        assert body["mode"] == mask.mode.value
        assert mask.mode.value in header
        assert the_other_mode not in header, (
            f"the header says {the_other_mode!r} above a body that says "
            f"mode: {body['mode']} — the artifact contradicts itself"
        )

    def test_the_header_says_the_MODE_came_from_the_mask_LEAF(self, tmp_path: Path) -> None:
        """Not decoration: it is the operator's only way to check the claim, and the leaf
        name is a hash of ``(mode, ids)`` — so quoting it PROVES the config addresses this
        checkpoint."""
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 7))
        ckpt = _write_under_nested_leaf(tmp_path, mask, SFT_LEAF)

        out = render(load_dumped_mask(ckpt), FORMAT_YAML)
        header = "\n".join(ln for ln in out.splitlines() if ln.startswith("#"))

        assert feature_mask_dirname(mask) in header

    def test_the_txt_header_agrees_with_the_mode_it_tells_you_to_paste(
        self, tmp_path: Path
    ) -> None:
        """``txt`` carries ids ONLY, so its header IS the mode. If it lies, the ids are
        applied under the wrong mode — inverting the mask."""
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 7))
        leaf = _write_under_its_real_leaf(tmp_path, mask)

        out = render(load_dumped_mask(leaf), FORMAT_TXT)

        assert "mode: blacklist" in out
        assert "whitelist" not in out

    def test_the_META_sourced_header_is_untouched(self, tmp_path: Path) -> None:
        """The recorded-mask path never made the false claim; it must keep saying where it
        read the mask from."""
        _write_checkpoint(
            tmp_path,
            vocab_ids=[1, 4, 6],
            meta_mask={"mode": "whitelist", "ids": [1, 4, 6]},
        )

        out = render(load_dumped_mask(tmp_path), FORMAT_YAML)

        assert "# source: meta_000005.json -> feature_mask_canonical" in out
        assert "RECONSTRUCTED" not in out


# ===========================================================================
# torch.compile's ``_orig_mod.`` key prefix
# ===========================================================================


class TestTorchCompileSavedCheckpoints:
    """``torch.compile`` wraps the module, so every key it saves is prefixed ``_orig_mod.``.

    All three sibling loaders strip it — ``nanochat/checkpoint_manager.py`` (both
    ``build_model`` and ``load_checkpoint``) and
    ``scripts/preprocess/convert_decoder_to_encdec.py``. This one did not, so
    ``feature_vocab_from_state_dict`` saw no ``encoder.feature_embed.weight`` and rejected a
    perfectly good compiled encdec checkpoint with "is this an encoder-decoder checkpoint?".
    """

    @staticmethod
    def _compile_prefix_the_weights(checkpoint_dir: Path, step: int = STEP) -> None:
        model_path = checkpoint_dir / f"model_{step:06d}.pt"
        state = t.load(model_path, map_location="cpu", weights_only=True)
        t.save({f"_orig_mod.{k}": v for k, v in state.items()}, model_path)

    def test_read_state_dict_strips_the_prefix(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, vocab_ids=[1, 4, 6], meta_mask=None)
        self._compile_prefix_the_weights(tmp_path)

        state = _read_state_dict(tmp_path, STEP)

        assert not any(key.startswith("_orig_mod.") for key in state)
        assert "encoder.feature_embed.weight" in state
        assert "encoder.feature_vocab_ids" in state

    def test_a_compile_saved_COMPACTED_checkpoint_recovers_its_mask(
        self, tmp_path: Path
    ) -> None:
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 7))
        leaf = _write_under_its_real_leaf(tmp_path, mask)
        self._compile_prefix_the_weights(leaf)

        dumped = load_dumped_mask(leaf)

        assert dumped.mask == mask
        assert dumped.vocab is not None
        assert dumped.vocab.original_ids == (0, 1, 3, 4, 5, 6)

    def test_a_compile_saved_checkpoint_with_a_META_mask_recovers_it(
        self, tmp_path: Path
    ) -> None:
        """The meta path reads the weights too — to cross-check the mask against them."""
        _write_checkpoint(
            tmp_path,
            vocab_ids=[1, 4, 6],
            meta_mask={"mode": "whitelist", "ids": [1, 4, 6]},
        )
        self._compile_prefix_the_weights(tmp_path)

        dumped = load_dumped_mask(tmp_path)

        assert dumped.mask == FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 4, 6))
        assert dumped.vocab is not None
        assert dumped.vocab.size == 3

    def test_stripping_the_prefix_does_not_defeat_the_mmap(self, tmp_path: Path) -> None:
        """The strip rebinds KEYS; it must not copy the storages the lazy read exists for."""
        _write_checkpoint(tmp_path, vocab_ids=[1, 4, 6], meta_mask=None)
        self._compile_prefix_the_weights(tmp_path)

        with patch.object(t, "load", wraps=t.load) as spy:
            state = _read_state_dict(tmp_path, STEP)

        assert all(c.kwargs.get("mmap") is True for c in spy.call_args_list)
        assert t.equal(
            state["encoder.feature_vocab_ids"], t.tensor([1, 4, 6], dtype=t.int64)
        )
