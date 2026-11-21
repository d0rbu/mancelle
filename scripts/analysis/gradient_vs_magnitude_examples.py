"""Qualitative side-by-side of MAGNITUDE vs GRADIENT-ATTRIBUTION feature
selection, emitted as YAML for eyeballing.

For a handful of example segments it writes, per segment: the text before it
(``prompt``), the segment text (``segment``), the text right after it
(``future`` — what gradient attribution attributes to), and the segment's SAE
features ranked several ways. Three rank the current segment's stored features:
by activation **magnitude**, by activation **percentile** (the real
``top_k_percentile`` selection criterion, present iff a pct overlay exists), and
by the stored **gradient score** (``percentile(a) · max over rows of
ReLU(-dL/df)``: the feature's activation percentile times its loss-helpfulness
sensitivity, pushed through the decoder — exactly what the producer bakes into
the overlay). Three more (percentile-only, present iff a pct overlay exists)
add: **following-segment** percentile (an oracle — the *next* segment's own top
features, "what is actually active next"), **delta-vs-previous** (current
features by percentile increase over the immediately previous segment), and
**delta-vs-all-previous** (increase over the running-max percentile across every
earlier segment in the doc). Each feature line shows ALL signals (``pct=``,
``Δpct=`` on the delta lists, ``mag=``, ``grad=``) and the Neuronpedia
description; the link is appended only with ``--show-urls``.

Every list ranks over the SAME candidate set the real ``top_k_percentile``
selection uses: the activation-FREQUENCY blacklist the selection pipeline
materialized (``feature_dir/blacklist/freq_min..._max....txt``, written by
``precompute_feature_blacklist`` with that run's bounds) is READ FROM DISK and
applied before ranking — there is no unfiltered codepath, and the bounds are
never assumed. ``by_gradient`` reads the stored ``pct_relu_neg_max`` overlay
(percentile × sensitivity), guarded at load so a stale bare-sensitivity overlay
is refused.

Pure reuse: base shard via :func:`steering.feature_io.load_features`
(``feature_mags`` co-aligned 1:1 with ``feature_ids``), grad overlay via
:mod:`steering.overlay_io`, source text via
:func:`steering.parquet_io.read_parquet_texts` sliced by the base shard's
``segment_char_starts/ends`` (no re-tokenization), model/SAE identity read
straight from the grad overlay manifest, Neuronpedia ids via
:func:`steering.sae_resolver.resolve_neuronpedia_ids`.

Usage::

    uv run python -m scripts.analysis.gradient_vs_magnitude_examples \
        --feature-dir out/data/features/s128/l13/a_res/p_max_pool/f16k/m_gemma-3-1b-pt/base

Only ``--feature-dir`` is required; the grad overlay (and the model/SAE
identity for the links) is auto-resolved from ``<feature_dir>/grad/<hash>/``.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch as t
import yaml
from loguru import logger

from errors import ManualFeatureSelectionError
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.freq_blacklist import BLACKLIST_SUBDIR
from steering.eval.neuronpedia_client import NeuronpediaClient
from steering.feature_io import glob_features, load_features, shard_stem_from_path
from steering.feature_paths import resolve_data_dir
from steering.gradient_overlay import (
    GRAD_OFFSETS_KEY,
    GRAD_SCORE_REDUCTION,
    GRAD_SCORES_KEY,
)
from steering.manual_feature_ids import parse_feature_ids_txt
from steering.overlay_io import (
    OverlayKind,
    load_overlay_shard,
    overlay_shard_path,
    read_manifest,
)
from steering.parquet_io import read_parquet_texts
from steering.sae_resolver import (
    ActivationType,
    L0Tier,
    SAERelease,
    SAEWidth,
    resolve_neuronpedia_ids,
)


def _resolve_grad_dir(feature_dir: Path, grad_dir: Path | None) -> Path:
    """The base-aligned grad overlay dir. Auto-pick when there is exactly one."""
    if grad_dir is not None:
        if not (grad_dir / "manifest.json").is_file():
            raise SystemExit(f"no manifest.json under {grad_dir}")
        return grad_dir
    candidates = sorted(p for p in (feature_dir / "grad").glob("*") if p.is_dir())
    if not candidates:
        raise SystemExit(
            f"no grad overlay under {feature_dir / 'grad'} — run "
            f"scripts.preprocess.precompute_gradient_attribution first."
        )
    if len(candidates) > 1:
        raise SystemExit(
            f"multiple grad overlays under {feature_dir / 'grad'}; pass one "
            f"with --grad-dir:\n  " + "\n  ".join(str(c) for c in candidates)
        )
    return candidates[0]


def _neuronpedia(descriptor: dict, cache_path: Path, fids: list[int], *, fetch: bool):
    """Resolve ``(url_of, desc_by_fid)`` for the displayed features.

    ``url_of(fid)`` is a browsable Neuronpedia URL (or None if the identity
    can't be mapped). ``desc_by_fid`` maps feature id -> human-readable
    description, sourced from the SHARED Neuronpedia cache the completion-TUI
    populates (``<feature_dir>/neuronpedia_cache.yaml``): offline by default
    (``cache_only``), or fetched + cached when ``fetch`` is set. Features
    Neuronpedia doesn't publish (e.g. some gemma-scope-2 SAEs) are simply
    absent. Best-effort throughout — never fatal."""
    try:
        l0_raw = descriptor.get("sae_l0")
        l0 = L0Tier(str(l0_raw)) if l0_raw not in (None, "", "none") else None
        model_id, source = resolve_neuronpedia_ids(
            descriptor["model"],
            SAERelease(descriptor["sae_release"]),
            ActivationType(descriptor["activation_type"]),
            int(descriptor["layer"]),
            SAEWidth(descriptor["sae_width"]),
            l0,
        )
    except Exception as exc:  # noqa: BLE001 - links/labels are a convenience, not a contract
        logger.warning(f"Neuronpedia disabled ({type(exc).__name__}: {exc})")
        return None, {}

    # resolve_neuronpedia_ids returns the repo's INTERNAL canonical identity
    # (round-trips with parse_neuronpedia_source). Neuronpedia's PUBLIC naming
    # drops the -pt/-it model suffix and the GemmaScope-2 L0 tier, e.g.
    # gemma-3-1b/13-gemmascope-2-res-16k (not gemma-3-1b-pt/...-16k-medium).
    # This fixes both the browsable link AND the description fetch below.
    for _suffix in ("-pt", "-it"):
        if model_id.endswith(_suffix):
            model_id = model_id[: -len(_suffix)]
            break
    if l0 is not None and source.endswith(f"-{l0.value}"):
        source = source[: -len(f"-{l0.value}")]

    def url_of(fid: int) -> str:
        return f"https://www.neuronpedia.org/{model_id}/{source}/{fid}"

    desc_by_fid: dict[int, str] = {}
    try:
        client = NeuronpediaClient(model_id, source, cache_path)
        desc_by_fid = client.get_descriptions(sorted(set(fids)), cache_only=not fetch)
        if fetch and not desc_by_fid:
            logger.warning(
                f"no descriptions fetched for {model_id}/{source} — Neuronpedia "
                f"may not publish this SAE."
            )
    except Exception as exc:  # noqa: BLE001 - descriptions are a convenience
        logger.warning(f"Neuronpedia descriptions unavailable ({type(exc).__name__}: {exc})")
    return url_of, desc_by_fid


def _resolve_frequency_blacklist(feature_dir: Path) -> tuple[tuple[int, ...], str]:
    """The activation-frequency blacklist ``(ids, source label)`` the SELECTION
    pipeline materialized — READ FROM DISK, never recomputed with assumed bounds.

    ``precompute_feature_blacklist`` writes ``feature_dir/blacklist/
    freq_min{m}_max{M}.txt`` using THAT run's configured bounds and feeds it to
    ``feature_mask`` via its ``path``, so reading this exact file is the only way
    to rank over the same candidate set production used. Assuming default bounds
    would silently mismatch a run that used different ones. Fail loud if the
    artifact is absent (run the blacklist step with your bounds first) or
    ambiguous (more than one bounds variant on disk)."""
    bl_dir = feature_dir / BLACKLIST_SUBDIR
    txts = sorted(bl_dir.glob("*.txt")) if bl_dir.is_dir() else []
    if not txts:
        raise SystemExit(
            f"no frequency-blacklist artifact under {bl_dir}. This tool ranks over "
            f"the SAME candidate set the selection pipeline uses, so it reads the "
            f"materialized blacklist rather than assuming default bounds. Generate "
            f"it with the SAME bounds your selection/training run used:\n"
            f"  uv run python -m scripts.preprocess.precompute_feature_blacklist "
            f"feature_dir={feature_dir} feature_mask_freq_min=<m> "
            f"feature_mask_freq_max=<M>"
        )
    if len(txts) > 1:
        raise SystemExit(
            f"multiple frequency-blacklist artifacts under {bl_dir}: "
            + ", ".join(p.name for p in txts)
            + " — ambiguous; keep only the one matching your selection bounds."
        )
    txt = txts[0]
    try:
        ids = parse_feature_ids_txt(txt)
    except ManualFeatureSelectionError as exc:
        raise SystemExit(f"invalid frequency-blacklist artifact {txt}: {exc}") from exc
    return tuple(sorted(ids)), f"{BLACKLIST_SUBDIR}/{txt.name}"


def _pick_base_shard(feature_dir: Path, grad_dir: Path, want: str | None) -> Path:
    """The base shard to sample from: ``want`` if given, else the first base
    shard that already has a grad overlay shard."""
    base_shards = glob_features(feature_dir)
    if not base_shards:
        raise SystemExit(f"no base feature shards under {feature_dir}")
    if want is not None:
        for p in base_shards:
            if shard_stem_from_path(p) == want:
                return p
        raise SystemExit(f"shard {want!r} not found under {feature_dir}")
    for p in base_shards:
        if overlay_shard_path(grad_dir, shard_stem_from_path(p)).exists():
            return p
    raise SystemExit(f"no base shard has a grad overlay shard in {grad_dir}")


def _resolve_pct_dir(feature_dir: Path, pct_dir: Path | None) -> Path | None:
    """The percentile overlay dir (``feature_dir/pct/<hash>``), or None — then
    the baseline ranking falls back to raw activation magnitude. Percentile is
    the better baseline: it's what ``top_k_percentile`` selection actually
    ranks by, and it demotes globally-dead-high features (each feature is
    ranked against its OWN distribution)."""
    if pct_dir is not None:
        return pct_dir if pct_dir.is_dir() else None
    # Only TRUE percentile overlays (manifest kind == "pct"); a pct/<hash>/ that
    # is merely the parent of a packed-row tree carries no percentile manifest
    # and would silently degrade this baseline to magnitude if picked.
    cands = [
        p
        for p in sorted((feature_dir / "pct").glob("*"))
        if p.is_dir()
        and (read_manifest(p) or {}).get("kind") == OverlayKind.PERCENTILE.value
    ]
    if not cands:
        logger.warning(
            f"no pct overlay under {feature_dir / 'pct'} — baseline ranks by "
            "RAW MAGNITUDE; run precompute_percentiles for the (better) "
            "percentile baseline."
        )
        return None
    if len(cands) > 1:
        logger.warning(f"multiple pct overlays; using {cands[0].name} (pass --pct-dir)")
    return cands[0]


def _clip(text: str, n: int, *, tail: bool) -> str:
    """Last/first ``n`` chars with an ellipsis marking the cut."""
    if len(text) <= n:
        return text
    return ("…" + text[-n:]) if tail else (text[:n] + "…")


def _feature_lines(
    idxs: list[int],
    fids: t.Tensor,
    mags: t.Tensor,
    grads: t.Tensor,
    url_of,
    desc_by_fid: dict[int, str],
    pcts: t.Tensor | None = None,
    deltas: t.Tensor | None = None,
) -> list[str]:
    lines: list[str] = []
    for i in idxs:
        fid = int(fids[i])
        parts = [f"f{fid}"]
        if pcts is not None:
            parts.append(f"pct={float(pcts[i]):.0f}")
        if deltas is not None:
            parts.append(f"Δpct={float(deltas[i]):+.0f}")
        parts += [f"mag={float(mags[i]):.4g}", f"grad={float(grads[i]):.4g}"]
        desc = desc_by_fid.get(fid)
        if desc:
            parts.append(f'"{desc.strip()}"')
        if url_of is not None:
            parts.append(url_of(fid))
        lines.append("  ".join(parts))
    return lines


def _max_pct_map(
    seg_indices,
    offsets: t.Tensor,
    fids_all: t.Tensor,
    pcts_all: t.Tensor,
) -> dict[int, float]:
    """Per-feature MAX percentile over the given global segments (fid -> pct).

    The baseline for the delta modes: a feature's strongest prior presence.
    Absent features implicitly map to 0 (handled by the caller's ``.get``)."""
    out: dict[int, float] = {}
    for s in seg_indices:
        lo, hi = int(offsets[s]), int(offsets[s + 1])
        for i in range(lo, hi):
            fid = int(fids_all[i])
            p = float(pcts_all[i])
            if p > out.get(fid, 0.0):
                out[fid] = p
    return out


def _delta_order(
    fids: t.Tensor,
    cur_pcts: t.Tensor,
    baseline_pct: dict[int, float],
    top_k: int,
) -> tuple[list[int], t.Tensor]:
    """Rank the current segment's features by percentile INCREASE over a
    baseline (``cur_pct - baseline_pct[fid]``, treating an absent baseline as
    0). Keep STRICTLY positive deltas (a feature that only fell drops out),
    descending, top-k. Returns the index order plus the per-feature delta tensor
    (1:1 with ``fids``) so the chosen lines can show ``Δpct``."""
    deltas = t.tensor(
        [
            float(cur_pcts[i]) - baseline_pct.get(int(fids[i]), 0.0)
            for i in range(len(fids))
        ],
        dtype=t.float32,
    )
    order = sorted(
        (i for i in range(len(fids)) if float(deltas[i]) > 0.0),
        key=lambda i: float(deltas[i]),
        reverse=True,
    )
    return order[:top_k], deltas


def build_examples(
    feature_dir: Path,
    grad_dir: Path | None,
    *,
    shard: str | None,
    num_examples: int,
    top_k: int,
    min_features: int,
    prefix_chars: int,
    suffix_chars: int,
    seed: int,
    data_dir: Path | None,
    dataset_phase: str | None,
    neuronpedia_cache: Path | None = None,
    fetch_descriptions: bool = False,
    pct_dir: Path | None = None,
    show_urls: bool = False,
) -> dict:
    grad_dir = _resolve_grad_dir(feature_dir, grad_dir)
    manifest = read_manifest(grad_dir) or {}
    descriptor = dict(manifest.get("metadata", {}))
    # Extra-sure the gradient score is PERCENTILE-WEIGHTED (pct * sensitivity):
    # refuse any overlay whose reduction isn't the current pct_relu_neg_max, so
    # `by_gradient` can never silently be a stale bare-sensitivity score.
    reduction = descriptor.get("score_reduction")
    if reduction != GRAD_SCORE_REDUCTION:
        raise SystemExit(
            f"grad overlay {grad_dir} has score_reduction={reduction!r}, but this "
            f"tool requires {GRAD_SCORE_REDUCTION!r} (percentile-weighted "
            f"pct * sensitivity). Regenerate the overlay with the current producer."
        )

    base_shard = _pick_base_shard(feature_dir, grad_dir, shard)
    stem = shard_stem_from_path(base_shard)
    data = load_features(base_shard)
    grad = load_overlay_shard(overlay_shard_path(grad_dir, stem))

    offsets = data["offsets"].to(t.int64)
    if "feature_mags" not in data:
        raise SystemExit(
            f"{stem} has no feature_mags — cannot rank by magnitude (re-extract?)"
        )
    fids_all = data["feature_ids"].to(t.int64)
    mags_all = data["feature_mags"].to(t.float32)
    grad_off = grad[GRAD_OFFSETS_KEY].to(t.int64)
    grad_all = grad[GRAD_SCORES_KEY].to(t.float32)
    if not t.equal(grad_off, offsets):
        raise SystemExit(
            f"{stem}: grad overlay offsets differ from base offsets — the grad "
            f"overlay is stale for this base shard."
        )

    # Percentile baseline (the real top_k_percentile selection criterion):
    # uint8 feature_percentiles, CSR-aligned 1:1 with the base feature_ids.
    # uint8 = round(pct/100*255), so pct ≈ uint8 * 100/255. None => fall back to
    # raw magnitude for the baseline ranking.
    resolved_pct_dir = _resolve_pct_dir(feature_dir, pct_dir)
    pcts_all: t.Tensor | None = None
    if resolved_pct_dir is not None:
        pct_shard = overlay_shard_path(resolved_pct_dir, stem)
        if pct_shard.exists():
            raw_pct = load_overlay_shard(pct_shard).get("feature_percentiles")
            if raw_pct is not None and len(raw_pct) == len(fids_all):
                pcts_all = raw_pct.to(t.float32) * (100.0 / 255.0)
            else:
                logger.warning(f"{stem}: pct overlay misaligned/missing — using magnitude")

    doc_ids = data["doc_ids"].to(t.int64)
    char_starts = data["segment_char_starts"].to(t.int64)
    char_ends = data["segment_char_ends"].to(t.int64)
    n_segments = int(data["n_segments"])

    # The ONLY codepath: rank over the SAME candidate set the real
    # top_k_percentile selection sees — apply the activation-FREQUENCY blacklist
    # the selection pipeline materialized, READ FROM the on-disk artifact (its
    # bounds, not assumed defaults — see _resolve_frequency_blacklist). Without
    # it the lists are polluted by globally high-firing features (e.g. f1041)
    # that production blacklists. Filters the flat CSR and rebuilds offsets so
    # every downstream view (segments, neighbours, delta baselines) is masked.
    blacklist_ids, blacklist_src = _resolve_frequency_blacklist(feature_dir)
    if blacklist_ids:
        keep = FeatureMask(
            mode=FeatureMaskMode.blacklist, ids=blacklist_ids
        ).keep_mask(fids_all)
        seg_of_feat = t.repeat_interleave(
            t.arange(n_segments, dtype=t.int64), offsets.diff()
        )
        seg_counts = t.bincount(seg_of_feat[keep], minlength=n_segments)
        offsets = t.cat([t.zeros(1, dtype=t.int64), seg_counts.cumsum(0)])
        fids_all, mags_all, grad_all = fids_all[keep], mags_all[keep], grad_all[keep]
        if pcts_all is not None:
            pcts_all = pcts_all[keep]
        logger.info(
            f"{stem}: frequency whitelist (from {blacklist_src}) dropped "
            f"{int((~keep).sum()):,} of {len(keep):,} feature-slots "
            f"({len(blacklist_ids):,} blacklisted ids)"
        )

    resolved_data_dir = Path(
        resolve_data_dir(str(data_dir) if data_dir is not None else None, dataset_phase)
    )
    texts = read_parquet_texts(resolved_data_dir / f"{stem}.parquet")

    # Eligible = NON-final segment (a real in-document future) with enough
    # stored features to make a top-k comparison meaningful.
    eligible = [
        g
        for g in range(n_segments)
        if g + 1 < n_segments
        and int(doc_ids[g + 1]) == int(doc_ids[g])  # not the doc-final segment
        and int(offsets[g + 1] - offsets[g]) >= min_features
    ]
    if not eligible:
        raise SystemExit(
            f"{stem}: no segment has >= {min_features} features AND a following "
            f"segment; lower --min-features."
        )
    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen = sorted(eligible[:num_examples])

    # Pass 1: gather each example's slices + top-k orders, and the union of
    # displayed feature ids (so the Neuronpedia descriptions are fetched once).
    # The percentile-only modes (following / delta-prev / delta-allprev) need a
    # pct overlay; they are computed iff pcts_all is present.
    pending: list[dict[str, Any]] = []
    displayed_fids: set[int] = set()
    for g in chosen:
        doc = int(doc_ids[g])
        lo, hi = int(offsets[g]), int(offsets[g + 1])
        fids = fids_all[lo:hi]
        mags = mags_all[lo:hi]
        grads = grad_all[lo:hi]
        if not (len(fids) == len(mags) == len(grads)):
            raise SystemExit(f"{stem} seg {g}: mag/grad slice length mismatch")
        pcts = pcts_all[lo:hi] if pcts_all is not None else None

        # First global segment of this doc (segments are doc-ordered, so the
        # doc's run is contiguous): walk back over same-doc neighbours.
        ds = g
        while ds > 0 and int(doc_ids[ds - 1]) == doc:
            ds -= 1

        order_mag = sorted(range(len(fids)), key=lambda i: float(mags[i]), reverse=True)
        order_grad = sorted(range(len(fids)), key=lambda i: float(grads[i]), reverse=True)
        top_mag, top_grad = order_mag[:top_k], order_grad[:top_k]
        ex_p: dict[str, Any] = {
            "g": g, "doc": doc, "prev_in_doc": g - ds,
            "fids": fids, "mags": mags, "grads": grads, "pcts": pcts,
            "top_mag": top_mag, "top_grad": top_grad, "top_pct": None,
        }
        displayed_fids.update(int(fids[i]) for i in (*top_mag, *top_grad))

        if pcts is not None:
            assert pcts_all is not None  # pcts is a slice of pcts_all
            pvals = [float(x) for x in pcts]  # local list avoids None-subscript in the key
            top_pct = sorted(range(len(fids)), key=lambda i: pvals[i], reverse=True)[:top_k]
            ex_p["top_pct"] = top_pct
            displayed_fids.update(int(fids[i]) for i in top_pct)

            # (a) following segment (g+1; same-doc & present by eligibility),
            # ranked by its OWN percentile — an oracle for "what's actually next".
            nlo, nhi = int(offsets[g + 1]), int(offsets[g + 2])
            nfids, nmags = fids_all[nlo:nhi], mags_all[nlo:nhi]
            ngrads, npcts = grad_all[nlo:nhi], pcts_all[nlo:nhi]
            npvals = [float(x) for x in npcts]
            top_following = sorted(
                range(len(nfids)), key=lambda i: npvals[i], reverse=True
            )[:top_k]
            ex_p.update(nfids=nfids, nmags=nmags, ngrads=ngrads, npcts=npcts,
                        top_following=top_following)
            displayed_fids.update(int(nfids[i]) for i in top_following)

            # (b) delta vs the immediately previous segment; (c) delta vs the
            # running max over ALL previous segments in the doc. Both rank the
            # CURRENT features by percentile increase (see _delta_order).
            prev_map = _max_pct_map([g - 1], offsets, fids_all, pcts_all) if g > ds else {}
            allprev_map = _max_pct_map(range(ds, g), offsets, fids_all, pcts_all)
            top_dprev, dprev = _delta_order(fids, pcts, prev_map, top_k)
            top_dall, dall = _delta_order(fids, pcts, allprev_map, top_k)
            ex_p.update(top_dprev=top_dprev, dprev=dprev, top_dall=top_dall, dall=dall)
            displayed_fids.update(int(fids[i]) for i in (*top_dprev, *top_dall))

        pending.append(ex_p)

    # Pass 2: resolve Neuronpedia URLs + descriptions once for the union, then
    # render each example's ranked lists (3 current-segment + up to 3 pct-only).
    cache_path = neuronpedia_cache or (feature_dir / "neuronpedia_cache.yaml")
    url_of, desc_by_fid = _neuronpedia(
        descriptor, cache_path, sorted(displayed_fids), fetch=fetch_descriptions
    )

    url_arg = url_of if show_urls else None
    baseline = "percentile" if pcts_all is not None else "magnitude"
    examples = []
    for p in pending:
        g, doc = p["g"], p["doc"]
        fids, mags, grads, pcts = p["fids"], p["mags"], p["grads"], p["pcts"]
        top_mag, top_grad, top_pct = p["top_mag"], p["top_grad"], p["top_pct"]
        baseline_top = top_pct if top_pct is not None else top_mag
        set_b = {int(fids[i]) for i in baseline_top}
        set_g = {int(fids[i]) for i in top_grad}
        jaccard = len(set_b & set_g) / len(set_b | set_g) if (set_b | set_g) else 0.0
        cs, ce = int(char_starts[g]), int(char_ends[g])
        text = texts[doc]
        ex: dict[str, Any] = {
            "doc": doc,
            "global_segment": g,
            "n_features": len(fids),
            "prev_segments_in_doc": p["prev_in_doc"],
            # Percentile-ceiling plateau width: how many of this segment's stored
            # features sit at the percentile CAP. magnitude_to_percentile clips
            # "above p99" to 99.5 -> uint8 254 -> pct≈99.6 (displays as 100), so
            # the reachable ceiling is uint8 254, NOT 255. When this is >> the
            # selection k (geometric, ~5-25), top_k_percentile mostly chooses
            # among ceiling ties (seeded-RNG tie-break), so the percentile RANK
            # carries little signal — read the pct lists as SETS, not orderings.
            "pct100_count": (int((pcts >= 99.5).sum()) if pcts is not None else None),
            "top_k_jaccard": round(jaccard, 4),  # baseline vs gradient
            "prompt": _clip(text[:cs], prefix_chars, tail=True),
            "segment": text[cs:ce],
            "future": _clip(text[ce:], suffix_chars, tail=False),
        }
        if top_pct is not None:
            ex["by_percentile"] = _feature_lines(
                top_pct, fids, mags, grads, url_arg, desc_by_fid, pcts
            )
        ex["by_magnitude"] = _feature_lines(top_mag, fids, mags, grads, url_arg, desc_by_fid, pcts)
        ex["by_gradient"] = _feature_lines(top_grad, fids, mags, grads, url_arg, desc_by_fid, pcts)
        if top_pct is not None:
            ex["by_following_percentile"] = _feature_lines(
                p["top_following"], p["nfids"], p["nmags"], p["ngrads"],
                url_arg, desc_by_fid, p["npcts"],
            )
            ex["by_delta_prev"] = _feature_lines(
                p["top_dprev"], fids, mags, grads, url_arg, desc_by_fid, pcts, p["dprev"],
            )
            ex["by_delta_allprev"] = _feature_lines(
                p["top_dall"], fids, mags, grads, url_arg, desc_by_fid, pcts, p["dall"],
            )
        examples.append(ex)

    return {
        "meta": {
            "feature_dir": str(feature_dir),
            "grad_overlay": str(grad_dir),
            "shard": stem,
            "attribution_mode": descriptor.get("attribution_mode"),
            # The grad score is pct * sensitivity (guarded at load). Shown so the
            # percentile-weighting is visible, not just assumed.
            "score_reduction": descriptor.get("score_reduction"),
            # BOS rows are excluded from pooling/gradient ONLY when the run set
            # ignore_bos_features (default False ⇒ BOS activations ARE included
            # and dominate the doc-initial segment's magnitudes).
            "ignore_bos_features": descriptor.get("ignore_bos_features"),
            # Every list ranks over the frequency-WHITELISTED candidate set
            # (production parity): the blacklist is READ from the on-disk artifact
            # the selection pipeline wrote (its bounds are encoded in the name),
            # never assumed.
            "frequency_filter": blacklist_src,
            "model": descriptor.get("model"),
            "sae": {
                "release": descriptor.get("sae_release"),
                "width": descriptor.get("sae_width"),
                "l0": descriptor.get("sae_l0"),
                "layer": descriptor.get("layer"),
                "activation_type": descriptor.get("activation_type"),
            },
            "top_k": top_k,
            "num_examples": len(examples),
            "baseline": baseline,  # "percentile" (the real selection criterion) or "magnitude"
            "descriptions_found": f"{len(desc_by_fid)}/{len(displayed_fids)}",
            "how_to_read": (
                "All lists rank over the frequency-WHITELISTED candidate set "
                "(see `frequency_filter`): the same dead + near-ubiquitous "
                "features that production's top_k_percentile selection drops are "
                "removed before ranking, so a globally high-firing feature like "
                "f1041 cannot appear. `by_gradient` is percentile-weighted "
                "(`score_reduction`, guarded at load). "
                "Each example is one conditioning segment. The first three lists "
                "rank the SAME stored (current-segment) features. `by_percentile` "
                "(present iff a pct overlay exists) ranks by activation PERCENTILE "
                "— what top_k_percentile selection actually uses, and the fairer "
                "baseline (each feature vs its OWN distribution, so globally "
                "high-firing features don't dominate). `by_magnitude` ranks by "
                "raw activation. `by_gradient` ranks by the STORED gradient "
                "score — `percentile(a) * max over the segment's rows of "
                "ReLU(-dL/df)` through the SAE decoder: the feature's activation "
                "percentile times its loss-helpfulness sensitivity to the "
                "`future` text (so a barely-active feature with a large per-unit "
                "gradient is demoted by its low percentile). The next three lists "
                "(present iff a pct overlay exists) are percentile-based "
                "alternatives: `by_following_percentile` ranks the FOLLOWING "
                "segment's own features by percentile — an ORACLE for what is "
                "actually active next (it peeks at the future; not a usable "
                "selection criterion, a ceiling). `by_delta_prev` ranks the "
                "current features by percentile INCREASE over the immediately "
                "previous segment (`Δpct = cur_pct - prev_seg_pct`, 0 if the "
                "feature was absent there; strictly-positive deltas only) — "
                "newly-prominent features. `by_delta_allprev` ranks by the "
                "increase over the running MAX percentile across ALL previous "
                "segments in the doc — features hitting a new high-water mark. "
                "`prev_segments_in_doc` is how many segments precede this one in "
                "the doc; when 0 the two delta lists degenerate to `by_percentile`. "
                "Each line shows all signals (`pct=` 0-100 when available, `Δpct=` "
                "on the delta lists, `mag=`, `grad=`), the Neuronpedia description "
                "in quotes when cached (see `descriptions_found`; pass "
                "--fetch-descriptions), and a link when --show-urls is set. "
                "top_k_jaccard is the overlap of the BASELINE (see `baseline`) "
                "vs gradient top-k sets (low => they disagree). NOTE on ties: "
                "these lists break exact-score ties by ascending feature id "
                "(deterministic), whereas production top_k_percentile selection "
                "breaks them with a SEEDED-RANDOM draw. When the percentile "
                "ceiling plateau (`pct100_count`) is wider than k, the tie-break "
                "decides membership, so `by_percentile`'s displayed SET (not just "
                "its order) can differ from what production actually selects — "
                "the k-th-slot pick is arbitrary in BOTH. Read `by_percentile` as "
                "representative high-percentile features, not production's exact "
                "selection. (`by_gradient` is continuous, so ties ~never bind.)"
            ),
        },
        "examples": examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature-dir", type=Path, required=True)
    ap.add_argument("--grad-dir", type=Path, default=None,
                    help="grad overlay dir; default = the single <feature_dir>/grad/<hash>")
    ap.add_argument("--shard", default=None, help="base shard stem (default: first with an overlay)")
    ap.add_argument("--num-examples", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-features", type=int, default=None,
                    help="only sample segments with >= this many stored features (default: top-k)")
    ap.add_argument("--prefix-chars", type=int, default=300)
    ap.add_argument("--suffix-chars", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", type=Path, default=None, help="source parquet dir (default: nanochat)")
    ap.add_argument("--dataset-phase", default=None)
    ap.add_argument("--pct-dir", type=Path, default=None,
                    help="percentile overlay dir (default: the single <feature_dir>/pct/<hash>); "
                         "absent => baseline falls back to raw magnitude")
    ap.add_argument("--neuronpedia-cache", type=Path, default=None,
                    help="Neuronpedia description cache (default: <feature_dir>/neuronpedia_cache.yaml)")
    ap.add_argument("--fetch-descriptions", action="store_true",
                    help="fetch missing feature descriptions from Neuronpedia (network) and cache them; "
                         "default is cache-only (offline)")
    ap.add_argument("--show-urls", action="store_true",
                    help="append the Neuronpedia URL to each feature line (default: off)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output YAML (default: out/analysis/gradient_vs_magnitude_<shard>.yaml)")
    args = ap.parse_args()

    result = build_examples(
        args.feature_dir,
        args.grad_dir,
        shard=args.shard,
        num_examples=args.num_examples,
        top_k=args.top_k,
        min_features=args.min_features if args.min_features is not None else args.top_k,
        prefix_chars=args.prefix_chars,
        suffix_chars=args.suffix_chars,
        seed=args.seed,
        data_dir=args.data_dir,
        dataset_phase=args.dataset_phase,
        neuronpedia_cache=args.neuronpedia_cache,
        fetch_descriptions=args.fetch_descriptions,
        pct_dir=args.pct_dir,
        show_urls=args.show_urls,
    )

    out = args.out or Path("out/analysis") / f"gradient_vs_magnitude_{result['meta']['shard']}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(result, sort_keys=False, allow_unicode=True, width=10_000),
        encoding="utf-8",
    )
    logger.info(f"Wrote {len(result['examples'])} examples to {out}")
    print(out)


if __name__ == "__main__":
    main()
