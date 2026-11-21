"""I/O helpers for precomputed feature-conditioning bag artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch as t
from safetensors.torch import safe_open

from nanochat.atomic_io import (
    atomic_json_dump,
    atomic_safetensors_save,
)
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.precomputed_bag_family import (
    PrecomputedBagFamily,
    coerce_precomputed_bag_family,
)
from steering.immutable_io import ImmutableArtifactError, install_bytes_once
from steering.safetensors_io import deterministic_safetensors_bytes

#: Safetensors metadata key :func:`write_ours_gcg_bag` stamps the derive-time
#: checkpoint's ``V``-fingerprint under (see
#: ``steering.eval.axbench_bpb.vocab_restriction.checkpoint_vocab_fingerprint``).
#: A single well-known constant so writers and readers can never drift on the key
#: name -- ``read_ours_gcg_bag`` returns it as part of the metadata mapping it
#: already returns, and callers verifying a checkpoint match read it via this key.
VOCAB_FINGERPRINT_META_KEY = "vocab_fingerprint"
FEATURE_CONTRACT_FINGERPRINT_META_KEY = "feature_contract_fingerprint"


def write_ours_gcg_bag(
    path: Any,
    bag: FeatureConditioning,
    *,
    metadata: Mapping[str, str],
    vocab_fingerprint: str,
) -> None:
    """Atomically write *bag* as ``{feature_ids, buckets}`` safetensors.

    ``vocab_fingerprint`` is the DERIVE-TIME checkpoint's ``V``-fingerprint (see
    ``checkpoint_vocab_fingerprint`` -- the ``"uncompacted"`` sentinel for a legacy
    checkpoint) and is stamped into the safetensors metadata under
    :data:`VOCAB_FINGERPRINT_META_KEY`, alongside the caller's own ``metadata``.

    This is a REQUIRED argument, not an optional/defaulted one: the on-disk bag
    cache key is ``(concept_id, mode, reg_lambda, family, prefix_id)`` under an
    operator-chosen ``run_tag`` -- it carries NO checkpoint/``V`` component. So
    reusing a ``run_tag`` across checkpoints with a different ``V`` is otherwise
    indistinguishable, at read time, from a legitimate cache hit, and would
    silently condition on the wrong feature ids. Every writer must positively
    declare which checkpoint it derived against; a silently-defaulted stamp (e.g.
    always ``"uncompacted"``) would mislabel real compacted derives and defeat the
    read-side check this enables (see ``steering.eval.axbench_bpb.intervention``).

    Raises:
        AxBenchBpbAdvError: ``metadata`` already carries
            :data:`VOCAB_FINGERPRINT_META_KEY` -- this function owns that key
            exclusively so a caller-supplied value can never be silently
            shadowed by the stamped one.
    """
    if VOCAB_FINGERPRINT_META_KEY in metadata:
        raise AxBenchBpbAdvError(
            f"metadata already carries {VOCAB_FINGERPRINT_META_KEY!r} "
            f"(value={metadata[VOCAB_FINGERPRINT_META_KEY]!r}); write_ours_gcg_bag "
            "stamps this key itself from the vocab_fingerprint argument -- callers "
            "must not set it directly in their metadata mapping."
        )
    fids = list(bag.feature_ids())
    buckets = [bucket for _fid, bucket in bag.entries]
    tensors = {
        "feature_ids": t.tensor(fids, dtype=t.int64),
        "buckets": t.tensor(buckets, dtype=t.int64),
    }
    full_metadata = {**dict(metadata), VOCAB_FINGERPRINT_META_KEY: vocab_fingerprint}
    if FEATURE_CONTRACT_FINGERPRINT_META_KEY in full_metadata:
        # Clustered bag files are content-addressed and byte-compared on resume.
        # Canonicalize the safetensors header because the Rust metadata hash map
        # does not promise stable JSON key order between processes. Keep the
        # exact legacy save_file path for unclustered artifacts.
        rendered = deterministic_safetensors_bytes(tensors, metadata=full_metadata)
        final_path = Path(path)
        try:
            install_bytes_once(
                rendered,
                final_path,
                surface="clustered precomputed bag",
            )
        except ImmutableArtifactError as exc:
            raise AxBenchBpbAdvError(
                "clustered precomputed bag content changed on resume; use a "
                f"fresh run_tag instead of overwriting {final_path}: {exc}"
            ) from exc
    else:
        atomic_safetensors_save(tensors, path, metadata=full_metadata)


def read_ours_gcg_bag(
    path: Any,
    *,
    expected_feature_contract_fingerprint: str | None = None,
) -> tuple[dict[int, int], dict[str, str]]:
    """Read a precomputed bag written by :func:`write_ours_gcg_bag`.

    The returned metadata mapping carries :data:`VOCAB_FINGERPRINT_META_KEY` when
    the artifact was written by a ``vocab_fingerprint``-stamping ``write_ours_gcg_bag``
    (every current writer); it is ABSENT for a bag written before that stamp
    existed.

    When ``expected_feature_contract_fingerprint`` is supplied, this reader
    validates this bag artifact's own
    :data:`FEATURE_CONTRACT_FINGERPRINT_META_KEY` stamp. With no expectation, a
    cluster-stamped artifact is rejected as an unclustered read; genuinely
    unstamped legacy bags remain readable. The scored-prefix reader applies the
    same rule to its separate JSON artifact. A flow consuming both artifacts
    must pass the same expected fingerprint to both readers. Neither leaf
    reader can by itself bind the two independently committed artifacts.
    """
    path_text = str(path)
    with safe_open(path_text, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        fids = handle.get_tensor("feature_ids")
        buckets = handle.get_tensor("buckets")
    actual_fingerprint = metadata.get(FEATURE_CONTRACT_FINGERPRINT_META_KEY)
    if expected_feature_contract_fingerprint is None:
        if actual_fingerprint is not None:
            raise AxBenchBpbAdvError(
                f"unclustered reader rejects cluster-stamped precomputed bag "
                f"artifact {path_text}"
            )
    elif actual_fingerprint != expected_feature_contract_fingerprint:
        raise AxBenchBpbAdvError(
            f"precomputed bag {path_text} feature-contract fingerprint mismatch: "
            f"expected={expected_feature_contract_fingerprint}, "
            f"actual={actual_fingerprint!r}"
        )
    bag = {
        int(fid): int(bucket)
        for fid, bucket in zip(fids.tolist(), buckets.tolist(), strict=True)
    }
    return bag, metadata


_SCORED_PREFIX_IDS_KEY = "scored_core_prefix_ids"


def write_ours_gcg_scored_prefixes(
    path: Any,
    core_prefix_ids: Sequence[str],
    *,
    concept_id: int,
    mode: SteeringMode,
    family: PrecomputedBagFamily | str,
    feature_contract_fingerprint: str | None = None,
) -> None:
    """Persist the scored ``core_prefix_id`` set for one bag family group.

    The JSON key remains ``"mode_tag"`` for compatibility with artifacts written
    before this neutral module existed.
    """
    bag_family = coerce_precomputed_bag_family(family)
    payload: dict[str, Any] = {
        "concept_id": int(concept_id),
        "mode": SteeringMode(mode).value,
        "mode_tag": bag_family.value,
        _SCORED_PREFIX_IDS_KEY: sorted({str(prefix_id) for prefix_id in core_prefix_ids}),
    }
    if feature_contract_fingerprint is not None:
        payload[FEATURE_CONTRACT_FINGERPRINT_META_KEY] = feature_contract_fingerprint
        rendered = json.dumps(payload, indent=2).encode("utf-8")
        final_path = Path(path)
        try:
            install_bytes_once(
                rendered,
                final_path,
                surface="clustered precomputed-bag scored-prefix artifact",
            )
        except ImmutableArtifactError as exc:
            raise AxBenchBpbAdvError(
                "clustered precomputed-bag scored-prefix content changed on "
                f"resume; use a fresh run_tag instead of overwriting "
                f"{final_path}: {exc}"
            ) from exc
    else:
        atomic_json_dump(payload, path)


def read_ours_gcg_scored_prefixes(
    path: Any,
    *,
    expected_feature_contract_fingerprint: str | None = None,
) -> list[str]:
    """Read scored ``core_prefix_id``s written by ``write_ours_gcg_scored_prefixes``.

    A missing artifact means the precomputed-bag derive did not run or did not
    complete for the requested group; callers should treat that as a pipeline
    inconsistency rather than a normal empty case.

    When ``expected_feature_contract_fingerprint`` is supplied, this reader
    validates this scored-prefix artifact's own
    :data:`FEATURE_CONTRACT_FINGERPRINT_META_KEY` stamp. With no expectation, a
    cluster-stamped artifact is rejected as an unclustered read. The bag reader
    applies the same rule to its separate safetensors artifact. A flow consuming
    both artifacts must pass the same expected fingerprint to both readers.
    Neither leaf reader can by itself bind the two independently committed
    artifacts.
    """
    p = Path(path)
    if not p.exists():
        raise AxBenchBpbAdvError(
            f"missing precomputed-bag scored-prefix artifact: {p}. The "
            "BUILD_CANDIDATES derive must write one per completed "
            "(concept, mode, family) group before scoring; a missing artifact "
            "means the derive did not run for this group."
        )
    try:
        payload = json.loads(p.read_text())
        ids = payload[_SCORED_PREFIX_IDS_KEY]
    except Exception as exc:
        raise AxBenchBpbAdvError(
            f"corrupt or unreadable precomputed-bag scored-prefix artifact {p}: {exc}"
        ) from exc
    actual_fingerprint = payload.get(FEATURE_CONTRACT_FINGERPRINT_META_KEY)
    if expected_feature_contract_fingerprint is None:
        if actual_fingerprint is not None:
            raise AxBenchBpbAdvError(
                f"unclustered reader rejects cluster-stamped scored-prefix "
                f"artifact {p}"
            )
    elif actual_fingerprint != expected_feature_contract_fingerprint:
        raise AxBenchBpbAdvError(
            f"scored-prefix artifact {p} feature-contract fingerprint mismatch: "
            f"expected={expected_feature_contract_fingerprint}, "
            f"actual={actual_fingerprint!r}"
        )
    return [str(prefix_id) for prefix_id in ids]


__all__ = [
    "FEATURE_CONTRACT_FINGERPRINT_META_KEY",
    "VOCAB_FINGERPRINT_META_KEY",
    "read_ours_gcg_bag",
    "read_ours_gcg_scored_prefixes",
    "write_ours_gcg_bag",
    "write_ours_gcg_scored_prefixes",
]
