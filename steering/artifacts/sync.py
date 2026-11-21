"""Push / pull wrappers around `huggingface_hub.sync_bucket`."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import cast

from huggingface_hub import sync_bucket
from loguru import logger

from steering.artifacts.categories import resolve_category
from steering.artifacts.config import ArtifactsConfig
from steering.artifacts.filters import resolve_filters
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.feature_io import (
    COMPRESSED_EXT,
    TensorKey,
    glob_features,
    read_feature_value_count,
)
from steering.overlay_io import (
    CHECKSUM_PREFIX_LEN,
    MANIFEST_FILENAME,
    compute_base_shard_content_hash,
    glob_overlay_shards,
    overlay_tensors_content_digest,
    read_manifest,
    validate_percentile_overlay_shard,
)
from steering.percentile_remote import (
    OVERLAY_STATISTICS_FILENAME,
    RemotePercentileError,
    materialize_overlay_statistics,
    parse_remote_percentile_receipt,
    validate_remote_percentile_contract,
    validate_remote_percentile_manifest,
)
from steering.remote_feature_inventory import (
    RemoteFeatureShard,
    remote_feature_inventory_digest,
)
from steering.remote_training import (
    VERIFIED_UPLOAD_CLAIMS_DIRNAME,
    RemoteObjectInfo,
    RemoteTrainingError,
    validate_local_verified_upload_claim,
    verified_local_object_info,
)
from steering.stats_io import load_statistics


def _is_remote_percentile_candidate(manifest_path: Path) -> bool:
    overlay_dir = manifest_path.parent
    has_file_marker = any(
        path.exists() or path.is_symlink()
        for path in (
            overlay_dir / OVERLAY_STATISTICS_FILENAME,
            overlay_dir / "_remote_contract.json",
            overlay_dir / "_receipts",
        )
    )
    manifest = read_manifest(overlay_dir)
    metadata = manifest.get("metadata") if manifest is not None else None
    return has_file_marker or (isinstance(metadata, dict) and "remote_contract_path" in metadata)


def _select_remote_percentile_manifests(local_features_root: Path) -> list[Path]:
    candidates_by_feature_dir: dict[Path, list[Path]] = {}
    for manifest_path in sorted(local_features_root.glob(f"**/pct/*/{MANIFEST_FILENAME}")):
        if _is_remote_percentile_candidate(manifest_path):
            feature_dir = manifest_path.parent.parent.parent
            candidates_by_feature_dir.setdefault(feature_dir, []).append(manifest_path)

    selected: list[Path] = []
    for feature_dir, candidates in candidates_by_feature_dir.items():
        root_statistics = feature_dir / OVERLAY_STATISTICS_FILENAME
        if root_statistics.exists() or root_statistics.is_symlink():
            statistics = load_statistics(root_statistics)
            checksum = compute_percentile_checksum(
                build_percentile_table(
                    statistics.per_feature_stats,
                    statistics.metadata.n_sae_features,
                )
            )
            expected = feature_dir / "pct" / checksum[:CHECKSUM_PREFIX_LEN] / MANIFEST_FILENAME
            if expected in candidates:
                selected.append(expected)
            continue
        if len(candidates) != 1:
            raise ValueError(
                "pulled feature directory has multiple remote percentile families but no "
                f"root statistics selector: {feature_dir}"
            )
        selected.append(candidates[0])
    return sorted(selected)


def _validate_verified_claim_directory(
    claim_root: Path,
    *,
    expected_objects: list[RemoteObjectInfo],
    canonical_parent: PurePosixPath,
) -> None:
    expected_by_name = {PurePosixPath(info.path).name: info for info in expected_objects}
    if len(expected_by_name) != len(expected_objects) or any(
        PurePosixPath(info.path).parent != canonical_parent for info in expected_objects
    ):
        raise ValueError(f"pulled percentile canonical-object inventory is malformed: {claim_root}")
    try:
        entries = list(claim_root.iterdir())
    except OSError as exc:
        raise ValueError(
            f"pulled percentile family lacks readable verified-upload claims: {claim_root}"
        ) from exc
    if (
        claim_root.is_symlink()
        or not claim_root.is_dir()
        or {path.name for path in entries} != set(expected_by_name)
    ):
        raise ValueError(f"pulled percentile verified-upload claim coverage differs: {claim_root}")
    for canonical_name, expected in expected_by_name.items():
        target_claim_dir = claim_root / canonical_name
        try:
            claim_entries = list(target_claim_dir.iterdir())
        except OSError as exc:
            raise ValueError(
                f"pulled percentile verified-upload claim is unreadable: {target_claim_dir}"
            ) from exc
        expected_claim = target_claim_dir / f"{expected.xet_hash}.json"
        if (
            target_claim_dir.is_symlink()
            or not target_claim_dir.is_dir()
            or claim_entries != [expected_claim]
        ):
            raise ValueError(
                f"pulled percentile verified-upload claim is not unique: {target_claim_dir}"
            )
        try:
            validate_local_verified_upload_claim(
                expected_claim,
                expected=expected,
            )
        except RemoteTrainingError as exc:
            raise ValueError(
                f"pulled percentile verified-upload claim is invalid: {expected_claim}"
            ) from exc


def materialize_pulled_percentile_statistics(local_features_root: Path) -> int:
    """Merge each selected complete remote-percentile statistics copy into its root."""
    materialized = 0
    for manifest_path in _select_remote_percentile_manifests(local_features_root):
        overlay_dir = manifest_path.parent
        overlay_statistics = overlay_dir / OVERLAY_STATISTICS_FILENAME
        contract_path = overlay_dir / "_remote_contract.json"
        receipts_dir = overlay_dir / "_receipts"
        manifest = read_manifest(overlay_dir)
        if manifest is None:
            raise ValueError(
                f"pulled percentile statistics has no readable manifest: {manifest_path}"
            )
        expected_manifest_payload = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
        try:
            observed_manifest_payload = manifest_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"pulled percentile manifest is unreadable: {manifest_path}") from exc
        if observed_manifest_payload != expected_manifest_payload:
            raise ValueError(f"pulled percentile manifest is not canonical: {manifest_path}")
        if not overlay_statistics.is_file():
            raise ValueError(
                "pulled percentile manifest has no scoped statistics artifact: "
                f"{overlay_statistics}"
            )
        feature_dir = overlay_dir.parent.parent
        try:
            contract_payload = contract_path.read_bytes()
            contract = json.loads(contract_payload)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"pulled percentile family has no readable remote contract: {contract_path}"
            ) from exc
        if not isinstance(contract, dict) or contract_payload != json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8"):
            raise ValueError(
                f"pulled percentile remote contract identity is invalid: {contract_path}"
            )
        try:
            validate_remote_percentile_contract(
                contract,
                surface="pulled percentile remote contract",
            )
        except RemotePercentileError as exc:
            raise ValueError(
                f"pulled percentile remote contract is malformed: {contract_path}: {exc}"
            ) from exc
        contract_sha256 = hashlib.sha256(contract_payload).hexdigest()
        contract_source = cast(dict[str, object], contract["source"])
        contract_table = cast(dict[str, object], contract["percentile_table"])
        contract_output = cast(dict[str, object], contract["output"])
        checksum = cast(str, contract_table["checksum"])
        source_prefix = cast(str, contract_source["prefix"])
        source_shard_count = cast(int, contract_source["shard_count"])
        source_inventory_sha256 = cast(str, contract_source["inventory_sha256"])
        output_prefix = PurePosixPath(cast(str, contract_output["prefix"]))
        expected_contract_path = (output_prefix / "_remote_contract.json").as_posix()
        statistics_info = verified_local_object_info(
            local_path=overlay_statistics,
            remote_path=(output_prefix / OVERLAY_STATISTICS_FILENAME).as_posix(),
        )
        try:
            validate_remote_percentile_manifest(
                manifest,
                checksum=checksum,
                contract=contract,
                contract_sha256=contract_sha256,
                statistics_info=statistics_info,
            )
        except RemotePercentileError as exc:
            raise ValueError(
                "pulled percentile manifest disagrees with its remote contract: "
                f"{manifest_path}: {exc}"
            ) from exc
        if overlay_dir.name != checksum[:CHECKSUM_PREFIX_LEN]:
            raise ValueError(f"pulled percentile path disagrees with its checksum: {manifest_path}")
        base_shard_content_hash = cast(str, manifest["base_shard_content_hash"])
        raw_feature_paths = glob_features(feature_dir)
        if source_shard_count != len(raw_feature_paths):
            raise ValueError(
                "pulled percentile source count differs from the local raw feature "
                f"family: {overlay_dir}"
            )
        expected_payload_names = {
            path.name.removesuffix(f"_features{COMPRESSED_EXT}") + COMPRESSED_EXT
            for path in raw_feature_paths
        }
        observed_payloads = {path.name: path for path in glob_overlay_shards(overlay_dir)}
        if not expected_payload_names or set(observed_payloads) != expected_payload_names:
            raise ValueError(
                "pulled percentile overlay payload coverage differs from the local raw "
                f"feature family: {overlay_dir}"
            )
        if compute_base_shard_content_hash(raw_feature_paths) != base_shard_content_hash:
            raise ValueError(
                "pulled percentile manifest base-shard identity differs from the local "
                f"raw feature family: {manifest_path}"
            )
        receipt_entries = list(receipts_dir.iterdir()) if receipts_dir.is_dir() else []
        receipt_paths = sorted(
            path
            for path in receipt_entries
            if path.is_file() and re.fullmatch(r"shard_\d{5}\.json", path.name)
        )
        receipt_claim_root = receipts_dir / VERIFIED_UPLOAD_CLAIMS_DIRNAME
        if set(receipt_entries) != {*receipt_paths, receipt_claim_root} or len(
            receipt_paths
        ) != len(raw_feature_paths):
            raise ValueError(
                "pulled percentile receipt coverage differs from the local raw feature "
                f"family: {overlay_dir}"
            )
        receipts = {}
        for receipt_path in receipt_paths:
            try:
                receipt_payload = receipt_path.read_bytes()
                decoded = json.loads(receipt_payload)
                raw_receipt = decoded if isinstance(decoded, dict) else None
            except (UnicodeError, json.JSONDecodeError, OSError):
                raw_receipt = None
            if raw_receipt is None:
                raise ValueError(f"pulled percentile receipt is unreadable: {receipt_path}")
            try:
                receipt = parse_remote_percentile_receipt(
                    raw_receipt,
                    surface=str(receipt_path),
                )
            except Exception as exc:
                raise ValueError(f"pulled percentile receipt is malformed: {receipt_path}") from exc
            if receipt_payload != json.dumps(
                receipt.to_json(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8"):
                raise ValueError(f"pulled percentile receipt is not canonical: {receipt_path}")
            filename_index = int(receipt_path.stem.removeprefix("shard_"))
            if receipt.shard_index != filename_index:
                raise ValueError(
                    f"pulled percentile receipt filename/index mismatch: {receipt_path}"
                )
            receipts[receipt.shard_index] = receipt
        if set(receipts) != set(range(len(raw_feature_paths))):
            raise ValueError(f"pulled percentile receipt indices are not contiguous: {overlay_dir}")

        receipt_inventory = tuple(
            RemoteFeatureShard(index=index, info=receipts[index].source)
            for index in range(len(raw_feature_paths))
        )
        if remote_feature_inventory_digest(receipt_inventory) != source_inventory_sha256:
            raise ValueError(
                f"pulled percentile receipt inventory differs from the run contract: {overlay_dir}"
            )

        for index, raw_path in enumerate(raw_feature_paths):
            payload_name = raw_path.name.removesuffix(f"_features{COMPRESSED_EXT}") + COMPRESSED_EXT
            payload_path = observed_payloads[payload_name]
            expected_numel = read_feature_value_count(raw_path)
            receipt = receipts[index]
            values = validate_percentile_overlay_shard(
                payload_path,
                expected_numel=expected_numel,
            )
            if (
                receipt.contract_sha256 != contract_sha256
                or receipt.n_values != expected_numel
                or receipt.source.path != PurePosixPath(source_prefix, raw_path.name).as_posix()
                or receipt.output.path != (output_prefix / payload_name).as_posix()
                or verified_local_object_info(
                    local_path=raw_path,
                    remote_path=receipt.source.path,
                )
                != receipt.source
                or verified_local_object_info(
                    local_path=payload_path,
                    remote_path=receipt.output.path,
                )
                != receipt.output
            ):
                raise ValueError(
                    f"pulled percentile payload differs from its receipt: {payload_path}"
                )
            if (
                overlay_tensors_content_digest({TensorKey.FEATURE_PERCENTILES.value: values})
                != receipt.semantic_content_sha256
            ):
                raise ValueError(
                    f"pulled percentile payload semantic digest differs: {payload_path}"
                )

        contract_info = verified_local_object_info(
            local_path=contract_path,
            remote_path=expected_contract_path,
        )
        manifest_info = verified_local_object_info(
            local_path=manifest_path,
            remote_path=(output_prefix / MANIFEST_FILENAME).as_posix(),
        )
        receipt_infos = [
            verified_local_object_info(
                local_path=receipt_path,
                remote_path=(output_prefix / "_receipts" / receipt_path.name).as_posix(),
            )
            for receipt_path in receipt_paths
        ]
        _validate_verified_claim_directory(
            overlay_dir / VERIFIED_UPLOAD_CLAIMS_DIRNAME,
            expected_objects=[
                contract_info,
                *[receipts[index].output for index in range(len(raw_feature_paths))],
                statistics_info,
                manifest_info,
            ],
            canonical_parent=output_prefix,
        )
        _validate_verified_claim_directory(
            receipt_claim_root,
            expected_objects=receipt_infos,
            canonical_parent=output_prefix / "_receipts",
        )
        if compute_base_shard_content_hash(raw_feature_paths) != base_shard_content_hash:
            raise ValueError(
                "local raw feature family changed during pulled percentile validation: "
                f"{feature_dir}"
            )
        for index, raw_path in enumerate(raw_feature_paths):
            payload_name = raw_path.name.removesuffix(f"_features{COMPRESSED_EXT}") + COMPRESSED_EXT
            receipt = receipts[index]
            if (
                verified_local_object_info(
                    local_path=observed_payloads[payload_name],
                    remote_path=receipt.output.path,
                )
                != receipt.output
            ):
                raise ValueError(
                    "pulled percentile payload changed during final validation: "
                    f"{observed_payloads[payload_name]}"
                )
        materialize_overlay_statistics(
            overlay_statistics_path=overlay_statistics,
            feature_root_statistics_path=feature_dir / OVERLAY_STATISTICS_FILENAME,
            percentile_checksum=checksum,
            expected_statistics_object=statistics_info,
        )
        materialized += 1
    return materialized


def sync_push(
    *,
    cfg: ArtifactsConfig,
    category_name: str,
    user_include: list[str] | None,
    user_exclude: list[str] | None,
    no_default_include: bool,
    no_default_exclude: bool,
    dry_run: bool,
    delete: bool,
) -> None:
    """Sync local files for a category up to its bucket subpath.

    `delete=True` removes files from the bucket that no longer exist locally
    — destructive, so callers expose this behind an explicit flag.
    """
    rc = resolve_category(category_name, cfg)
    if not rc.local_root.exists():
        logger.warning(
            f"sync_push: skipping category={category_name} — local root "
            f"{rc.local_root} does not exist (nothing to push)"
        )
        return
    filters = resolve_filters(
        default_include=rc.config.default_include,
        default_exclude=rc.config.default_exclude,
        user_include=user_include,
        user_exclude=user_exclude,
        no_default_include=no_default_include,
        no_default_exclude=no_default_exclude,
    )
    logger.info(
        f"sync_push: category={category_name} {rc.local_root} → {rc.remote_uri} "
        f"dry_run={dry_run} delete={delete} filters={filters}"
    )
    sync_bucket(
        str(rc.local_root),
        rc.remote_uri,
        dry_run=dry_run,
        delete=delete,
        include=filters.include if filters.include else None,
        exclude=filters.exclude if filters.exclude else None,
    )


def sync_pull(
    *,
    cfg: ArtifactsConfig,
    category_name: str,
    user_include: list[str] | None,
    user_exclude: list[str] | None,
    no_default_include: bool,
    no_default_exclude: bool,
    dry_run: bool,
) -> None:
    """Sync a category's bucket subpath down to its local root.

    Pull never deletes local files — there's no risk-justified use case for
    "delete files that no longer exist on the bucket" on the pull side.
    """
    rc = resolve_category(category_name, cfg)
    filters = resolve_filters(
        default_include=rc.config.default_include,
        default_exclude=rc.config.default_exclude,
        user_include=user_include,
        user_exclude=user_exclude,
        no_default_include=no_default_include,
        no_default_exclude=no_default_exclude,
    )
    logger.info(
        f"sync_pull: category={category_name} {rc.remote_uri} → {rc.local_root} "
        f"dry_run={dry_run} filters={filters}"
    )
    sync_bucket(
        rc.remote_uri,
        str(rc.local_root),
        dry_run=dry_run,
        include=filters.include if filters.include else None,
        exclude=filters.exclude if filters.exclude else None,
    )
    if category_name == "features" and not dry_run:
        materialized = materialize_pulled_percentile_statistics(rc.local_root)
        logger.info(
            f"sync_pull: materialized {materialized} pulled percentile statistics artifact(s)"
        )
