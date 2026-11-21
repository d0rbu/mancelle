"""Shared strict-JSON helpers whose former copies were proven identical.

The following definitions are deliberately different and must not be folded
into this module without a per-site equivalence proof:

- ``steering/eval/autointerp_filter/strict_json.py:canonical_json``
  (adds ``allow_nan=False``)
- ``steering/eval/autointerp_filter/strict_json.py:reject_json_constant``
  (different error message)
- ``steering/eval/axbench_base/stage0c_api_generation.py`` (adds
  ``allow_nan=False`` and wraps errors as ``AxBenchBaseError``)
- ``steering/eval/cluster_contract_identity.py`` (pre-validates string keys and
  finite floats and documents a deliberate ``-0.0`` policy)
- ``steering/conditional/feature_clusters.py:FeatureClusterConsolidation.canonical_json``
  (a method that calls the module helper)
- ``steering/eval/splice_selection_certificate.py`` (omits
  ``ensure_ascii=False``)
- ``steering/eval/axbench_bpb_adv/artifact_identity.py``;
  ``steering/eval/axbench_bpb_adv/universal_gcg_checkpoint.py``;
  ``steering/eval/autointerp_filter/strict_json.py:canonical_sha256``;
  ``steering/remote_training.py:_strict_sha256`` (SHA-256 byte/hash variants)
- ``steering/safetensors_layout.py:_unique_json_object`` (different
  signature and error)
- ``steering/eval/axbench_base/synth/positive_checkpoint.py:_canonical_json_bytes``;
  ``steering/eval/cluster_contract_identity.py:_canonical_json`` (domain
  validators with different error classes and traversal)
- Every JSON loader (each has distinct error, validation, and fd-binding
  behavior)
- ``steering/eval/autointerp_filter/strict_json.py:canonical_json_equal`` and
  ``steering/remote_training.py:_strict_json_value_v2`` (unique)

Changing ``sort_keys``, ``separators``, or ``ensure_ascii`` here silently changes
every artifact digest derived through this shared serializer.
"""

import json
from collections.abc import Mapping
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting ambiguous duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def string_keyed_mapping(value: object) -> dict[str, object] | None:
    """Return a plain copy when ``value`` is a mapping with only string keys.

    JSON objects always have string keys, but callers of public in-memory parsers
    can supply arbitrary ``Mapping`` implementations. Normalizing once at that
    boundary gives downstream validators an honest key type without an unchecked
    cast. Values deliberately remain opaque until their schema-specific parser
    validates them.
    """
    if not isinstance(value, Mapping):
        return None
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        result[key] = item
    return result


def reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def canonical_json_str_fallback(payload: Any) -> str:
    """Canonical JSON that stringifies otherwise unserializable values.

    This intentionally unlovely name distinguishes the ``default=str`` and
    default-ASCII behavior from :func:`canonical_json`; the two serializers
    produce different bytes for the same input and must never be merged.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
