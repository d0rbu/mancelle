"""Neuronpedia activation fetch + per-feature disk cache."""
from __future__ import annotations

import json as _json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
import yaml
from loguru import logger

from errors import DataLoadError
from steering.eval.feature_filter.types import SCHEMA_VERSION, InvalidHistogramError
from steering.eval.neuronpedia_client import (
    NEURONPEDIA_API_KEY_ENV,
    NEURONPEDIA_BASE_URL,
    NeuronpediaAuthError,
    NeuronpediaFetchError,
    NeuronpediaNotPublishedError,
    NeuronpediaRateLimitError,
    _parse_retry_after,
    _require_api_key,
    _sleep_for_retry_after,
    _sleep_with_backoff,
)
from steering.eval.neuronpedia_config import NeuronpediaConfig

__all__ = [
    "ActivationEntry",
    "FeatureActivationPayload",
    "NeuronpediaActivationCache",
    "fetch_feature_activations",
]


@dataclass(frozen=True)
class ActivationEntry:
    tokens: list[str]
    values: list[float]
    max_value: float
    max_value_token_index: int
    bin_min: float
    bin_max: float
    bin_contains: float


@dataclass(frozen=True)
class FeatureActivationPayload:
    feature_id: int
    description: str | None
    entries: list[ActivationEntry]
    freq_hist_bar_heights: list[float]
    freq_hist_bar_values: list[float]
    frac_nonzero: float
    max_act_approx: float


def _finite_float(value: object, field: str, feature_id: int) -> float:
    if value is None:
        raise NeuronpediaFetchError(
            f"feature {feature_id}: field {field!r} is None"
        )
    if not isinstance(value, (str, int, float)):
        raise NeuronpediaFetchError(
            f"feature {feature_id}: field {field!r} not a float "
            f"(got {type(value).__name__})"
        )
    try:
        f = float(value)
    except ValueError as exc:
        raise NeuronpediaFetchError(
            f"feature {feature_id}: field {field!r} not a float (got {type(value).__name__})"
        ) from exc
    if math.isnan(f) or math.isinf(f):
        raise NeuronpediaFetchError(
            f"feature {feature_id}: field {field!r} is NaN/inf ({f!r})"
        )
    return f


def _parse_activation_entry(raw: dict, feature_id: int) -> ActivationEntry:
    tokens_raw = raw.get("tokens") or []
    values_raw = raw.get("values") or []
    if len(tokens_raw) != len(values_raw):
        raise NeuronpediaFetchError(
            f"feature {feature_id}: tokens (len={len(tokens_raw)}) != values (len={len(values_raw)})"
        )
    tokens = [str(t) for t in tokens_raw]
    values = [_finite_float(v, "values[i]", feature_id) for v in values_raw]
    max_value = _finite_float(raw.get("maxValue", 0.0), "maxValue", feature_id)
    max_idx = int(raw.get("maxValueTokenIndex", 0))
    if tokens:
        max_idx = max(0, min(max_idx, len(tokens) - 1))
    else:
        max_idx = 0
    return ActivationEntry(
        tokens=tokens,
        values=values,
        max_value=max_value,
        max_value_token_index=max_idx,
        bin_min=_finite_float(raw.get("binMin", 0.0), "binMin", feature_id),
        bin_max=_finite_float(raw.get("binMax", 0.0), "binMax", feature_id),
        bin_contains=_finite_float(raw.get("binContains", 0.0), "binContains", feature_id),
    )


def _parse_payload(payload: object, feature_id: int) -> FeatureActivationPayload:
    if not isinstance(payload, dict):
        raise NeuronpediaFetchError(
            f"feature {feature_id}: response is not a dict ({type(payload).__name__})"
        )
    raw_acts = payload.get("activations") or []
    if not isinstance(raw_acts, list):
        raise NeuronpediaFetchError(
            f"feature {feature_id}: 'activations' is not a list ({type(raw_acts).__name__})"
        )
    entries: list[ActivationEntry] = []
    for raw in raw_acts:
        if not isinstance(raw, dict):
            continue
        entries.append(_parse_activation_entry(raw, feature_id))

    bar_heights_raw = payload.get("freq_hist_data_bar_heights") or []
    bar_values_raw = payload.get("freq_hist_data_bar_values") or []
    if not isinstance(bar_heights_raw, list) or not isinstance(bar_values_raw, list):
        raise InvalidHistogramError(
            f"feature {feature_id}: freq_hist_data_bar_* not lists"
        )
    if len(bar_heights_raw) != len(bar_values_raw):
        raise InvalidHistogramError(
            f"feature {feature_id}: bar_heights len={len(bar_heights_raw)} "
            f"!= bar_values len={len(bar_values_raw)}"
        )
    bar_heights = [_finite_float(h, "freq_hist_data_bar_heights", feature_id) for h in bar_heights_raw]
    bar_values = [_finite_float(v, "freq_hist_data_bar_values", feature_id) for v in bar_values_raw]
    for h in bar_heights:
        if h < 0:
            raise InvalidHistogramError(f"feature {feature_id}: negative bin height {h}")

    frac_nonzero = _finite_float(payload.get("frac_nonzero", 0.0), "frac_nonzero", feature_id)
    if not 0.0 <= frac_nonzero <= 1.0:
        raise NeuronpediaFetchError(
            f"feature {feature_id}: frac_nonzero {frac_nonzero} outside [0, 1]"
        )

    description: str | None = None
    explanations = payload.get("explanations") or []
    if isinstance(explanations, list) and explanations:
        first = explanations[0]
        if isinstance(first, dict):
            d = first.get("description")
            if isinstance(d, str) and d:
                description = d

    return FeatureActivationPayload(
        feature_id=feature_id,
        description=description,
        entries=entries,
        freq_hist_bar_heights=bar_heights,
        freq_hist_bar_values=bar_values,
        frac_nonzero=frac_nonzero,
        max_act_approx=_finite_float(payload.get("maxActApprox", 0.0), "maxActApprox", feature_id),
    )


def fetch_feature_activations(
    model_id: str,
    sae_source: str,
    feature_id: int,
    *,
    config: NeuronpediaConfig | None = None,
) -> FeatureActivationPayload:
    """Fetch a single feature's full activation payload from Neuronpedia."""
    cfg = config or NeuronpediaConfig()
    api_key = _require_api_key()
    url = f"{NEURONPEDIA_BASE_URL}/feature/{model_id}/{sae_source}/{feature_id}"
    headers = {"X-Api-Key": api_key, "Accept-Encoding": "gzip"}
    last_retry_after: float | None = None
    for attempt in range(cfg.max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=cfg.timeout_seconds)
        except requests.RequestException as exc:
            if attempt < cfg.max_retries:
                _sleep_with_backoff(attempt, cfg)
                continue
            raise NeuronpediaFetchError(
                f"Network error after {attempt + 1} attempts for feature {feature_id}: {exc}"
            ) from exc
        status = resp.status_code
        if status is None:
            raise NeuronpediaFetchError(
                f"Neuronpedia returned no HTTP status code for feature {feature_id}"
            )
        if status == 200:
            try:
                body = resp.json()
            except (ValueError, requests.JSONDecodeError) as exc:
                raise NeuronpediaFetchError(
                    f"Malformed JSON for feature {feature_id}: {resp.text[:200]!r}"
                ) from exc
            return _parse_payload(body, feature_id)
        if status == 401:
            raise NeuronpediaAuthError(
                f"Neuronpedia returned 401 for feature {feature_id}; check {NEURONPEDIA_API_KEY_ENV}."
            )
        if status == 429:
            last_retry_after = _parse_retry_after(
                resp.headers.get("Retry-After"), cap_seconds=cfg.backoff_max_seconds,
            )
            if attempt < cfg.max_retries:
                _sleep_for_retry_after(last_retry_after, attempt, cfg)
                continue
            raise NeuronpediaRateLimitError(
                f"Neuronpedia rate-limited feature {feature_id} after {attempt + 1} attempts",
                retry_after_seconds=last_retry_after,
            )
        if 500 <= status < 600:
            if attempt < cfg.max_retries:
                _sleep_with_backoff(attempt, cfg)
                continue
            raise NeuronpediaFetchError(
                f"Neuronpedia returned {status} for feature {feature_id} after "
                f"{attempt + 1} attempts: {resp.text[:200]!r}"
            )
        if status == 404:
            # Typed signal that Neuronpedia does not host this feature/SAE
            # yet (expected for gemma-scope-2 releases it hasn't ingested).
            # Load-bearing invariant: this 404 check MUST precede the generic
            # non-retryable 4xx raise below so the typed error wins and the
            # runner can skip gracefully. (Branch ORDER differs from
            # neuronpedia_client.fetch_feature_description, which checks 404
            # first; both are correct since HTTP statuses are mutually
            # exclusive.) 404 is non-retryable — single request, no backoff.
            # Body is user-opaque server text — log status + url + length
            # only, matching this module's 4xx/5xx privacy convention.
            body_len = len(resp.text or "")
            logger.warning(
                f"Neuronpedia 404: status=404 url={url} body_len={body_len}"
            )
            raise NeuronpediaNotPublishedError(
                f"Neuronpedia does not host activations for feature "
                f"{feature_id} ({model_id}/{sae_source}); expected for "
                f"un-ingested gemma-scope-2 SAEs. (body suppressed for "
                f"privacy; see logs for status + url + length)"
            )
        raise NeuronpediaFetchError(
            f"Neuronpedia returned {status} for feature {feature_id}: {resp.text[:200]!r}"
        )
    raise NeuronpediaFetchError(
        f"Internal: retry loop exited without return for feature {feature_id}"
    )


class NeuronpediaActivationCache:
    """Disk-backed cache for Neuronpedia feature activation payloads.

    Directory layout::

        <cache_dir>/
            manifest.yaml          # model_id, sae_source, schema_version
            features/
                f_000042.json      # one file per feature, atomically written
    """

    _MANIFEST_NAME = "manifest.yaml"
    _FEATURES_DIR = "features"

    def __init__(
        self,
        model_id: str,
        sae_source: str,
        cache_dir: Path | str,
        *,
        config: NeuronpediaConfig | None = None,
    ) -> None:
        self._model_id = model_id
        self._sae_source = sae_source
        self._cache_dir = Path(cache_dir)
        self._config = config or NeuronpediaConfig()
        self._features_dir = self._cache_dir / self._FEATURES_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._features_dir.mkdir(parents=True, exist_ok=True)
        self._init_manifest()

    def _manifest_path(self) -> Path:
        return self._cache_dir / self._MANIFEST_NAME

    def _init_manifest(self) -> None:
        manifest_path = self._manifest_path()
        if manifest_path.is_file():
            with manifest_path.open() as fh:
                existing = yaml.safe_load(fh) or {}
            if (
                existing.get("model_id") != self._model_id
                or existing.get("sae_source") != self._sae_source
            ):
                raise DataLoadError(
                    f"Cache manifest mismatch: expected model_id={self._model_id!r} "
                    f"sae_source={self._sae_source!r}, got "
                    f"model_id={existing.get('model_id')!r} "
                    f"sae_source={existing.get('sae_source')!r}"
                )
            if existing.get("schema_version") != SCHEMA_VERSION:
                raise DataLoadError(
                    f"Cache manifest schema_version {existing.get('schema_version')} "
                    f"!= {SCHEMA_VERSION}"
                )
        else:
            manifest = {
                "model_id": self._model_id,
                "sae_source": self._sae_source,
                "schema_version": SCHEMA_VERSION,
            }
            with manifest_path.open("w") as fh:
                yaml.safe_dump(manifest, fh)

    def _feature_path(self, feature_id: int) -> Path:
        return self._features_dir / f"f_{feature_id:06d}.json"

    def get_cached(self, feature_id: int) -> FeatureActivationPayload | None:
        """Return the cached payload for *feature_id*, or None on miss/corrupt."""
        path = self._feature_path(feature_id)
        if not path.is_file():
            return None
        try:
            data = _json.loads(path.read_text())
            entries = [ActivationEntry(**e) for e in data.get("entries", [])]
            return FeatureActivationPayload(
                feature_id=data["feature_id"],
                description=data.get("description"),
                entries=entries,
                freq_hist_bar_heights=data["freq_hist_bar_heights"],
                freq_hist_bar_values=data["freq_hist_bar_values"],
                frac_nonzero=data["frac_nonzero"],
                max_act_approx=data["max_act_approx"],
            )
        except Exception as exc:
            logger.warning(
                "Corrupt cache entry for feature {fid} at {path}: {exc}",
                fid=feature_id,
                path=path,
                exc=exc,
            )
            path.unlink(missing_ok=True)
            return None

    def get(self, feature_id: int) -> FeatureActivationPayload:
        """Return the payload for *feature_id*, fetching from Neuronpedia on cache miss."""
        cached = self.get_cached(feature_id)
        if cached is not None:
            return cached

        payload = fetch_feature_activations(
            self._model_id,
            self._sae_source,
            feature_id,
            config=self._config,
        )
        self._write_cached(feature_id, payload)
        return payload

    def _write_cached(self, feature_id: int, payload: FeatureActivationPayload) -> None:
        """Atomically write *payload* to the feature cache file."""
        target = self._feature_path(feature_id)
        tmp = target.with_suffix(".tmp")
        data = asdict(payload)
        tmp.write_text(_json.dumps(data))
        tmp.replace(target)

    def get_many(self, feature_ids: list[int]) -> dict[int, FeatureActivationPayload]:
        """Fetch multiple features concurrently, returning a mapping.

        Partial-result semantics: a feature Neuronpedia does not host
        (``NeuronpediaNotPublishedError``, i.e. HTTP 404 — expected for
        un-ingested gemma-scope-2 SAEs) is **skipped and omitted** from the
        returned dict, mirroring ``neuronpedia_client.get_descriptions``.
        Any other exception (auth/rate-limit/fetch/5xx) still propagates.
        Not used by the runner today (it uses singular ``.get()``); this
        is defense-in-depth for direct callers.

        Note: not-published (404) results are NOT negatively cached (same
        as the description cache) — each call re-hits the network for an
        un-ingested SAE. The feature-filter runner deduplicates instead at
        the JSONL/resume layer (a ``not_published`` row is persisted and
        skipped on resume), so repeated network hits do not occur there.
        """
        out: dict[int, FeatureActivationPayload] = {}
        missing: list[int] = []
        for fid in feature_ids:
            cached = self.get_cached(fid)
            if cached is not None:
                out[fid] = cached
            else:
                missing.append(fid)
        if not missing:
            return out

        def _one(fid: int) -> tuple[int, FeatureActivationPayload]:
            return fid, self.get(fid)

        with ThreadPoolExecutor(
            max_workers=self._config.max_concurrent, thread_name_prefix="np_cache",
        ) as ex:
            futures = [ex.submit(_one, fid) for fid in missing]
            for fut in as_completed(futures):
                try:
                    fid, payload = fut.result()
                except NeuronpediaNotPublishedError as exc:
                    logger.warning(
                        f"get_many: skipping not-published feature: {exc}"
                    )
                    continue
                out[fid] = payload
        return out
