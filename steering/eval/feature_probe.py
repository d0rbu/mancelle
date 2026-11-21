# steering/eval/feature_probe.py
"""SAE feature probe for completion steering eval.

Computes raw (pre-activation) or SAE-processed feature activations
for generated completions, enabling objective measurement of whether
feature conditioning causally affects model outputs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import matplotlib
import torch as t
from loguru import logger
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from steering.eval.multiseg_generator import DocumentCompletion
from steering.eval.probe_activations import build_probe_models, get_probe_activations
from steering.feature_paths import parse_feature_dir
from steering.sae_resolver import SAERelease, resolve_sae
from steering.segmentation import token_to_char_offsets

__all__ = [
    "ProbeResults",
    "apply_sae_activation",
    "apply_sae_activation_batched",
    "build_completion_mask",
    "compute_probe_for_completions",
    "compute_raw_probe",
    "compute_raw_probe_batched",
    "compute_sae_probe",
    "group_probe_results",
    "max_pool_masked",
    "max_pool_masked_batched",
    "plot_probe_ablation",
    "plot_probe_dose_response",
]


@dataclass(frozen=True)
class ProbeResults:
    """All probe scores for one feature, grouped by condition."""

    feature_id: int
    condition_scores: dict[str, list[float]]


class EncodableSAE(Protocol):
    """Minimal SAE surface needed by the post-activation probe."""

    def encode(self, activations: t.Tensor, /) -> t.Tensor: ...


def compute_raw_probe(
    act: t.Tensor,
    W_enc: t.Tensor,
    b_enc: t.Tensor,
    b_dec: t.Tensor,
    feature_id: int,
    *,
    apply_b_dec: bool = True,
) -> t.Tensor:
    """Compute the pre-activation linear probe for a single SAE feature.

    Args:
        act: Activations from the base model, shape ``[seq_len, d_model]``.
        W_enc: SAE encoder weight, shape ``[d_model, n_features]``.
        b_enc: SAE encoder bias, shape ``[n_features]``.
        b_dec: SAE decoder bias, shape ``[d_model]``.
        feature_id: Which SAE feature to probe.
        apply_b_dec: Whether to subtract ``b_dec`` before the linear transform
            (must match ``sae.cfg.apply_b_dec_to_input``).

    Returns:
        Tensor of shape ``[seq_len]`` — raw pre-activation probe values.
    """
    sae_in = act - b_dec if apply_b_dec else act
    return sae_in @ W_enc[:, feature_id] + b_enc[feature_id]


def compute_raw_probe_batched(
    act: t.Tensor,  # (seq_len, d_model), float
    W_enc: t.Tensor,  # (d_model, n_features)
    b_enc: t.Tensor,  # (n_features,)
    b_dec: t.Tensor,  # (d_model,)
    feature_ids: t.Tensor,  # (n_feats,) int64
    *,
    apply_b_dec: bool = True,
) -> t.Tensor:
    """Vectorized pre-activation probe for multiple SAE features at once.

    Equivalent to stacking :func:`compute_raw_probe` results for each
    ``feature_id`` in ``feature_ids``, but implemented as a single matmul
    over ``W_enc[:, feature_ids]``.

    Args:
        act: Activations from the base model, shape ``[seq_len, d_model]``.
        W_enc: SAE encoder weight, shape ``[d_model, n_features]``.
        b_enc: SAE encoder bias, shape ``[n_features]``.
        b_dec: SAE decoder bias, shape ``[d_model]``.
        feature_ids: Which SAE features to probe, int64 tensor of shape ``[n_feats]``.
        apply_b_dec: Whether to subtract ``b_dec`` before the linear transform
            (must match ``sae.cfg.apply_b_dec_to_input``).

    Returns:
        Tensor of shape ``[seq_len, n_feats]`` — raw pre-activation probe values,
        column ``i`` equals ``compute_raw_probe(act, W_enc, b_enc, b_dec,
        int(feature_ids[i]), apply_b_dec=apply_b_dec)``.
    """
    sae_in = act - b_dec if apply_b_dec else act
    return sae_in @ W_enc[:, feature_ids] + b_enc[feature_ids]


def apply_sae_activation(
    pre_act: t.Tensor,
    sae: object,
    feature_id: int,
) -> t.Tensor:
    """Apply the SAE's activation function to pre-activation probe values.

    Detects the SAE's nonlinearity and applies it element-wise:

    - **JumpReLU** (``sae.threshold`` exists): ``relu(pre_act) * (pre_act > threshold[fid])``
    - **ReLU** (no threshold): ``relu(pre_act)``

    This is equivalent to ``sae.encode(act)[:, feature_id]`` but avoids
    computing all features.

    Args:
        pre_act: Pre-activation probe values, shape ``[seq_len]``.
        sae: SAELens SAE object.
        feature_id: Which SAE feature's activation function to apply.

    Returns:
        Post-activation tensor, shape ``[seq_len]``, values >= 0.

    Raises:
        ValueError: If the SAE uses an unsupported activation function
            (e.g., TopK) that cannot be decomposed per-feature.
    """
    threshold_tensor = getattr(sae, "threshold", None)
    if threshold_tensor is not None:
        # JumpReLU: ReLU then gate by learned per-feature threshold
        threshold = threshold_tensor[feature_id]
        return t.relu(pre_act) * (pre_act > threshold).float()

    # Check for unsupported activation types before falling back to ReLU
    activation_fn = getattr(getattr(sae, "cfg", None), "activation_fn", None)
    if activation_fn is not None and activation_fn not in ("relu", "ReLU"):
        raise ValueError(
            f"Unsupported SAE activation function '{activation_fn}' for "
            f"per-feature post-activation probe. Use compute_sae_probe() "
            f"(full encode) instead."
        )

    # ReLU fallback
    return t.relu(pre_act)


def apply_sae_activation_batched(
    pre_act: t.Tensor,  # (seq_len, n_feats)
    sae: object,
    feature_ids: t.Tensor,  # (n_feats,) int64
) -> t.Tensor:
    """Vectorized nonlinearity gate for multiple SAE features at once.

    Matches :func:`apply_sae_activation` per-feature semantics column-by-column:

    - **JumpReLU** (``sae.threshold`` exists): ``relu(x) * (x > threshold[fids])``
    - **ReLU** (no threshold): ``relu(x)``
    - Unsupported activation: raises ``ValueError`` with the same message as
      the scalar version.

    Args:
        pre_act: Pre-activation probe values, shape ``[seq_len, n_feats]``.
        sae: SAELens SAE object.
        feature_ids: Int64 tensor of shape ``[n_feats]`` — the feature IDs
            corresponding to the columns of ``pre_act``. Used only when the
            SAE has a per-feature threshold (JumpReLU).

    Returns:
        Post-activation tensor of shape ``[seq_len, n_feats]``.

    Raises:
        ValueError: If the SAE uses an unsupported activation function
            (e.g., TopK). The error message matches the scalar version.
    """
    threshold_tensor = getattr(sae, "threshold", None)
    if threshold_tensor is not None:
        # JumpReLU: ReLU then gate by learned per-feature threshold
        thresholds = threshold_tensor[feature_ids]  # (n_feats,)
        return t.relu(pre_act) * (pre_act > thresholds).float()

    # Check for unsupported activation types before falling back to ReLU.
    # Error message must match the scalar version character-for-character.
    activation_fn = getattr(getattr(sae, "cfg", None), "activation_fn", None)
    if activation_fn is not None and activation_fn not in ("relu", "ReLU"):
        raise ValueError(
            f"Unsupported SAE activation function '{activation_fn}' for "
            f"per-feature post-activation probe. Use compute_sae_probe() "
            f"(full encode) instead."
        )

    # ReLU fallback
    return t.relu(pre_act)


def compute_sae_probe(
    act: t.Tensor,
    sae: EncodableSAE,
    feature_id: int,
) -> t.Tensor:
    """Compute SAE-processed probe for a single feature (with nonlinearity).

    Uses ``sae.encode()`` which applies the SAE's activation function
    (JumpReLU for GemmaScope). Values below the learned threshold are zeroed.

    Args:
        act: Activations, shape ``[seq_len, d_model]``.
        sae: SAELens SAE object with ``.encode()`` method.
        feature_id: Which feature to extract.

    Returns:
        Tensor of shape ``[seq_len]`` — post-activation probe values (>= 0).
    """
    all_features = sae.encode(act)
    return all_features[:, feature_id]


def build_completion_mask(
    char_offsets: list[int],
    completion_char_start: int,
) -> t.Tensor:
    """Build a boolean mask identifying which tokens belong to the completion.

    Args:
        char_offsets: Character-level token boundaries from
            ``token_to_char_offsets()``.  Length ``n_tokens + 1``.
        completion_char_start: Character offset where the completion begins
            (typically ``len(pre_segment_text)``).

    Returns:
        Bool tensor of shape ``[n_tokens]``.  ``True`` if the token's
        character start position is >= ``completion_char_start``.
    """
    offsets = t.tensor(char_offsets[:-1], dtype=t.long)
    return offsets >= completion_char_start


def max_pool_masked(probe: t.Tensor, mask: t.Tensor) -> float:
    """Max-pool probe values over masked (completion) positions.

    Args:
        probe: Per-token probe values, shape ``[seq_len]``.
        mask: Bool tensor of shape ``[seq_len]``.

    Returns:
        Maximum probe value among True positions, or ``float('nan')``
        if the mask is all-False.
    """
    selected = probe[mask]
    if selected.numel() == 0:
        logger.warning("Completion mask is empty — no tokens in completion span")
        return float("nan")
    return selected.max().item()


def max_pool_masked_batched(
    probes: t.Tensor,  # (seq_len, n_feats)
    mask: t.Tensor,  # (seq_len,)
) -> t.Tensor:
    """Vectorized masked max over the completion span for N features at once.

    Semantic parity with :func:`max_pool_masked` per column:

    - If the mask is all-False (no tokens in the completion span), returns
      a tensor of NaN values with shape ``(n_feats,)`` and logs one warning.
    - Otherwise, returns ``probes[mask].max(dim=0).values`` — a finite
      per-feature max.

    Args:
        probes: Per-token probe values, shape ``[seq_len, n_feats]``.
        mask: Bool tensor of shape ``[seq_len]``.

    Returns:
        Tensor of shape ``[n_feats]`` on the same device and dtype as ``probes``.
    """
    if not mask.any():
        logger.warning("Completion mask is empty — no tokens in completion span")
        return t.full(
            (probes.shape[1],),
            float("nan"),
            dtype=probes.dtype,
            device=probes.device,
        )
    selected = probes[mask]  # (n_masked, n_feats)
    return selected.max(dim=0).values


def group_probe_results(
    raw_scores: list[tuple[int, str, float]],
) -> dict[int, ProbeResults]:
    """Group (feature_id, condition, value) triples into ProbeResults.

    Args:
        raw_scores: List of ``(feature_id, condition, value)`` tuples.

    Returns:
        Dict mapping feature_id to :class:`ProbeResults`.
    """
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for fid, cond, val in raw_scores:
        grouped[fid][cond].append(val)
    return {
        fid: ProbeResults(feature_id=fid, condition_scores=dict(cond_scores))
        for fid, cond_scores in grouped.items()
    }


def compute_probe_for_completions(
    completions: list[DocumentCompletion],
    feature_dir: Path,
    *,
    use_raw_activation: bool = True,
    device: t.device,
) -> dict[int, ProbeResults]:
    """Run the full feature probe pipeline on generated completions.

    Loads the base model + SAE (auto-derived from *feature_dir*), runs each
    completion through the model, computes the probe for the target feature,
    max-pools over completion tokens, and groups results by feature and condition.

    Args:
        completions: From the GENERATE stage.
        feature_dir: Feature directory — parsed to derive SAE config.
        use_raw_activation: ``True`` for pre-activation probe (no nonlinearity),
            ``False`` for standard SAE encode (JumpReLU).
        device: Torch device.

    Returns:
        Dict mapping feature_id to :class:`ProbeResults`.
    """
    dir_info = parse_feature_dir(str(feature_dir))
    layer = dir_info.layer
    activation_type = dir_info.activation_type
    width = dir_info.width
    model_name = dir_info.model

    resolved = resolve_sae(
        model=model_name,
        release=SAERelease.gemmascope,
        activation_type=activation_type,
        layer=layer,
        width=width,
    )
    logger.info(
        f"Resolved SAE: release={resolved.release}, id={resolved.sae_id}, hook={resolved.hook_name}"
    )

    base_model, sae, gemma_tokenizer, nnterp_model = build_probe_models(
        model_name=model_name,
        layer=layer,
        device=str(device),
        resolved=resolved,
    )
    logger.info(f"Loaded base model ({model_name}) and SAE on {device}")

    hook_name = resolved.hook_name
    apply_b_dec = bool(sae.cfg.apply_b_dec_to_input)
    W_enc = sae.W_enc
    b_enc = sae.b_enc
    b_dec = sae.b_dec
    if (
        not isinstance(W_enc, t.Tensor)
        or not isinstance(b_enc, t.Tensor)
        or not isinstance(b_dec, t.Tensor)
    ):
        raise TypeError("Feature probing requires tensor-valued SAE W_enc, b_enc, and b_dec")

    raw_scores: list[tuple[int, str, float]] = []
    total_samples = sum(len(c.samples) for c in completions)
    pbar = tqdm(total=total_samples, desc="Probing", unit="sample")

    for comp in completions:
        feature_id = comp.feature_id
        pre_text = comp.pre_segment_text
        completion_char_start = len(pre_text)

        for sample in comp.samples:
            full_text = pre_text + sample.text
            tokens = gemma_tokenizer.encode(full_text)

            char_offsets = token_to_char_offsets(gemma_tokenizer, tokens)
            mask = build_completion_mask(char_offsets, completion_char_start)

            tokens_tensor = t.tensor([tokens], device=device)
            with t.no_grad():
                act = get_probe_activations(
                    base_model,
                    tokens_tensor,
                    layer,
                    hook_name=hook_name,
                    activation_type=activation_type,
                    nnterp_model=nnterp_model,
                )
            act = act.squeeze(0).float()

            if use_raw_activation:
                probe = compute_raw_probe(
                    act,
                    W_enc,
                    b_enc,
                    b_dec,
                    feature_id,
                    apply_b_dec=apply_b_dec,
                )
            else:
                probe = compute_sae_probe(act, sae, feature_id)

            value = max_pool_masked(probe, mask)
            raw_scores.append((feature_id, str(sample.condition), value))
            pbar.update(1)

    pbar.close()

    del base_model, sae, nnterp_model
    if device.type == "cuda":
        t.cuda.empty_cache()
    logger.info("Freed base model + SAE from GPU")

    return group_probe_results(raw_scores)


def plot_probe_ablation(
    real_mean: float,
    ablated_mean: float,
    feature_id: int,
    feature_description: str,
    output_path: Path | str,
    real_stderr: float = 0.0,
    ablated_stderr: float = 0.0,
) -> None:
    """Bar chart comparing real vs ablated probe values."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Real", "Ablated"]
    means = [real_mean, ablated_mean]
    stderrs = [real_stderr, ablated_stderr]

    fig, ax = plt.subplots()
    x = list(range(len(labels)))
    ax.bar(x, means, yerr=stderrs, capsize=4, align="center")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Probe value")
    title_desc = feature_description[:60]
    ax.set_title(f"Feature {feature_id}: {title_desc}")

    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved probe ablation plot to {output_path}")


def plot_probe_dose_response(
    bucket_means: list[float],
    bucket_stderrs: list[float],
    feature_id: int,
    feature_description: str,
    output_path: Path | str,
    real_baseline: float | None = None,
) -> None:
    """Line chart: bucket index vs probe value with error bars."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = list(range(len(bucket_means)))

    fig, ax = plt.subplots()
    ax.errorbar(
        x,
        bucket_means,
        yerr=bucket_stderrs,
        marker="o",
        capsize=4,
        label="Probe value",
    )
    if real_baseline is not None:
        ax.axhline(
            real_baseline,
            linestyle=":",
            color="blue",
            label="Real conditioning baseline",
        )

    ax.set_xlabel("Bucket index")
    ax.set_ylabel("Probe value")
    title_desc = feature_description[:60]
    ax.set_title(f"Feature {feature_id}: {title_desc}")
    ax.legend()

    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved probe dose-response plot to {output_path}")
