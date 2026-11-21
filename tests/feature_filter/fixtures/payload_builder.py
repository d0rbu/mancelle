"""Build synthetic FeatureActivationPayload objects for sampler tests."""
from steering.eval.feature_filter.neuronpedia_activations import (
    ActivationEntry,
    FeatureActivationPayload,
)


def build_entry(
    *,
    max_value: float,
    bin_min: float,
    bin_max: float,
    bin_contains: float = 0.0,
    tokens: list[str] | None = None,
) -> ActivationEntry:
    toks = tokens if tokens is not None else ["t"]
    return ActivationEntry(
        tokens=toks,
        values=[max_value] * len(toks),
        max_value=max_value,
        max_value_token_index=0,
        bin_min=bin_min,
        bin_max=bin_max,
        bin_contains=bin_contains,
    )


def build_bin_structured_payload(
    *,
    feature_id: int = 1,
    max_act: float = 10.0,
    per_bin: int = 5,
    description: str | None = "desc",
    overall_topk: int = 20,
) -> FeatureActivationPayload:
    """Produce a payload with the 6-bin Neuronpedia structure (1 sentinel
    `binMin=-1` bucket + 5 quintile bins, each with `per_bin` entries)."""
    step = max_act / 5.0
    entries: list[ActivationEntry] = []
    # Sentinel -1 bucket (overall top-K duplicate)
    for i in range(overall_topk):
        entries.append(build_entry(
            max_value=max_act - i * 0.01,
            bin_min=-1.0, bin_max=max_act,
            tokens=[f"top{i}"],
        ))
    # 5 quintiles, low -> high
    for q in range(5):
        lo, hi = q * step, (q + 1) * step
        for j in range(per_bin):
            mv = lo + (hi - lo) * (1.0 - j / max(per_bin - 1, 1)) if q > 0 else 0.0
            entries.append(build_entry(
                max_value=mv,
                bin_min=lo, bin_max=hi,
                bin_contains=0.9 if q == 0 else 10 ** (-q - 2),
                tokens=[f"b{q}_{j}"],
            ))
    return FeatureActivationPayload(
        feature_id=feature_id,
        description=description,
        entries=entries,
        freq_hist_bar_heights=[1.0] * 5,
        freq_hist_bar_values=[step * i for i in range(5)],
        frac_nonzero=0.01,
        max_act_approx=max_act,
    )


# Retained for legacy callers still constructing arbitrary payloads.
def build_payload(
    *,
    feature_id: int = 1,
    entries: list[ActivationEntry] | None = None,
    frac_nonzero: float = 0.01,
    max_act_approx: float = 10.0,
    description: str | None = "desc",
) -> FeatureActivationPayload:
    if entries is None:
        return build_bin_structured_payload(
            feature_id=feature_id, max_act=max_act_approx,
            description=description,
        )
    return FeatureActivationPayload(
        feature_id=feature_id,
        description=description,
        entries=entries,
        freq_hist_bar_heights=[1.0],
        freq_hist_bar_values=[0.0],
        frac_nonzero=frac_nonzero,
        max_act_approx=max_act_approx,
    )
