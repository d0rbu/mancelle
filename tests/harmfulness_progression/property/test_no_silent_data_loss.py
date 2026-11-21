"""Property test: render_progression_chart never silently drops a record's
metric. Multiset of metric values across all rendered lines = multiset of
input records' metric values."""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.harmfulness_progression import (
    DEFAULT_COND_SOURCE_MAP,
    DEFAULT_SIGNAL_STYLES,
    DEFAULT_STAGES,
    HarmfulnessMetric,
    HarmfulnessMetrics,
    HarmfulnessRecord,
    render_progression_chart,
)

# Small fixed alphabets so hypothesis converges fast.
SOURCES = list(DEFAULT_COND_SOURCE_MAP.keys())
SIGNALS = ["Safety delta-base", "Capability delta-base"]


@st.composite
def _record(draw) -> HarmfulnessRecord:
    return HarmfulnessRecord(
        model_tag="d12_baseline",
        source=draw(st.sampled_from(SOURCES)),
        signal=draw(st.sampled_from(SIGNALS)),
        full_conditioning=False,
        metrics=HarmfulnessMetrics(
            asr=draw(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
            ),
            mean_score=draw(
                st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False)
            ),
            safe_rate=draw(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
            ),
        ),
    )


@settings(max_examples=50, deadline=None)
@given(
    records=st.lists(_record(), min_size=1, max_size=9, unique_by=lambda r: (r.source, r.signal))
)
def test_no_silent_data_loss_in_progression_chart(records) -> None:
    fig = render_progression_chart(
        records,
        source_to_stage=DEFAULT_COND_SOURCE_MAP,
        stage_labels=list(DEFAULT_STAGES),
        signal_order=SIGNALS,
        signal_styles=DEFAULT_SIGNAL_STYLES,
        metric=HarmfulnessMetric.ASR,
        ylabel="ASR",
        title="prop",
    )
    ax = fig.axes[0]
    # Aggregate y-values across data lines (skip vlines: xdata of length 2 with same values).
    rendered = []
    for line in ax.lines:
        xs = np.asarray(line.get_xdata()).tolist()
        ys = np.asarray(line.get_ydata()).tolist()
        if len(xs) >= 1 and not (len(xs) == 2 and xs[0] == xs[1]):
            rendered.extend(round(float(y), 9) for y in ys)

    expected = [round(r.metrics.asr, 9) for r in records]
    assert Counter(rendered) == Counter(expected), (
        f"Rendered multiset {Counter(rendered)} != expected {Counter(expected)}"
    )
    plt.close(fig)
