"""Live HF-folder enumeration: catch a new L0 tier appearing upstream.

Skipped by default because it downloads via the HF API. Run manually:
    uv run pytest tests/steering/test_v2_schema_drift.py -m very_slow -v
"""

import re

import pytest
from huggingface_hub import HfApi

from steering.sae_resolver import L0Tier

pytestmark = pytest.mark.very_slow


def test_270m_it_tiers_match_l0_tier_enum():
    api = HfApi()
    files = api.list_repo_files("google/gemma-scope-2-270m-it")
    pat = re.compile(
        r"^(resid_post|mlp_out|attn_out)/layer_\d+_width_\w+_l0_(?P<tier>\w+)/"
    )
    seen: set[str] = set()
    for f in files:
        m = pat.match(f)
        if m:
            seen.add(m.group("tier"))
    known = {t.value for t in L0Tier}
    new = seen - known
    assert not new, (
        f"New L0 tier(s) observed on HuggingFace that are not in L0Tier: "
        f"{sorted(new)}. Add them to L0Tier and update the validation table."
    )
