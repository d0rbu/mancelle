"""Regression guards for the documented SFT loss objective."""

from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
_SFT_PIPELINE_DOCS = (
    "docs/pipelines/chat-training.md",
    "docs/pipelines/encoder-decoder-training.md",
)
_OBSOLETE_UNCONDITIONAL_WORDING = (
    "Every non-padding target token is supervised",
    "Every non-padding target token contributes",
)


def test_sft_pipeline_docs_do_not_claim_unconditional_full_sequence_loss() -> None:
    for relative_path in _SFT_PIPELINE_DOCS:
        text = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
        for obsolete_wording in _OBSOLETE_UNCONDITIONAL_WORDING:
            assert obsolete_wording not in text, (relative_path, obsolete_wording)
