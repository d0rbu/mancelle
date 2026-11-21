"""Keep the extraction-to-training context contract at one default."""

import inspect
from pathlib import Path

from hydra import compose, initialize_config_dir

from steering.context_defaults import DEFAULT_CONTEXT_WINDOW
from steering.extract_features import extract_document_features, process_shard

REPO_ROOT = Path(__file__).resolve().parents[2]


def _compose(config_dir: str, config_name: str):
    with initialize_config_dir(
        version_base=None,
        config_dir=str(REPO_ROOT / "configs" / config_dir),
    ):
        return compose(config_name=config_name)


def test_extraction_splice_packing_and_training_share_context_default():
    extraction = _compose("extraction", "extract_features")
    splice = _compose("extraction", "splice_feature_shards")
    packed = _compose("extraction", "precompute_packed_rows")
    training = _compose("training", "base_train")

    assert extraction.context_window == DEFAULT_CONTEXT_WINDOW
    assert splice.context_window == DEFAULT_CONTEXT_WINDOW
    assert packed.context_len == DEFAULT_CONTEXT_WINDOW
    assert training.max_seq_len == DEFAULT_CONTEXT_WINDOW


def test_feature_extraction_call_defaults_use_shared_context_contract():
    assert (
        inspect.signature(extract_document_features)
        .parameters["context_window"]
        .default
        == DEFAULT_CONTEXT_WINDOW
    )
    assert (
        inspect.signature(process_shard).parameters["context_window"].default
        == DEFAULT_CONTEXT_WINDOW
    )
