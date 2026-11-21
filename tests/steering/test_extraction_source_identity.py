"""Actual extraction and both save paths bind documents to loaded SAE coordinates."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from steering.extraction_identity import ExtractionSourceIdentity, extraction_documents_sha256
from steering.feature_io import MetadataKey, glob_features, load_features, read_shard_metadata
from tests.steering.test_extract_features_loop import tiny_params
from tests.steering.test_extraction_backend_provenance import _run_loop


@pytest.mark.parametrize("input_depth", [0, 1])
@pytest.mark.parametrize("output_depth", [0, 1])
def test_real_extraction_binds_ordered_text_and_dictionary(tmp_path, output_depth, input_depth):
    identity = ExtractionSourceIdentity("tiny-test-model", "tiny-test-release", "tiny-test-sae")
    params = replace(tiny_params(tmp_path / "out"), extraction_source_identity=identity)
    texts = ["alpha beta", "gamma delta", "alpha beta"]
    path = tmp_path / "shard_00000.parquet"
    pq.write_table(pa.table({"text": texts}), path)
    _run_loop(tmp_path, params, [str(path)], output_depth, input_depth)
    shard = glob_features(params.output_dir)[0]
    loaded = load_features(shard)
    for key, value in identity.metadata().items():
        assert loaded[key] == value
    assert loaded[MetadataKey.N_SAE_FEATURES] == params.sae.cfg.d_sae
    assert loaded[MetadataKey.N_DOCS] == len(texts)
    assert loaded[MetadataKey.EXTRACTION_DOCUMENTS_SHA256] == extraction_documents_sha256(texts)
    assert read_shard_metadata(shard)[MetadataKey.EXTRACTION_DOCUMENTS_SHA256] == loaded[MetadataKey.EXTRACTION_DOCUMENTS_SHA256]
    before = shard.read_bytes()
    # Resume cannot backfill identity from new arguments into existing bytes.
    params = replace(params, extraction_source_identity=ExtractionSourceIdentity("other", "other", "other"))
    assert _run_loop(tmp_path, params, [str(path)], output_depth, input_depth) == []
    assert shard.read_bytes() == before


def test_document_digest_binds_boundaries_order_and_exact_text():
    sequences = [["ab", "c"], ["a", "bc"], ["c", "ab"], ["ab", "C"], ["ab", "c", ""],
                 ["é"], ["e\u0301"]]
    assert len({extraction_documents_sha256(texts) for texts in sequences}) == len(sequences)
