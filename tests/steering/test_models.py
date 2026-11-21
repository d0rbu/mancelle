"""Tests for steering.models — shared model configuration."""

from steering.models import EXTRACTION_MODELS


class TestExtractionModels:
    def test_all_entries_have_hf_model_id(self) -> None:
        for name, cfg in EXTRACTION_MODELS.items():
            assert cfg.hf_model_id, f"{name} missing hf_model_id"

    def test_gemma_2b_config(self) -> None:
        cfg = EXTRACTION_MODELS["gemma-2-2b"]
        assert cfg.hf_model_id == "google/gemma-2-2b"
