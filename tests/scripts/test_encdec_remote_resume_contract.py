from __future__ import annotations

import inspect

from scripts.train.conditional_encdec_train import (
    build_shared_state,
    train_one_config,
)


def test_trainer_wires_v2_semantics_lineage_and_rebatch_metadata():
    build_source = inspect.getsource(build_shared_state)
    train_source = inspect.getsource(train_one_config)

    assert "build_remote_semantic_fingerprint_v2(" in build_source
    assert "remote_resolved_training_config=" in build_source
    assert "remote_selection_contracts=" in build_source
    assert "semantic_fingerprint_v2=shared.remote_run_fingerprint" in train_source
    assert 'dataloader_resume_state.setdefault(\n                "device_batch_size"' in train_source
    assert "current_total_batch_size % world_microbatch_tokens" in train_source
    assert "build_remote_semantic_identity(" in train_source
    assert "_save_meta[REMOTE_RUN_META_KEY] = checkpoint_lineage" in train_source
