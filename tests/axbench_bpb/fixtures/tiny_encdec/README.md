# Tiny encoder-decoder checkpoint fixture

Built by `tests/axbench_bpb/fixtures/regenerate_fixtures.py::build_tiny_encdec()`
with seed 12345. Config: decoder 2-layer n_head=2 n_kv_head=2 n_embd=32 vocab=128
seq_len=64; encoder depth=2 n_head=2 n_kv_head=2 n_embd=32 head_dim=16;
cross-attention frequency=1; n_sae_features=64 num_buckets=4. Used by
`tests/axbench_bpb/functional/test_score_sample_kernel.py` via the `tiny_encdec`
conftest fixture (Task 6.5).

To regenerate:

    uv run python -m tests.axbench_bpb.fixtures.regenerate_fixtures
