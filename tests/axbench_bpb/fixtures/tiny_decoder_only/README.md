# Tiny decoder-only checkpoint fixture

Built by `tests/axbench_bpb/fixtures/regenerate_fixtures.py::build_tiny_decoder_only()`
with seed 12345. Config: 2-layer, n_head=2, n_kv_head=2, n_embd=32, vocab=128,
seq_len=64. Used by `tests/axbench_bpb/functional/test_score_sample_kernel.py`
via the `tiny_decoder_only` conftest fixture (Task 6.5).

To regenerate:

    uv run python -m tests.axbench_bpb.fixtures.regenerate_fixtures
