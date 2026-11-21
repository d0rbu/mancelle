# Native SM12 FA4 SFT evaluation dtype fix

PR: https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1725

Branch: `fix/sm12-fa4-eval-dtype`, targeting `codex/issue-1719-transformer-head`.
Fix commit: `148edbc8597271ffbef5583cdd0a76337e954989`.
The branch was created with `git checkout -b` from the actual initial HEAD,
`5780fdf64b581dad99c5ba5c01521e29768a17ac`, a fixture-only successor to the
supplied `9cb4b9418`. GitHub and origin both confirmed that starting revision
as PR #1724's current head. Commit, push, and PR creation succeeded.

`AGENTS.md` and the relevant referenced documentation were read. `CLAUDE.md`
was absent from this worktree and its ancestor directories. All Python tools
ran through `uv run`; no vendored files were edited and no whole-repo ty run
was performed.

## Root cause

- `scripts/train/conditional_encdec_sft.py:3903` creates a CUDA BF16 autocast
  context. Both validation at `:5110` and training at `:5299` use it.
  `evaluate_encdec_bpb` (`:627`, model call at `:688`) runs without gradients
  and requests LM loss without feature targets, so the feature head is not
  reached in this failure.
- `nanochat/common.py:382` resolves `NANOCHAT_DTYPE` into `COMPUTE_DTYPE`.
  `steering/conditional/encoder_decoder/model.py:431` casts RoPE buffers to
  that dtype, and `:889,921` cast the decoder activations. Value embeddings
  are also cast to the residual activation dtype at `:924`. These casts do
  not disable autocast's operator policies.
- `nanochat/gpt.py:108` produces Q/K/V together in the fused projection.
  RoPE at `:124` retains BF16 inputs with BF16 buffers. The Q/K RMSNorm calls
  at **`:125`** return **FP32 under CUDA autocast**, while V remains BF16.
  The installed PyTorch 2.13 header `torch/include/ATen/autocast_mode.h:876`
  lists `rms_norm` under its FP32 policy. Fake CUDA tensors verify the actual
  dispatch policy on this CPU-only machine for both BF16 and FP16.
- PyTorch's existing FlexAttention autocast shim casts all three inputs to
  the active autocast dtype (`torch/_higher_order_ops/flex_attention.py:307`).
  The native project adapter previously omitted that conversion at
  `steering/conditional/encoder_decoder/sm12_fa4.py:35` in the starting
  commit. CuTe checks low-precision V and matching Q/K/V dtypes in the
  installed `flash_attn/cute/interface.py:637,641`, matching the reported error.

This is an omitted native-adapter autocast conversion, not evidence of a
different eval-only autocast configuration. Training uses the same context;
uniformly BF16 production training inputs were not established here. The
operator policy and CPU reproduction are verified; the diagnosis of the
reported physical GPU run is inferred from that evidence and its traceback.

## Change and regression

`steering/conditional/encoder_decoder/sm12_fa4.py:35` now explains the SFT
caller and converts Q/K/V to the active autocast dtype before native CuTe.
Casting only K/V to `q.dtype` would make all inputs FP32, which CuTe rejects.
Without autocast the adapter preserves the caller's dtypes. No fallback was
added. `docs/reference/architecture.md:426` corrects the intermediate-dtype
contract.

`tests/steering/test_sm12_fa4_eval_dtype.py` adds:

- Actual CUDA autocast dispatch checks using storage-free fake CUDA tensors.
- Actual `evaluate_encdec_bpb` calls with real CPU decoder projections,
  GQA, RoPE, normalization, value embeddings, and LM loss. A test-only norm
  policy reproduces CUDA's FP32 promotion, and the installed full-prefix
  adapter reaches a fake native kernel through its CPU test seam. Both
  decoder layers must enter with FP32 Q/K and low-precision V and reach the
  kernel with uniformly BF16/FP16 inputs. Encoder output is supplied as a
  valid packed fixture and cross-attention is excluded to isolate the decoder.
- Exact cast-value, matching-input storage preservation, and nonzero-gradient
  checks with and without autocast.
- Real SM12 eager/compiled BPB validation cases, skipped without SM12 CUDA.

Before the fix, both CPU BPB cases and both mixed-dtype gradient cases failed
at the native boundary: **4 failed, 4 passed, 2 skipped**. Afterward all eight
CPU cases pass. These results do not count skipped GPU tests as validation.

## Commands and results

Python **3.13.2**, torch **2.13.0+cu130**, `torch.cuda.is_available() == False`.
The existing environment was reused without dependency changes:

```bash
export UV_PROJECT_ENVIRONMENT=/home/d0rb/Documents/Github/nano-gpt-pretrain-steer/.venv
export UV_NO_SYNC=1 UV_CACHE_DIR=/tmp/uv-cache-fa4-dtype
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
```

Completed distributed test runs additionally used `CUDA_VISIBLE_DEVICES=''`
and sandbox escalation to allow local Gloo/torchrun sockets.

| Exact command | Result |
| --- | --- |
| `uv run pytest tests/steering/test_sm12_fa4_eval_dtype.py -q` | Before: **4 failed, 4 passed, 2 skipped**. After: **8 passed, 2 skipped**. |
| `uv run pytest tests/steering/test_residual_feature_prediction_head.py tests/steering/selfcond tests/scripts/test_residual_head_production_recipe.py -q` | **523 passed, 1 failed, 4 skipped, 20 deselected**. Existing optimizer compilation-limit failure, reproduced below. |
| `uv run pytest tests/steering/test_residual_feature_prediction_head.py tests/steering/selfcond tests/scripts/test_residual_head_production_recipe.py -q --tb=short` | Untouched starting commit at `/tmp/sm12-fa4-base-5780fdf64`: identical **523 passed, 1 failed, 4 skipped, 20 deselected**. |
| `uv run pytest tests/steering/selfcond/test_training_integration.py::test_head_only_optimizer_group_step_and_state_roundtrip -q --tb=short` | **1 passed** on the fix branch in isolation. |
| `uv run pytest tests/steering/test_sm12_fa4.py tests/steering/test_sm12_fa4_eval_dtype.py tests/scripts/test_encdec_sft_training_loop.py -q` | **83 passed, 2 skipped, 36 deselected** with socket access. |
| `uv run ruff format steering/conditional/encoder_decoder/sm12_fa4.py tests/steering/test_sm12_fa4_eval_dtype.py` | **1 reformatted, 1 unchanged**. |
| `uv run ruff check steering/conditional/encoder_decoder/sm12_fa4.py tests/steering/test_sm12_fa4_eval_dtype.py` | Passed. |
| `uv run ruff format --check steering/conditional/encoder_decoder/sm12_fa4.py tests/steering/test_sm12_fa4_eval_dtype.py` | **2 files already formatted**. |
| `git diff --check` | Passed. |

The requested combined suite is **not fully green**: on both the fix and the
untouched starting commit, `test_head_only_optimizer_group_step_and_state_roundtrip`
hits `torch._dynamo.config.recompile_limit=8` in
`nanochat/optim.py:21::adamw_step_fused`. The last guard failure is a parameter
rank mismatch, expected 1 versus actual 2. Passing that test in isolation
supports an existing suite-order/cache interaction. This unrelated failure
was not altered or hidden by increasing the compiler limit.

The initial sandbox attention/SFT run had **80 passed, 3 failed, 2 skipped,
36 deselected**, with Gloo `Operation not permitted` errors. The same command
passed with socket access. The initial sandbox attempt of the requested suite
was interrupted and superseded by the completed run above. These overlapping
runs are not additive pass counts.

## Original CPU-only qualification limits

No physical CUDA kernel executed in that original investigation. The later
integrated H200 and SM120 qualification is recorded in
[the transformer-head experiment record](residual-feature-head.md#final-stack-merge-review). Native CuTe numerical behavior,
CUDA/Inductor compilation, native backward, the migrated step-250000 checkpoint,
the 10-layer production feature head, two-GPU SFT, memory use, and throughput
remain unverified. The added real-SM12 tests cover a tiny decoder BPB forward;
they do not constitute a production-checkpoint qualification.
