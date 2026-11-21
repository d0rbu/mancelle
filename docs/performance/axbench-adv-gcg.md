# AxBench-Adv GCG performance

This covers appended-token GCG in `axbench-adv`. Its Transformers baselines
are DiffMean, LinearProbe, LoRA, and PromptSteering. BPB-Adv has a separate
execution path and benchmark.

The [stack progress tables](axbench-adv-gcg-stack.md) break the recorded bounded
searches into preparation, filtering, gradients, scoring, generation, and driver
time, with a column for each PR. They retain missing measurements explicitly.

## Selected-factor integration after PR #1716

The performance stack now includes main `1d448b1abc`, which changes the
activation-L2 target to use the selected method factor. The optimized SiLU,
RMS and scoring-dispatch modules are byte-identical to their previously
qualified versions. The merge conflict was in the integration oracle:
its reference call now retains both the independent selected-factor loss
calculation and the native, unfused scoring path.

The combined RTX 4090 qualification passes 306 targeted CPU/functional tests
(76 GPU cases deselected) and 12 real-model integration cases. At the stack
tip, ten cases cover DiffMean and LinearProbe at factors 0, 0.6, 1 and 3, plus
LoRA and PromptSteering at factor one. Each checks both objectives, suffix
lengths 1/32, native score/gradient references, repeated gradients, bounded
searches and generation. The two preceding PR heads each also pass the
DiffMean factor-0.6 case. The upstream factor-zero cross-batch policy is
preserved; fixed-shape comparisons against native scoring remain exact.

A separate factor-one comparison against complete frozen PR #1714 modules
(`9754f3c1b`) passes all 56 measured gradient/score comparisons exactly.
The oracle fault check first completes an uncorrupted response-margin search,
then adds 0.125 only to fused RMS output during factor-0.6 L2 scoring; the
independent `checked_score` comparison rejects it. Ruff, formatting and the
scoped type check pass. The full `tests/` suite was not run.

These are compatibility checks, with no new performance claim. Historical
timings retain their original factor-one fixtures and receipts; the new L2
identity marker must not be removed to reuse old artifacts. Earlier remote
results and queued Hopper jobs use their retained source snapshots with
identical optimized kernels and dispatchers. Combined selected-factor runtime
verification described here is on the RTX 4090.

Sources, commands, logs, tensors, search receipts and resource audits are under
`out/axbench-gcg-performance/main1716-integration/`. Its collection contains
739 files and 11,442,970 bytes, with manifest SHA-256
`d04050d73b25daab354cd7cf02ffa93938a98cc9443de0a726638a2696f76207`.

## One backward per physical baseline batch

`GCGSession._side_batch` previously differentiated each scalar class-row loss
against the entire batch of input embeddings, retaining only that row's suffix
slice. With a physical batch of four, that traversed the same model graph four
times. The supported baseline forwards have independent rows. For row `i`,
`d(sum_j loss_j)/d(embeds_i) = d(loss_i)/d(embeds_i)`. Passing the scalar losses
as a tuple to `autograd.grad` therefore obtains the row gradients in one traversal.

The objectives remain separate until after extracting each row's suffix
gradient and converting it to CPU FP32. In particular, the implementation does
not average the class-row losses or move the response-margin alpha into the
backward graph. Forward padding, physical batches, member order, model state
guards, candidate scoring, token filtering, IPC, and the shared search loop keep
their existing contracts.

The project-model adapter keeps its rowwise implementation. An experimental
joint VJP on the real compiled project decoder did not pass the numerical gate:
activation-L2 gradient differences reached several percent. Identical compiler
buffer-donation settings did not eliminate the discrepancy. CPU reference
attention alone was insufficient to qualify that change.

## RTX 4090 evidence

The reference was `6b741afce7df3e89618a7e99b80fae3b9e1ed935`; its baseline
GCG worker is identical at the subsequent main revision `4fbd63966`. The
workload used SmolLM3-3B `it-soup-APO`, concept zero from the c50 bundle,
factor one, all 72 prepared training pairs, and four selected pairs per
gradient call. Physical batch size was four. Candidate scoring used the same
four pairs and two fixed suffixes. Token budgets were one and 32.

The environment was Python 3.12.11, PyTorch 2.5.1+cu124, Transformers 5.16.1,
PyVene 0.1.8, and PEFT 0.13.2. Timings are medians of five warm, synchronized,
unprofiled calls. Preparation and model loading are excluded. These are
**gradient-call speedups**, not full-search or campaign speedups.

| Method | Objective | Suffix tokens | Reference (s) | Joint backward (s) | Speedup |
|---|---|---:|---:|---:|---:|
| DiffMean | `response_margin` | 1 | 1.2498 | 0.5006 | 2.50× |
| DiffMean | `response_margin` | 32 | 1.3606 | 0.5471 | 2.49× |
| DiffMean | `activation_l2` | 1 | 1.0099 | 0.4209 | 2.40× |
| DiffMean | `activation_l2` | 32 | 1.1001 | 0.4614 | 2.38× |
| LinearProbe | `response_margin` | 1 | 1.2828 | 0.5204 | 2.46× |
| LinearProbe | `response_margin` | 32 | 1.3914 | 0.5651 | 2.46× |
| LinearProbe | `activation_l2` | 1 | 1.0316 | 0.4376 | 2.36× |
| LinearProbe | `activation_l2` | 32 | 1.1202 | 0.4770 | 2.35× |
| LoRA | `response_margin` | 1 | 1.2905 | 0.5245 | 2.46× |
| LoRA | `response_margin` | 32 | 1.3621 | 0.5704 | 2.39× |
| LoRA | `activation_l2` | 1 | 1.0127 | 0.4432 | 2.28× |
| LoRA | `activation_l2` | 32 | 1.1198 | 0.4794 | 2.34× |
| PromptSteering | `response_margin` | 1 | 1.6023 | 0.6563 | 2.44× |
| PromptSteering | `response_margin` | 32 | 1.6982 | 0.7058 | 2.41× |
| PromptSteering | `activation_l2` | 1 | 1.2573 | 0.5505 | 2.28× |
| PromptSteering | `activation_l2` | 32 | 1.3962 | 0.5932 | 2.35× |

All 16 gradient tensors and 16 candidate-score tensors were exactly equal to
the reference. The DiffMean response-margin profile at one suffix token
confirmed the mechanism: attention backward calls fell from 288 to 72, and
`aten::mm` calls fell from 2,530 to 1,012. Activation-L2 attention backward
calls fell from 232 to 58. Profiles explain the work reduction; the table uses
separate unprofiled timings.

The optimization is not a memory reduction. For the first DiffMean
response-margin cell, the observed combined gradient/score peak increased from
14.22 GB to 14.56 GB. The benchmark now runs one method per process so models
from earlier methods cannot contaminate peak-memory measurements.

### Bounded whole-search check

A separate DiffMean run used all 72 pairs, one suffix token, two search steps,
16 proposals per step, top-k 32, four sampled pairs, and physical batch size
two. Each arm ran one warmup and three timed searches, including preparation
and final 128-token sampled generation with a fake judge. All eight paired
search receipts and final generations matched exactly.

| Objective | Reference total (s) | Joint-backward total (s) | Reference gradient stage (s) | Joint gradient stage (s) |
|---|---:|---:|---:|---:|
| `response_margin` | 20.244 | 19.599 | 1.577 | 1.028 |
| `activation_l2` | 21.936 | 21.033 | 1.204 | 0.803 |

These medians show a 3.3–4.3% total speedup in the repeated run. An earlier
reference-first run showed approximately unchanged total time; preparation
varied enough to obscure the smaller gradient savings. The whole-search probes
shared the host with CPU pytest, although the GPU was dedicated to the probe.
They establish end-to-end parity and identify stages, not a stable campaign
throughput claim. Candidate scoring took roughly half to three-fifths of total
time, which limits the effect of a gradient-only optimization.

## Hopper qualification

Engaging job `22986682` completed on an NVIDIA H200 using `mit_normal_gpu`,
account/QoS `mit_amf_advanced_gpu`. It compared independent reference and
candidate processes using the checked-in benchmark, four pairs, physical batch
size eight, five warm repetitions, and both objectives at one/32 suffix tokens.
Python was 3.12.1; the PyTorch, CUDA, Transformers, PyVene, and PEFT versions
matched the 4090 experiment. The resolved model commit was
`cfb32d505f5025ec9be4e704f70cfbf5bdf8da94`.

DiffMean gradient speedups were 3.63–3.74× for response margin and 3.50–3.56×
for activation L2. LoRA speedups were 3.64–3.69× and 3.50–3.57× respectively.
All eight gradient tensors and eight score tensors matched exactly. These
larger-batch results are separate matched comparisons; they are not an H200
versus 4090 hardware-speed comparison.

## Blackwell qualification

Engaging job `22988796` completed on an NVIDIA RTX PRO 6000 Blackwell Server
Edition, driver 590.48.01, using `mit_preemptable` with account `mit_general`
and QoS `normal`. It repeated the Hopper workload with the same model commit
and non-PyTorch package versions. Both reference and candidate used Python
3.12.1 and PyTorch 2.7.1+cu128, CUDA 12.8. DiffMean gradient speedups were
3.74–3.88× for response margin and 3.68–3.78× for activation L2; LoRA speedups
were 3.77–3.91× and 3.65–3.82×. All eight gradient tensors and eight score
tensors matched exactly.

The benchmark used an isolated dependency overlay, adding
`--index https://download.pytorch.org/whl/cu128 --with torch==2.7.1+cu128`
to its `uv run --no-sync` invocation alongside Hydra and Loguru.
[PyTorch 2.7 introduced Blackwell support with CUDA 12.8 wheels](https://pytorch.org/blog/pytorch-2-7/).
The vendored lock now selects PyTorch 2.7.1+cu128 from an explicit CUDA 12.8
index on Linux and Windows, so normal `--frozen` workers use that runtime.
Other platforms retain PyPI. Transformers, PyVene, and PEFT versions are
unchanged; only PyTorch and its CUDA, Triton, and SymPy dependencies move.

Job `22994521` qualified the frozen environment on the same Blackwell GPU:
75 focused vendor tests and all four real baseline worker tests passed.
Those real tests compare every visited clean state, gradient, and candidate
score with the rowwise, full-logit reference and run bounded searches and
sampled generation with a fake judge. `uv pip check` passed, and the runtime
reported native `sm_120` support.

After rebasing onto the current host-rendering changes, job `22997700`
also passed all 90 vendor numerical and host tests, all four real baseline
worker tests, and the normal frozen preparation smoke test. The baseline
pipeline, worker-client, and GCG adapter subset passed 111 local tests;
its one environment-installing preparation test ran in that allocation.

H200 job `22994522` ran the frozen environment with DiffMean and LoRA, batch
size 8, four members, both objectives, and suffix lengths 1 and 32. All eight
gradient tensors and eight score tensors matched the earlier PyTorch
2.5.1/CUDA 12.4 Hopper results exactly. These checks qualify the environment;
they are not an additional performance-speedup claim.

## Activation-L2 vocabulary projection

Activation-L2 consumes a decoder-layer residual, so its full vocabulary-logit
output was unused. `GCGSession._capture_hidden` now requests
`logits_to_keep=1`. In the locked Transformers 5.16.1 implementations of Llama,
Qwen2, SmolLM3, and GPT-NeoX, this slices the LM-head input after all decoder
layers have run. All hidden positions and their derivatives are retained.
Zero would request every logit. PyVene receives the option inside its base-input
dictionary; PEFT and plain-model forwards receive the same model argument.
Response-margin forwards continue to compute all logits.

A fresh comparison against the preceding batched-backward implementation
(`05a316706`; its production code is unchanged by the rebase to `e27cc2905`)
used the same SmolLM3, source artifacts, RTX 4090, software versions, four-pair
minibatch, physical batch of four, and five warm repetitions described above.
Reference/candidate execution order alternated across the four methods.
All **16 gradient tensors and 16 candidate-score tensors matched exactly**,
including response-margin controls and suffix lengths 1 and 32.

| Method | Activation-L2 scoring, k=1 | Activation-L2 scoring, k=32 | Peak allocation saved, k=1 / k=32 |
|---|---:|---:|---:|
| DiffMean | 446.7 → 423.7 ms (1.054×) | 513.5 → 491.7 ms (1.044×) | 0.423 / 0.451 GB |
| LinearProbe | 430.4 → 417.3 ms (1.031×) | 466.9 → 450.3 ms (1.037×) | 0.423 / 0.451 GB |
| LoRA | 429.5 → 418.3 ms (1.027×) | 459.5 → 440.4 ms (1.043×) | 0.423 / 0.451 GB |
| PromptSteering | 525.6 → 505.0 ms (1.041×) | 560.4 → 541.9 ms (1.034×) | 0.511 / 0.539 GB |

Activation-L2 gradient calls improved 1.011–1.034×. Response-margin timing
ratios were 0.986–1.008×, with identical peak allocations. These are direct
worker-call measurements, not whole-search throughput. The saved allocation
is the warm gradient/scoring peak, measured after preparation and warmup.

The paired DiffMean CUDA traces confirm that the vocabulary projection inputs
changed from `[4,459,2048]` / `[4,443,2048]` to `[4,1,2048]`; the vocabulary
width remains 128,256. Reports, full tensors, and losslessly compressed traces
are under `out/axbench-gcg-performance/loop2/` in the measurement workspace.
The benchmark command below reproduces this comparison from separate checkouts.

The real-model CPU matrix checks exact clean hidden states, row losses,
gradients, and scores against a reference that restores the full LM-head
computation and original rowwise backward. It also checks the actual LM-head
input extent, including the unchanged response-margin path. The CUDA worker
oracle checks every prepared clean activation and every gradient/score consumed
by the bounded search against that reference.

## Reproduce a matched comparison

### Project activation-L2

The project adapter also requests `logits_to_keep=1` for activation capture.
`EncoderDecoderGPT.forward` applies that option immediately before its LM
head, after the complete decoder and final normalization. Its default zero
retains all logits, and positive values are rejected with training targets.
Feature logits and KV-cache contents retain every input position. The shared
SelfConditioning, SelfConditioningMasked, and NLSteering GCG adapter uses the
option; the rowwise project backward remains intact.

The real-checkpoint comparison on the RTX 4090 covered SelfConditioning and
SelfConditioningMasked, both objectives, suffix lengths one/32, and physical
batches one/two. All 16 gradient tensors, 16 score tensors, and six generation
records matched the preceding implementation exactly. The CUDA test also
compares every individual clean activation, row loss, and suffix gradient
against a full-LM-head forward. NLSteering shares the changed call site but
has not received a separate real-artifact run in this measurement.

The timing workload used the normally loaded 12-layer, width-768 project
checkpoint at step 54,600, all 72 prepared pairs, four selected pairs, physical
batch four, two candidates, and factor one. Software was Python 3.13.2 and
PyTorch 2.13.0+cu130 on the RTX 4090, driver 580.82.09. After warming both
projections, each cell used 21 alternating-order reference/candidate pairs:

| Method | Suffix tokens | Reference score median (ms) | Reduced score median (ms) | Median paired speedup |
|---|---:|---:|---:|---:|
| SelfConditioning | 1 | 119.741 | 114.641 | 1.050× |
| SelfConditioning | 16 | 119.290 | 113.642 | 1.051× |
| SelfConditioningMasked | 1 | 126.622 | 121.590 | 1.041× |
| SelfConditioningMasked | 16 | 123.612 | 119.113 | 1.042× |

These activation-L2 scoring gains were positive in 18–20 of the 21 pairs per
cell. Gradient ratios were only 1.006–1.013×, with mixed pairwise results;
they do not establish a reliable gradient-latency improvement. Unchanged
response-margin controls ranged from 0.991–1.015×. At one/16 suffix tokens,
scoring peak allocation fell by 0.321/0.341 GB and gradient peak by
0.358/0.373 GB for both methods. Every timed gradient/score pair matched
exactly. Separate-process comparisons also produced byte-identical complete
tensor files, but their response-margin controls drifted more, so the timing
table uses paired measurements. These are warm runtime-call measurements,
not full-search or campaign throughput.

Reports, full tensors, raw paired timings, and compressed profiles are under
`out/axbench-gcg-performance/loop3/` in the measurement workspace. Focused
tests cover default/training behavior, invalid counts, oversized counts,
input gradients, compact/uncompacted feature heads, and complete KV-cache
prefill/decode contents. The real project test also passed at its original
physical gradient batch size one with the default compiler settings.

Use the root environment for the project benchmark. Its input has the same
`source`, `concept`, and ordered `rows` structure as the baseline benchmark
below. The source additionally needs `target_meta_path`,
`feature_statistics_path`, `project_direction_root`, and `project_layer`, as
for the real project GCG test. The checkpoint and direction-fit identities
are validated before measurement. Concept zero is used, and all supplied
training rows are prepared; `members` selects the timed gradient/scoring
minibatch.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run \
  python -m scripts.benchmark.benchmark_axbench_adv_gcg_project \
  input=/absolute/path/gcg-benchmark-input.json \
  output=/absolute/path/project-reference full_logits=true profile=true

CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run \
  python -m scripts.benchmark.benchmark_axbench_adv_gcg_project \
  input=/absolute/path/gcg-benchmark-input.json \
  output=/absolute/path/project-candidate full_logits=false profile=true
```

Repeat with `method=SelfConditioningMasked`. Defaults live in
`configs/eval/benchmark_axbench_adv_gcg_project.yaml`. Each arm runs in a fresh
process, warms each case, and stores complete tensors, per-call timings,
separate gradient/scoring peak allocations, source/checkpoint digests, and
optional losslessly compressed traces. Loading, feature preparation, and
compilation are excluded from the warm-call timing. A benchmark-only judge
rejects any accidental evaluation call.

For smaller timing differences, use one prepared cell and alternate reference
and candidate order on every pair:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run \
  python -m scripts.benchmark.benchmark_axbench_adv_gcg_project \
  input=/absolute/path/gcg-benchmark-input.json \
  output=/absolute/path/project-paired paired=true repeats=21
```

This warms both projections, checks every pair with zero tolerance, and records
both sets of raw timings and peaks. `median_paired_speedup` is the median of
the reference/candidate ratios within pairs, rather than the ratio of the two
unpaired medians. Reference patch setup and numerical assertions are outside
the measured interval. `paired=true` requires `full_logits=false`.
The default `paired_reference=logits` selects the full LM-head reference.
Use `paired_reference=state` to compare the current state guard with the
frozen implementation from commit `85d44829f9ba4201aff9680f24ee5d6089430c09`.
That reference lives in `tests/axbench_adv/gcg_state_reference.py`; it retains
the original guard's complete implementation, independently of production.

Both arms explicitly set `torch._functorch.config.donated_buffer=False`
before loading the model. On the measured PyTorch 2.13 environment, the
unchanged reference with physical batch four otherwise raises because its
retained backward graph uses donated compiled buffers. This is an existing
project-runtime compatibility limit, not a fix supplied by LM-head slicing.
The benchmark records this setting; these results do not qualify the default
compiler setting for larger project gradient batches.

A separate RTX 4090 probe on the same Torch 2.13 installation warmed a
compiled backward with donation enabled, then changed the flag to false.
The first retained row gradient matched its fresh-forward reference exactly;
the remaining three had relative L2 errors of 1.0004–1.0019. Changing the
flag after compilation is therefore not a safe workaround for that cached
backward. The probe and results are retained as
`probe_donated_warm_cache.py` in the measurement scratch directory and
`out/axbench-gcg-performance/loop6/donated-warm-cache.json`. No runtime flag
change is included in these performance patches.

### Transformers baselines

Create a source JSON containing the `prepared_dir`, `main_dump_dir`, and
`aux_dump_dir` of the audited baseline bundle, as for the real-GPU tests in
[AxBench-Adv evaluation](../evaluation/axbench-adv.md). In the root environment,
materialize the benchmark's exact ordered training population:

```python
import json
from pathlib import Path

from steering.eval.axbench_adv.gcg_driver import _training_rows
from steering.eval.axbench_upstream.data import select_training_pairs

source = json.loads(Path("gcg-source.json").read_text())
training = select_training_pairs(Path(source["prepared_dir"]), max_concepts=1)[0]
rows, _ = _training_rows(training, master_seed=42, evaluation_prompts=())
Path("gcg-benchmark-input.json").write_text(json.dumps({
    "source": source, "concept": training.concept, "rows": rows,
}, indent=2))
```

Run the benchmark in the **vendored Python 3.12 environment**. Hydra and Loguru
are benchmark-only dependencies supplied with `--with`:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTHONPATH=vendor/axbench:. \
uv run --project vendor/axbench --frozen \
  --with hydra-core==1.3.2 --with loguru==0.7.3 \
  python -m scripts.benchmark.benchmark_axbench_adv_gcg_hf \
  input=/absolute/path/gcg-benchmark-input.json \
  output=/absolute/path/new-DiffMean method=DiffMean
```

Repeat for LinearProbe, LoRA, and PromptSteering. Each output directory must
be new. Defaults are in `configs/eval/benchmark_axbench_adv_gcg_hf.yaml`.
`profile=true` records CPU/CUDA Chrome traces and operator tables for the first
configured token count of each objective, after the unprofiled timings.
`report.json` records the input digest, production-worker digest, source commit
when available, model commit, package versions, physical batch sizes, attention
backend, raw timings, and peak allocations. `numerics.safetensors` preserves
the complete gradients and score matrices. No generation or judge calls occur.

For an older checkout, copy the benchmark script, its YAML,
`scripts/benchmark/gcg_paired.py`, and
`tests/axbench_adv/gcg_state_reference.py` into the reference worktree; keep
its production worker untouched. Run the same input
and overrides on the same GPU and environment in a separate process. Compare
matching names, shapes, dtypes, and values before using a timing delta:

```python
from pathlib import Path

import torch as t
from safetensors.torch import load_file

reference = load_file(Path("old-DiffMean/numerics.safetensors"))
candidate = load_file(Path("new-DiffMean/numerics.safetensors"))
assert reference.keys() == candidate.keys()
for name in reference:
    t.testing.assert_close(candidate[name], reference[name], rtol=0, atol=0)
```

Do not use cross-device comparisons, different physical batch partitions, or
profiler wall time as a numerical/performance reference. Keep compilation,
attention selection, deterministic settings, model artifacts, and tokenizer
revision fixed. Deterministic CUDA settings belong to qualification; production
backend defaults are unchanged.

For a paired baseline state-guard comparison, add `paired_state=true repeats=21`
to the baseline command above. Both baseline and project paired runs use
`scripts/benchmark/gcg_paired.py`, alternate reference/candidate order, and
assert exact equality for every timed pair and warmup. Each result is cloned
before the next operation, including tensors read from overwritten worker
sidecars. Scoring comparisons retain Python-float precision through FP64
snapshots. Reference context setup, snapshot copying, and numerical checks
are outside the measured interval. Reports include hashes of the benchmark,
guard, and frozen reference sources.

The baseline benchmark writes a full FP32 input-embedding sidecar during
preparation. Its output directory needs sufficient temporary space. A
RAM-backed output directory can be used for both arms when measuring runtime
work with sidecar I/O kept in RAM; record that choice alongside the results.

## State-guard overhead

The shared GCG guard snapshots model modes, parameter gradients, hook
registries, and RNG state around runtime calls. The guard now copies only
nonempty hook registries and restores mode/flag attributes only when their
identities changed. It still retains every original registry object, clears
newly added hooks, restores removed hooks in order, and restores state on
exceptions. Module traversal, calls to `eval()`, parameter freezing, gradient
snapshots, and RNG handling follow the original implementation.

The project CPU profile identified repeated dictionary copies and attribute
writes in this guard. On the normally loaded 253-module, 164-parameter
project model, 101 alternating isolated guard measurements reduced the
median from 2.677 ms to 1.458 ms. Every pair favored the candidate. Separate
snapshot and flag-restoration ablations both reduced runtime; the isolated
measurement is not an end-to-end speedup.

The first real-runtime experiments used the same RTX 4090, checkpoint,
software, and workload described above: 72 prepared pairs, four selected
pairs, physical batch four, two candidates, five warmups per arm, and 21
alternating timed pairs. All gradients and scores matched exactly, and peak
GPU allocations were identical. Candidate-scoring results were:

| Method | Objective | Suffix tokens | Original median (ms) | Candidate median (ms) | Median paired speedup |
|---|---|---:|---:|---:|---:|
| SelfConditioning | response_margin | 1 | 98.920 | 95.105 | 1.034× |
| SelfConditioning | response_margin | 16 | 134.082 | 130.334 | 1.043× |
| SelfConditioning | activation_l2 | 1 | 125.881 | 120.133 | 1.048× |
| SelfConditioning | activation_l2 | 16 | 112.727 | 107.779 | 1.042× |
| SelfConditioningMasked | response_margin | 1 | 93.924 | 89.573 | 1.050× |
| SelfConditioningMasked | response_margin | 16 | 121.900 | 115.512 | 1.055× |
| SelfConditioningMasked | activation_l2 | 1 | 111.967 | 107.379 | 1.042× |
| SelfConditioningMasked | activation_l2 | 16 | 109.812 | 106.075 | 1.037× |

Scoring was faster in 16–21 of the 21 pairs per case. Response-margin
gradient paired ratios were 1.013–1.041×; activation-L2 gradient results were
mixed and establish no reliable improvement. These are incremental warm
runtime-call measurements relative to the preceding LM-head optimization.
They do not measure full-search throughput. The original experiments and
isolated ablations are retained under
`out/axbench-gcg-performance/loop5/paired-guard-{selfconditioning,masked}/`.
The checked-in `paired_reference=state` mode reproduces this comparison
without replacing the benchmark harness.
An independent SelfConditioning replay through that command matched all
168 timed pairs exactly and measured scoring ratios of 1.037–1.048× across
the four cases. Peak allocations differed by at most 4 KiB. Its report and
full tensors are under `loop5/paired-state-cli-selfconditioning/`.

The corresponding baseline runs used the RTX 4090 and the earlier locked
PyTorch 2.5.1+cu124 environment, with four selected pairs, physical batch
four, two candidates, both objectives, and suffix lengths one/32. The
sidecars were on a RAM-backed filesystem in both arms. Each method matched
exactly in all 168 timed gradient/score pairs, and peak allocations were
identical. The eight per-case median paired ratios for each method were:

| Method | Minimum speedup | Maximum speedup | Faster pairs per 21-pair case |
|---|---:|---:|---:|
| DiffMean | 1.0019× | 1.0039× | 15–21 |
| LinearProbe | 1.0012× | 1.0038× | 14–20 |
| LoRA | 1.0021× | 1.0039× | 18–20 |
| PromptSteering | 1.0018× | 1.0030× | 14–20 |

These baseline gains are small; the stronger performance result is in the
project scoring table. Baseline reports and complete tensors are retained
under `out/axbench-gcg-performance/loop5/baseline-*-ram/`. The frozen
PyTorch 2.7.1+cu128 environment also passed the 90-test model matrix and the
eight CPU state/paired-timer tests in Slurm job `23001816`; that run checks
compatibility rather than GPU performance.

## Prepared project response weights

Each project response is fixed for a prepared GCG cell. Its response-margin
byte weights now come from the target tokenizer once during preparation and
remain with that row as a detached FP32 tensor on the model device. The loss
reduction, per-row backward, member order, and suffix routing stay unchanged.
Activation-L2 rows keep no byte weights. Repreparing a cell constructs new rows
and weights; there is no global cache or new configuration option.

Preparation explicitly disables inference mode while creating and transferring
these constants. Without that protection, preparing under
`torch.inference_mode()` retained inference tensors that a later normal
backward could not save. The frozen pre-cache loss implementation succeeded
on the same prepared rows, establishing a regression introduced by weight
reuse. Construction and the device transfer both belong inside the guard:
moving an ordinary CPU tensor to CUDA outside it creates an inference tensor
again. The caller's inference and gradient modes are restored, including when
zero-byte validation raises. This correction changes preparation only and
has no separate speedup claim.

The correction passed 20 focused CPU tests, including score/gradient comparisons
against independently constructed multibyte weights. It also passed a CUDA
transfer/backward test and the real project checkpoint test for SelfConditioning and
SelfConditioningMasked. The latter prepares the k=32 response-margin cells
under inference mode before checking normal gradients against independently
decoded weights, full logits and the original state guard. Normal k=1
preparation, both L2 lengths and generation remain covered. Removing the
guard fails the CPU regression; moving only the CUDA transfer outside it
fails the focused GPU test. Coverage reaches all four added executable lines
and both preparation branches. The real checkpoint test passed in 80.35 s with
the compiler defaults unchanged. Evidence is retained under
`out/axbench-gcg-performance/loop13/`.

The reference for this measurement is commit
`462077c7a4e4de62c50e7f1b0aa4ac8a390ac7b2`. Its complete original
`_project_batch_values` method was loaded from frozen source, whose AST was
checked against that commit. The paired harness switches only that method,
uses the shared independent numerical snapshots, and alternates order for
21 measured pairs after five warmups. Both arms share the same prepared cell
and set buffer donation false before loading the model, as described above.
The workload and RTX 4090 environment match the project benchmark: 72 prepared
pairs, four selected members, physical batch four, and two candidates.

| Method | Suffix tokens | Reference scoring (ms) | Prepared weights (ms) | Median paired speedup | Faster pairs |
|---|---:|---:|---:|---:|---:|
| SelfConditioning | 1 | 85.9743 | 84.6543 | 1.01372× | 17/21 |
| SelfConditioning | 16 | 109.1105 | 108.6051 | 1.00465× | 13/21 |
| SelfConditioningMasked | 1 | 85.8647 | 84.6285 | 1.00981× | 14/21 |
| SelfConditioningMasked | 16 | 109.8713 | 108.7463 | 1.01193× | 19/21 |

These are small improvements in warmed response-margin scoring calls.
Gradient timings were mixed, and activation-L2 controls were approximately
unchanged. Every one of the 252 measured gradient/scoring pairs was exact.
An earlier CPU-weight-only variant also produced 168 exact pairs and
response-margin scoring ratios of 1.00720× and 1.00781×.

For this population, the 144 prepared CUDA weight tensors contain 75,632
payload bytes. Cloning all of them increased live allocated memory by 106,496
bytes (104 KiB). Both paired arms already hold the prepared weights, so their
peak difference does not measure this additional resident cost. The cache
size scales with the total number of prepared response tokens.

A follow-up collected Python garbage before each arm, outside the timed
interval, and retained every allocation-peak sample. Scoring ratios were
1.01386× (20/21 faster pairs) and 1.00776× (18/21), with another 84 exact
gradient/scoring pairs. An approximately 80 MB long-gradient peak discrepancy
in the original run disappeared in this controlled run: every gradient pair
differed by only 3 KiB, and scoring peaks were identical. The earlier transient
peak difference is excluded from memory-saving claims. These results are in
`loop7/paired-device-weights-explicit-gc/`.

Reports, raw timings, and full numerical tensors are retained under
`out/axbench-gcg-performance/loop7/paired-device-weights-{selfconditioning,masked}/`.
The frozen reference and paired harnesses are in the measurement scratch
directory as `reference-project-inputs.py`, `paired_project_response_weights.py`,
and `paired_project_device_weights.py`; each report records their source
digests. The existing native project benchmark can also be run in separate
reference/current checkouts with `full_logits=false`.

Focused tests exercise multibyte target-tokenizer output, unequal and reordered
responses, stale state from a previous objective, repeated scoring/backward,
unchanged cached tensor identity, preparation RNG restoration, and zero-byte
validation restricted to response-margin. The real-model numerical oracle
independently decodes each response again for its reference forward, alongside
the full LM-head and original state-guard references.
The targeted CPU unit/functional subset passed 40 tests, and the real project
GPU test passed in 85.49 s with the extended oracle and generation regression.
Five deliberately introduced faults failed the intended assertions: wrong
tokenizer, reversed weights, stale objective state, decoding activation-L2
responses, and decoding again during search. Coverage reached every added
executable statement reported by coverage.py; both preparation choices are
checked explicitly. These are focused checks, rather than full-module coverage.

## Batched candidate prompt validation

The baseline CPU trace identified repeated scalar prompt tokenization inside
`GCGSession._candidate`. Each candidate visits every prepared positive and
negative prompt, including when scoring and gradients revalidate a previously
filtered suffix. With 72 training pairs, that is 144 complete prompts per
candidate. The scalar tokenizer-only profile spent 29 ms in 144 Rust
`encode_batch` calls and another 4 ms validating the returned token IDs.

Fast tokenizers now receive those complete strings in one batch. The suffix's
standalone decode/re-encode check, special-token settings, row order, fixed
prefix/tail comparison, and error strings retain their original contracts.
The implementation keeps no cross-call cache. It validates the entire batch's
shape and ID types before consuming any row. A tokenizer exception or malformed
batch retries scalar encoding lazily in the original order; thus an earlier
framing mismatch still wins over an exception from a later row. Slow tokenizers
retain scalar encoding. Temporary CPU storage scales with the prepared prompt
population; no GPU tensors or model forwards change.

The reference is the frozen scalar candidate validator from
`4e3fb993e50d5f0685fb4380aa619c2e723a503c`. A CPU filter-only comparison used the
cached SmolLM3 tokenizer, all 72 real training pairs, source-selected wrappers,
and actual PromptSteering prefix recovery. It constructed production completion
layouts without loading a model and restricted the allowed pool to the two
canonical candidates being measured. Each cell alternated reference/candidate
order for 21 timed pairs after five warmups, comparing the complete result.
The host was an AMD Ryzen 9 9950X with 32 available logical CPUs; tokenizer
parallelism and Rayon thread counts used their library defaults. The Python
3.12.11 / Torch 2.5.1 / Transformers 5.16.1 environment matches the earlier
local baseline measurements.

| Method | Suffix tokens | Scalar filtering (ms) | Batched filtering (ms) | Median paired speedup |
|---|---:|---:|---:|---:|
| DiffMean | 1 | 59.9594 | 9.1304 | 6.3245× |
| DiffMean | 32 | 64.2319 | 10.0166 | 6.2198× |
| LinearProbe | 1 | 60.0788 | 9.4165 | 6.2509× |
| LinearProbe | 32 | 63.9380 | 10.3728 | 6.0866× |
| LoRA | 1 | 60.5492 | 9.2792 | 6.2072× |
| LoRA | 32 | 63.9004 | 10.0635 | 6.1632× |
| PromptSteering | 1 | 86.3854 | 12.7327 | 6.8608× |
| PromptSteering | 32 | 89.5594 | 13.7291 | 6.3054× |

Each call validates two candidates. All 168 timed pairs matched exactly,
and the batched arm was faster in every pair. These filter-only gains include
token-ID validation and canonicalization; the larger tokenizer-only ratios
exclude part of that work and are not used as the performance claim. Raw
results and the scalar profile are under
`out/axbench-gcg-performance/loop8/{paired-filter,tokenizer-only}/`.

A paired whole-search check used DiffMean on the RTX 4090, the same 72-pair
population, four selected pairs, physical batch two, one suffix token, two
optimizer steps, and 16 proposals per step. Both arms used one persistent
worker. Each objective ran one warm pair followed by three measured pairs
with alternating order. Timing includes cell preparation and normal 128-token
sampled generation, with a fake judge; model loading is outside the interval.

| Objective | Scalar whole search (s) | Batched whole search (s) | Median paired speedup |
|---|---:|---:|---:|
| `response_margin` | 18.8080 | 17.6173 | 1.06759× |
| `activation_l2` | 19.9068 | 18.9459 | 1.04890× |

All three measured pairs improved for each objective. All eight pairs,
including warmups, produced byte-identical search receipts and identical
generation results. The ratios are medians of within-pair ratios, which need
not equal the ratio of the separately reported medians. These are bounded
search results from one concept, rather than full-campaign throughput or
judged attack-quality results. Timings, stage breakdowns, receipts, and worker
arm records are in `loop8/whole-search-diffmean-v2/` under the evidence root.
The first harness invocation selected the base Python interpreter instead of
the vendor environment and failed before model loading; it is excluded.

The targeted CPU checks passed 111 tests; one subsequently added standalone
round-trip rejection check also passed. They cover malformed candidate
payloads, batch exceptions, missing/short/extra/invalid batch output, first
framing-error precedence, successful-batch rejection, and real fast-tokenizer
UTF-8/BOS/EOS behavior without downloading a model. Coverage reaches all
34 statements and 20 branches across `_encode_many` and `_candidate`, not the
entire worker module. Eight deliberately introduced faults failed their
intended assertions: omitted batching, truncation, omitted special tokens,
unchecked batch length, unchecked row IDs, eager scalar fallback, reversed
rows, and omitted fixed tails. The CUDA worker oracle additionally compares
every candidate it visits to the frozen scalar validator, alongside its
existing numerical references. All four real SmolLM3 baseline GPU tests passed
in 178.62 s, covering both objectives, suffix lengths one/32, candidate
validation, exact clean activations/gradients/scores, IPC sidecars, bounded
optimizer searches, and normal sampled generation.

A frozen-environment Blackwell recheck of this candidate-validator change
completed successfully in Slurm job `23009653`. Its worker source matches
`e4c3507ce9ace1f6d1298332f65dc1c2534065ba`. On an RTX PRO 6000 Blackwell Server
Edition (compute capability 12.0, driver 590.48.01), Python 3.12.1 / Torch
2.7.1+cu128 passed 134 targeted CPU/model tests and all four real baseline GPU
cases (414.88 s). Package compatibility checks passed. Logs and hardware,
package and test records are in `loop8/blackwell-23009653/` under the evidence
root. This result covers the batched candidate validator at that source revision.

## Allowed-token pool construction

The SmolLM3 baseline rebuilds its canonical allowed-token pool on each
preparation. The scalar path took a median 1.2165 s across seven warmed calls;
its CPU profile visited 128,000 non-excluded vocabulary entries and made
124,259 scalar encoding calls. Those calls dominated the profile.

Fast tokenizers now check round trips in batches of at most 512 eligible
strings. Decode validation and exclusions are unchanged. A failed batch falls
back to per-token checks, preserving the rule that an expected encoding
failure skips only its own ID. Unexpected scalar errors still propagate.
Slow tokenizers keep interleaved scalar decoding and encoding. There is no
persistent cache or new configuration option.

Against the frozen scalar implementation from
`e4c3507ce9ace1f6d1298332f65dc1c2534065ba`, the actual implementation reduced
median pool-construction time from 1.2077 s to 0.6978 s. The median paired
speedup was 1.72794×, with all 21 measured pairs faster after three warm pairs.
Every pair produced exactly the same ordered pool of 122,908 IDs from the
128,256-entry vocabulary. An earlier unbounded-batch prototype yielded only
1.04097× with much wider timing variation. These are preparation component
measurements, rather than whole-search gains.

Whole DiffMean searches then alternated the frozen and current pool builders
inside one persistent 4090 worker. Each objective used one warm pair and three
timed pairs, 72 prepared training pairs, four selected members, physical batch
size two, k=1, two search steps, 16 proposals, top-k 32, and normal 128-token
final generation with a fake judge. Timing includes preparation and generation
but excludes model loading. Response-margin median elapsed time fell from
17.6397 s to 17.1620 s, with a median paired speedup of 1.02797×. Activation-L2
fell from 18.9650 s to 18.4788 s, with a median paired speedup of 1.03051×.
All six measured pairs improved. Search receipts were byte-identical and
generated text matched exactly in all eight pairs, including warmups.

An independent allocation probe measured a 14,070,066-byte Python-tracked
peak for bounded batching versus 13,945,400 bytes for the scalar reference.
The rejected unbounded prototype peaked at 129,049,632 bytes. These are
incremental `tracemalloc` allocations, including the returned pool; they
exclude native tokenizer allocations and are not total process RSS or timing
measurements.

The final targeted check passed 133 CPU tests; the 66 worker-specific tests
also passed in the installed Python 3.12 vendor environment. Coverage reached
all 35 statements and 18 branches in the two pool-construction functions.
Eleven deliberate faults failed their intended checks, including relaxed
exclusions, skipped round trips, reversed IDs, lost final chunks, aborted
fallback, empty-pool acceptance, and unbounded batches. All four real baseline
GPU tests passed (195.21 s), with every preparation additionally checking its
complete ordered pool against the frozen scalar reference. New test/reference
files pass type checking; checking the production worker still reports the
same pre-existing optional-`logits` diagnostic reproduced on its parent.

The real-tokenizer measurements use the same local Python 3.12.11 / Torch
2.5.1 / Transformers 5.16.1 environment and Ryzen 9950X host as the preceding
baseline measurements. Profiles, complete ordered pools, raw timings, and
harness source digests are under `out/axbench-gcg-performance/loop10/`.

The project RustBPE tokenizer was separately examined through real checkpoint
preparation. Its complete two-candidate filter over 72 pairs took 5.9659 ms
at k=1 and 7.5966 ms at k=16. Its existing threaded batch encoding API was
slower than scalar encoding in all 42 paired tests: 1.9213 ms versus 0.4455 ms
at k=1, and 2.0294 ms versus 0.8380 ms at k=16 for 144 prompt strings.
Project scoring and gradients already avoid repeating candidate validation.
Batching was therefore not introduced into that adapter. This separate
rejected experiment is recorded in `loop9/project-filter-decision.json`.

## Rejected response-only vocabulary projection

A response-margin trace attributed 23.036 ms of 360.033 ms of GPU kernel time
to four vocabulary-projection kernels. Restricting that projection to response
positions looked attractive, but three probes did not meet the acceptance gate.

Cropping the projection in both scoring and gradients preserved the selected
forward logits and losses exactly, but changed the suffix gradient: the
DiffMean probe had relative L2 error 0.02122 and maximum absolute error
0.0005484. Repeating the original gradient was exact. Isolating the LM-head
backward reproduced differences before the rest of the decoder backward;
the selected hidden states and forward logits still matched exactly.

Cropping only scoring with an integer `logits_to_keep` retained exact scores
and gradients, but was slightly slower (paired ratios 0.99395× and 0.99770×
at k=1/32). The trace showed that the noncontiguous tail changed the four head
matrix multiplications to batched matrix multiplications; their measured
kernel time increased to 24.728 ms. Lower allocation peaks alone did not
justify this variant.

Using indexed response positions instead produced contiguous head inputs and
about 1.033–1.035× faster DiffMean/LoRA scoring in small paired probes. It also
introduced score differences. A fixed diagnostic budget of `rtol=1e-4`,
`atol=1e-6` was set before expanding the workload. The complete bounded
DiffMean searches then failed that budget: four of 244 raw score comparisons
per response-margin pair exceeded it, repeatedly across four pairs. The worst
was 2.7229447365 versus 2.7216668129 (absolute difference 0.0012779236).
Optimizer requests, gradients, selected suffixes and final generated text
still matched, but those matches did not override the raw-score failure.
Activation-L2 remained exact.

No projection change was retained, and the tolerance was not widened. The
failed whole-search run exited with status 1; its timings are not accepted
performance evidence. The profiles, isolated backward checks, per-score
comparisons and rejected patch are retained under
`out/axbench-gcg-performance/loop11/`, with the decision recorded in
`response-projection-decision.json`.

## Prepared baseline response tensors

A warmed two-candidate, four-member response-margin scoring call constructed
48 tensors from Python sequences. Of these, 32 recreated the same response
targets and byte weights; the remaining 16 built model inputs. The CPU profile
attributes substantial waiting time to those constructors because they also
synchronize with earlier GPU work. That attributed time is not a prediction
of removable computation. Separate paired timings establish the actual gain.

The baseline session now retains only the fixed target IDs and FP32 byte
weights, keyed by the prepared sample ID and actual logits device. Entries
are created lazily, reused across candidates and gradient calls, and cleared
whenever the existing cell reset runs. Activation-L2 creates no entries.
The original model forwards, cross entropy, byte-weighted reduction, pair
arithmetic and backward computation remain unchanged.

Construction explicitly disables inference mode for these constant tensors.
This permits scoring inside `torch.inference_mode()` followed by a normal
gradient call: autograd must be able to save the targets and weights. The
caller retains its original mode. A regression test reproduces the failure
without this protection and compares both operations to the frozen parent;
the sequence also passed exactly on the real 4090 runtime.

The reference is the complete `_side_batch` body from
`5b29e4b0460dc4a339465d6ade23cb5a28efe435`. Its test copy was checked by AST
comparison, allowing only the function name/signature and four explicit
module-helper bindings. The paired harness runs the actual implementation
against this reference, alternating order for 21 measured pairs after three
warm pairs. It uses the same SmolLM3 checkpoint and local Python 3.12.11 /
Torch 2.5.1 / Transformers 5.16.1 environment as the preceding baseline
measurements: 72 prepared pairs, four selected members, physical batch four,
two scoring candidates, and k=1/32.

Scoring calls:

| Method | Suffix tokens | Reference (ms) | Reused tensors (ms) | Median paired speedup | Faster pairs |
|---|---:|---:|---:|---:|---:|
| DiffMean | 1 | 374.4975 | 372.9665 | 1.00458× | 19/21 |
| DiffMean | 32 | 400.5979 | 399.1056 | 1.00306× | 20/21 |
| LinearProbe | 1 | 374.5227 | 373.4789 | 1.00296× | 20/21 |
| LinearProbe | 32 | 400.6296 | 399.2596 | 1.00364× | 19/21 |
| LoRA | 1 | 376.0217 | 374.6398 | 1.00320× | 19/21 |
| LoRA | 32 | 401.8963 | 400.3613 | 1.00409× | 19/21 |
| PromptSteering | 1 | 456.0947 | 454.3341 | 1.00328× | 20/21 |
| PromptSteering | 32 | 484.9774 | 483.4313 | 1.00333× | 21/21 |

Gradient calls:

| Method | Suffix tokens | Reference (ms) | Reused tensors (ms) | Median paired speedup | Faster pairs |
|---|---:|---:|---:|---:|---:|
| DiffMean | 1 | 471.7003 | 470.1603 | 1.00285× | 21/21 |
| DiffMean | 32 | 511.8961 | 508.7939 | 1.00452× | 21/21 |
| LinearProbe | 1 | 473.1894 | 471.3307 | 1.00371× | 21/21 |
| LinearProbe | 32 | 511.3237 | 509.5257 | 1.00331× | 19/21 |
| LoRA | 1 | 473.5895 | 472.0258 | 1.00354× | 20/21 |
| LoRA | 32 | 512.4256 | 510.8148 | 1.00283× | 19/21 |
| PromptSteering | 1 | 581.0037 | 579.1902 | 1.00303× | 21/21 |
| PromptSteering | 32 | 624.8489 | 623.0340 | 1.00302× | 19/21 |

All 336 measured numerical comparisons were exact; 318 paired calls were
faster with reuse. These are modest improvements in warmed component calls.
Ratios are medians of individual paired ratios, so they need not equal the
ratio of the separately reported latency medians.

The same source also passed on an H200 (Slurm `23020001`, `mit_normal_gpu`,
account/QoS `mit_amf_advanced_gpu`). That run used eight members and physical
batch eight, Python 3.12.1, Torch 2.5.1/CUDA 12.4, Transformers 5.16.1 and
driver 590.48.01. DiffMean scoring speedups were 1.00823–1.00912× and gradients
1.00848–1.00851×; LoRA scoring was 1.00692–1.00895× and gradients
1.00704–1.00793×. All 168 measured comparisons were exact and all 168 calls
were faster. The eight cache regression tests also passed. These are warmed
component measurements, not whole-search measurements. The worker SHA-256
matches the local final source; reports and full tensors are preserved under
`loop12/hopper-23020001/`.

A full DiffMean replay used the same bounded search as the preceding checks:
72 prepared pairs, four selected members, physical batch two, k=1, two steps,
16 proposals, top-k 32 and normal 128-token sampled generation with a fake
judge. Preparation and generation are inside the interval; model loading is
outside. Each objective ran one warm pair and three timed pairs.

| Objective | Reference search (s) | Reused tensors (s) | Median paired speedup | Faster pairs |
|---|---:|---:|---:|---:|
| `response_margin` | 17.1832 | 17.1220 | 1.00235× | 3/3 |
| `activation_l2` control | 18.4800 | 18.5067 | 0.99825× | 0/3 |

The response-margin whole-search improvement is small; the unchanged L2
control illustrates variation of a similar order. No general campaign gain
is inferred from these three timed pairs. Across all eight pairs, every
optimizer request and gradient hash matched exactly, as did all 1,872 raw
score comparisons, byte-for-byte search receipts and generated outputs.
The final comparison checks the complete optimizer trace rather than relying
only on the selected suffix.

For all 72 pairs, the 288 retained tensors contain 207,060 payload bytes.
Clearing only their dictionary releases 273,408 bytes (267 KiB) of live CUDA
allocation. This is additional resident GPU storage, not a peak-memory saving;
both paired timing arms already hold it. The cache scales with response tokens
for the rows and devices actually used within the cell. A separate profile
confirms 16 tensor constructions in the warmed scoring call versus 48 in the
reference. The first memory-audit harness failed because its profiler-counter
parser used the wrong function name; the corrected run checked the raw profile
entries and completed successfully.

The revised targeted suite passed 141 CPU tests, including eight focused
response-tensor cases. The vendor environment passed all 74 worker tests and
the 90-case real-model CPU matrix. All four real baseline GPU cases passed
(193.20 s), with exact scores and gradients against the extended independent
reference. Tests cover unequal byte weights, reordered and partial batches,
alpha zero/0.25/one, inference-mode scoring before backward, tensor identity,
reused sample IDs on reprepare, preparation failure, close and objective
switching. Eight deliberate faults failed their intended checks, including
removing inference-mode protection. Coverage reaches all nine added executable
lines and both cache branches; it is not whole-worker coverage. New tests and
the reference pass type checking; the worker retains the same pre-existing
optional-`logits` diagnostic reproduced on its parent.

Profiles, source hashes, numerical tensors, raw paired samples and the memory
audit are retained under `out/axbench-gcg-performance/loop12/`. Results ending
in `-final` use the implementation with inference-mode protection; the earlier
ordinary-mode probes are preserved separately.

## Activation-L2 decoder depth

Activation-L2 consumes a configured decoder block's residual; later blocks
cannot affect that already-produced residual in the inspected sequential
forwards. The SmolLM3 workload captures zero-based block 28 of 36. A reference
scoring trace executed all 36 blocks four times. Blocks 29–35 accounted for
62.857 ms of the operation's 334.726 ms of correlated device time. The actual
implementation executes 29 blocks, with no calls after the target. The raw
CUDA kernel count falls from 8,772 to 7,172, and summed kernel durations from
333.657 ms to 277.844 ms. Correlated device time includes transfers; these
profile figures are separate from the synchronized paired latency results.

The change is confined to activation capture for Llama, Qwen2, SmolLM3 and
GPT-NeoX. It temporarily registers a prefix of the existing decoder blocks,
then restores the original `ModuleList` object in `finally`. Block objects,
parameters, configuration, positions, masks, interventions, one-position
LM-head handling, losses and backward arithmetic retain their existing paths.
The final norm and one-position vocabulary projection still run; their unused
logits are outside the activation-L2 objective.
Response-margin and generation use the full decoder. Unrecognized families
or containers, and targets that require no proper prefix, keep the original
full-depth behavior. There is no new configuration, model copy or kernel.

The worker already requires one nonnegative configured steering layer and
loads the base model onto its configured device. Each worker processes
requests sequentially. The prefix therefore has the same lifetime as the
existing activation-capture call; it is restored before the returned hidden
state is differentiated or the next request can generate text.

The independent reference is the complete `_capture_hidden` body from
`700f84f6c5629d53d4063693cd645c548d191e92`. Its test copy matches that body by
AST comparison, allowing only the signature/name and four module-helper
bindings. Each benchmark also checks all 144 clean class-row activations
against a full-depth preparation at both suffix lengths.

The local paired workload uses the same pinned SmolLM3 3B source and vendor
environment as the preceding measurements: RTX 4090, 72 prepared pairs,
four selected members, physical batch four, two candidates, k=1/32, three
warm pairs and 21 alternating-order measured pairs. The actual production
worker SHA-256 is
`174b8f6f2155bfd67a56a16fd0d104d284e6243970657797cb8de3f5fb3b3d8c`.

Scoring:

| Method | Suffix tokens | Reference (ms) | Prefix (ms) | Median paired speedup | Faster pairs |
|---|---:|---:|---:|---:|---:|
| DiffMean | 1 | 363.3891 | 299.2812 | 1.21543× | 21/21 |
| DiffMean | 32 | 382.8911 | 314.6186 | 1.21698× | 21/21 |
| LinearProbe | 1 | 356.8874 | 293.4152 | 1.21594× | 21/21 |
| LinearProbe | 32 | 382.1240 | 313.6526 | 1.21861× | 21/21 |
| LoRA | 1 | 357.5369 | 293.5586 | 1.21847× | 21/21 |
| LoRA | 32 | 383.1631 | 314.2402 | 1.21949× | 21/21 |
| PromptSteering | 1 | 433.7553 | 355.5940 | 1.21984× | 21/21 |
| PromptSteering | 32 | 459.7665 | 376.7474 | 1.22071× | 21/21 |

Suffix gradients:

| Method | Suffix tokens | Reference (ms) | Prefix (ms) | Median paired speedup | Faster pairs |
|---|---:|---:|---:|---:|---:|
| DiffMean | 1 | 379.8359 | 347.8226 | 1.09243× | 21/21 |
| DiffMean | 32 | 414.7648 | 380.5996 | 1.08996× | 21/21 |
| LinearProbe | 1 | 380.8789 | 348.6796 | 1.09190× | 21/21 |
| LinearProbe | 32 | 413.2879 | 378.3336 | 1.09165× | 21/21 |
| LoRA | 1 | 381.5755 | 349.3402 | 1.09220× | 21/21 |
| LoRA | 32 | 413.4144 | 379.0077 | 1.09076× | 21/21 |
| PromptSteering | 1 | 467.9287 | 428.7184 | 1.09227× | 21/21 |
| PromptSteering | 32 | 504.1490 | 462.5627 | 1.08990× | 21/21 |

All 672 measured local numerical comparisons were exact. Every one of the
336 activation-L2 comparisons was faster. The unchanged response-margin
control had paired ratios of 0.99835–1.00098× and zero peak-allocation delta.
Ratios are medians of paired ratios, not ratios of separately reported medians.

Observed gradient peak-allocation reductions across the four local methods
were 1.228–1.570 GB. A separate DiffMean run collects Python garbage before
every arm, outside the timed interval, and records every warm/timed peak.
The reduction persists: 1,227,972,608–1,228,533,760 bytes at k=1 and
1,312,718,848 bytes at k=32. All 12 measured numerical comparisons in this
follow-up were exact. DiffMean scoring saves only 7,520,256–8,749,056 bytes.
The batch-eight Hopper/Blackwell runs observed gradient reductions of
2.460–2.618 GB. These are allocator peak measurements, not model-size changes.

The bounded DiffMean whole-search replay includes preparation and normal
128-token sampled generation with a fake judge; model loading is outside the
interval. It uses 72 prepared pairs, four selected members, physical batch two,
k=1, two steps, 16 proposals and top-k 32. Each objective ran one warm pair
and three alternating-order timed pairs.

| Objective | Full depth (s) | Prefix (s) | Median paired speedup | Faster pairs |
|---|---:|---:|---:|---:|
| `activation_l2` | 18.5111 | 16.1464 | 1.14624× | 3/3 |
| `response_margin` control | 17.1423 | 17.1409 | 0.99946× | 1/3 |

Across all eight warm/timed pairs, every optimizer request, gradient hash,
and all 1,872 raw score comparisons matched exactly. Search receipts and
sampled generations were byte-identical. L2 scoring's stage median fell from
9.5422 s to 7.8207 s and preparation from 5.3625 s to 4.7763 s. This is a
bounded single-concept replay, not a judged quality campaign or a timing claim
for every checkpoint.

The identical production source also passed two larger-GPU runs, each using
72 prepared pairs, eight selected members, physical batch eight, k=1/32,
three warm pairs and 21 measured pairs:

| GPU / Slurm job | Methods timed | L2 scoring speedup | L2 gradient speedup | Exact measured pairs |
|---|---|---:|---:|---:|
| H200 / `23024383` | DiffMean, LoRA | 1.19569–1.19899× | 1.08267–1.08874× | 168/168 |
| RTX PRO 6000 Blackwell / `23024385` | DiffMean | 1.21306–1.21536× | 1.08797–1.08904× | 84/84 |

Hopper used Torch 2.5.1/CUDA 12.4 and the nonpreemptible `mit_normal_gpu`
partition with account/QoS `mit_amf_advanced_gpu`; Blackwell used the existing
Torch 2.7.1/CUDA 12.8 environment on `mit_preemptable`, with no requeue.
Both used Python 3.12.1, Transformers 5.16.1, PyVene 0.1.8 and PEFT 0.13.2.
Hopper passed 119 targeted CPU/model tests. Blackwell passed 185 targeted
CPU/model tests and all four real baseline GPU tests (247.54 s), including
normal generation and the extended numerical oracle. There were 160/168
faster timed calls on Hopper and 84/84 on Blackwell. These remote timings
measure components; the full-search timing above is local.

The new unit cases cover first/middle/last target blocks, full-depth exact
hidden states and input gradients, block execution counts, original container
identity, parameter/state-dictionary identity, mixed modes, hook registries,
normal forwards before/after capture, nested scopes, unsupported structures,
and primary failures in the retained target or head. Real PyVene restoration
checks use an actual skipped block. All four model families, four methods,
both objectives and FP32/BF16 run through the extended independent reference.

The targeted local suite passed 162 tests (one vendor launcher deselected),
and the vendor real-model CPU matrix passed all 90 cases. All four real
baseline GPU tests passed locally (187.35 s), including the extended exact
oracle and normal generation. Eight deliberate faults failed their intended
assertions. Coverage reaches every added helper
statement (14) and branch (four); this is not whole-worker coverage. The new
tests and reference pass type checking. The worker retains the same existing
optional-`logits` diagnostic verified on its parent.

Profiles, numerical tensors, source hashes, raw paired measurements, full
optimizer traces, coverage, mutation logs and remote results are retained
under `out/axbench-gcg-performance/loop14/`. The prototype measurements are
kept separately from results ending in `-final`.

## Activation-L2 with no projected logit positions

After limiting decoder depth, activation capture still projected the final
position through the vocabulary head. A paired trace of the actual previous
implementation attributed 8.967 ms to four such projections in one DiffMean
scoring call. Those logits are not consumed by activation-L2. The current
implementation keeps the normal head call but gives it zero token positions;
the corresponding trace records zero device time inside all four head calls.

The inspected Llama, Qwen2, SmolLM3 and GPT-NeoX forwards accept a tensor of
logit-position indices. Capture passes a fresh empty one-dimensional Long
tensor on the embedding device. An integer zero would instead select **all**
positions, so it is not interchangeable with the empty tensor. The private
forward adapter explicitly recognizes tensors before testing the integer
argument; it never asks for an empty tensor's Boolean value. Other model
families retain one-position projection. The existing decoder-prefix scope,
collector, interventions, masks, losses and gradient arithmetic are preserved.
The head still executes its normal module call, including hooks and exceptions.
Response-margin and generation retain their existing full-logit paths. No
cache, model replacement, new setting, or device-specific kernel is added.

The incremental reference freezes `_capture_hidden` and `_forward` from
`d77d25518a510138c31d9243ab041949638ce6c6`, including its decoder prefix and
one-position projection. The benchmark compiles both frozen function bodies,
while the permanent numerical tests also compare against the earlier full-depth,
full-logit, rowwise-backward reference. The decoder-prefix helper is unchanged
by AST comparison. Both preparations compare all 144 clean class-row residuals
at k=1 and k=32.

The actual production worker SHA-256 is
`2e5e4067f1364328176f187d385eebfe7bc6ea9a13422317f25f70140480fc8b`.
Local measurements use the pinned SmolLM3 3B source on RTX 4090, 72 prepared
pairs, four selected members, physical batch four and two candidates. Each
objective/length/operation has three warm pairs and 21 alternating-order,
synchronized measured pairs. All tables below compare this source directly
with the one-position parent; the earlier wrapper-based prototype is stored
separately. The first component runs shared the new argument-forwarding guard
between arms. Those samples remain under `*-shared-guard/`; the accepted
timing runs freeze both changed functions, including for response-margin.
The benchmark verifies reference routing and restores both functions after
normal exit or an injected exception before loading the model.

Scoring:

| Method | Suffix tokens | Reference (ms) | No logits (ms) | Median paired speedup | Faster pairs |
|---|---:|---:|---:|---:|---:|
| DiffMean | 1 | 295.1430 | 287.4662 | 1.02693× | 21/21 |
| DiffMean | 32 | 314.5567 | 306.7793 | 1.02561× | 21/21 |
| LinearProbe | 1 | 294.7099 | 286.6673 | 1.02747× | 21/21 |
| LinearProbe | 32 | 315.7001 | 307.8789 | 1.02440× | 21/21 |
| LoRA | 1 | 294.5941 | 285.3995 | 1.03117× | 21/21 |
| LoRA | 32 | 315.1899 | 306.0304 | 1.02938× | 21/21 |
| PromptSteering | 1 | 356.0175 | 346.7470 | 1.02594× | 21/21 |
| PromptSteering | 32 | 377.3033 | 367.8237 | 1.02573× | 21/21 |

Suffix gradients:

| Method | Suffix tokens | Reference (ms) | No logits (ms) | Median paired speedup | Faster pairs |
|---|---:|---:|---:|---:|---:|
| DiffMean | 1 | 349.6684 | 345.5092 | 1.01220× | 21/21 |
| DiffMean | 32 | 381.4479 | 377.3314 | 1.00960× | 20/21 |
| LinearProbe | 1 | 349.9369 | 345.9751 | 1.01115× | 21/21 |
| LinearProbe | 32 | 380.9049 | 376.2883 | 1.01065× | 20/21 |
| LoRA | 1 | 349.8209 | 345.4244 | 1.01267× | 21/21 |
| LoRA | 32 | 378.8752 | 374.4930 | 1.01229× | 20/21 |
| PromptSteering | 1 | 428.6796 | 424.1632 | 1.01040× | 21/21 |
| PromptSteering | 32 | 462.0181 | 457.6763 | 1.00976× | 21/21 |

All 672 measured local numerical comparisons were exact. All 168 L2 scoring
calls and 165/168 gradient calls were faster. The response-margin control
had paired ratios of 0.99890–1.00125×. Peak CUDA allocation was unchanged
in every local case. Ratios are medians of paired ratios, not ratios of the
separately reported latency medians.

The first bounded DiffMean whole-search replay includes preparation and normal
128-token sampled generation with a fake judge; model loading is excluded.
It uses 72 prepared pairs, four selected members, physical batch two, k=1,
two search steps, 16 proposals and top-k 32. Both changed function bodies
are frozen from the parent for the reference arm. Each objective has one
warm pair and three alternating-order measured pairs.

| Objective | One position (s) | No logits (s) | Median paired speedup | Faster pairs |
|---|---:|---:|---:|---:|
| `response_margin` control | 17.0101 | 17.2109 | 0.98834× | 0/3 |
| `activation_l2` | 16.1735 | 15.9079 | 1.01670× | 3/3 |

Every optimizer request, gradient hash and all 1,872 raw score comparisons
matched exactly across the eight warm/timed pairs.
The persisted search receipts and sampled generations were byte-identical.
This is a bounded single-concept replay with a fake judge, not a judged
quality campaign or a qualification of every checkpoint.

The response-margin control was slower in that first replay, with its
difference concentrated in preparation. A second replay instruments the
producer and client preparation stages. Its response-margin median ratio
is 1.00098×; unchanged canonical token-pool construction varies from
0.532 to 0.740 s, with both labeled arms exhibiting faster and slower calls.
No cause is assumed for that CPU variation. The second L2 ratio is
1.01876×, but instrumentation and only three timed pairs limit the
precision of either whole-search estimate. The repeat also preserves every
optimizer request, gradient, 1,872 raw scores, receipt and generation exactly.
Both runs are retained; the performance claim rests on the repeated paired
component measurements, with whole-search timings reported as bounded observations.

The same production source ran on H200 and RTX PRO 6000 Blackwell. Both timing
runs use DiffMean, eight selected members, physical batch eight, k=1/32, three
warm pairs and 21 alternating-order measured pairs. Both reference functions
are frozen from the parent, and all 144 clean class-row activations are exact
at each suffix length.

| GPU / Slurm job | L2 scoring speedup | L2 gradient speedup | Exact measured pairs | Faster calls |
|---|---:|---:|---:|---:|
| H200 / `23030142` | 1.00580–1.00858× | 1.00317–1.00349× | 84/84 | 76/84 |
| RTX PRO 6000 Blackwell / `23030143` | 1.01978–1.02286× | 1.00610–1.00782× | 84/84 | 82/84 |

Hopper uses Torch 2.5.1/CUDA 12.4; Blackwell uses Torch 2.7.1/CUDA 12.8.
Both use Python 3.12.1, Transformers 5.16.1, PyVene 0.1.8 and PEFT 0.13.2.
The H200 job used nonpreemptible `mit_normal_gpu` with account/QoS
`mit_amf_advanced_gpu`; Blackwell used `mit_preemptable` with no requeue.
The earlier seven-pair timing runs shared the forwarding guard and are
superseded by the jobs above. Their identical-source test qualifications
remain valid: H200 job `23028326` passed 122 targeted CPU/model cases;
Blackwell job `23028329` passed 188 targeted CPU/model cases and all four
real-worker GPU cases (398.78 s), including normal generation and the extended
independent numerical oracle. Remote timings are component measurements;
the full-search timing is local.

All four real baseline GPU tests passed locally (187.67 s), including
normal generation and the extended exact oracle. The focused local suite
passed 165 tests (one vendor launcher deselected), and the real-model CPU
matrix passed all 90 cases. The updated unit fixture
implements the same integer-or-index-tensor API as the inspected Transformers
forwards; its old integer-only shortcut caused three initial fixture failures.
The real-model matrix already passed before that fixture correction.

The tests explicitly require zero projected positions for each of the four
families, preserve one-position fallback for an unknown or absent model type,
and check an inference-mode capture followed by normal backward. Existing
first/middle/last-layer, hook/container identity, exception, generation,
FP32/BF16, PyVene and LoRA checks continue against the independent references.
Eight isolated mutations failed their intended checks: retaining one position,
using floating-point indices, qualifying every family or only Llama, asking an
empty tensor for its Boolean value, omitting the tensor argument, and losing
the selector on either the additive or plain forward route.

Combined unit and real-model coverage reaches all four changed selection and
argument-forwarding statements and all four outgoing branches of their two
conditions. This is not whole-worker coverage. Ruff passes, and the extended
capture tests pass type checking. Configured worker type checking retains the same
optional-`logits` diagnostic as its parent; the older tokenizer test fixture
also retains its pre-existing method-override diagnostic, reproduced on the
parent. No full repository test run was performed.

Source snapshots, profiles, exact numerical tensors, raw paired timings,
optimizer traces, mutation/coverage records and remote qualification artifacts
are retained under `out/axbench-gcg-performance/loop15/`.

## Canonical token-ID normalization

A preparation profile on the preceding implementation calls `_ids` 124,836
times, with 0.162 s cumulative instrumented CPU time. Most canonical token
round trips return a built-in list containing one built-in integer. The
worker now returns that integer as a tuple directly. The original mapping,
tensor, nesting, element-type and error checks remain the fallback. Exact
type checks are intentional: list subclasses can override indexing or length,
and Boolean values are integer subclasses but invalid token IDs. This change
adds three source lines and no cache, configuration or tensor operation.

The accepted measurement uses the actual production worker, SHA-256
`7b5881eaa62bfdbe2c28b9945395ebbf4ac25dad1d83fe6b3711c760bb1f074a`,
against the frozen `_ids` body from
`5d786892d0db69b957683a58e7b2b7dbeb58a695`. It rebuilds the complete allowed-ID
pool using the pinned SmolLM3 tokenizer, revision
`cfb32d505f5025ec9be4e704f70cfbf5bdf8da94`, on the local Ryzen 9950X host with
Python 3.12.11. There are three warm pairs and 21 measured alternating-order
pairs; explicit garbage collection is outside each timed interval.

| Operation | Reference (s) | Singleton path (s) | Median paired speedup | Faster pairs |
|---|---:|---:|---:|---:|
| Build the complete canonical token pool | 0.595668 | 0.557551 | 1.06445× | 19/21 |

All 21 measured pairs produced the same ordered 122,908 token IDs exactly.
Their JSON tuple SHA-256 is
`17e9e9568a43dd0ce16e79366d96ce0391c1c07074771858ff74ce8c90b2b184`.
The speedup is the median of paired ratios; this is a CPU preparation component
measurement, not a whole-search speedup. An earlier wrapper-based prototype
also preserved the pool and is retained separately.

The real RTX 4090 DiffMean replay freezes the same parent helper and checks
that both encoding routes actually use the selected arm. Both objectives use
72 prepared pairs, four selected members, physical batch two, k=1, two search
steps, 16 proposals, top-k 32 and normal 128-token sampled generation with a
fake judge. Across eight warm/measured pairs, every optimizer request,
gradient hash and all 1,872 raw scores match exactly. Persisted search receipts
are byte-identical and generation strings match exactly. A second CUDA
process was observed during this replay at 21:41:23 UTC on 2026-09-18; its
workload and timing effect are unknown. The raw timings are retained but
explicitly disqualified for performance claims. This does not affect the
separate, earlier CPU pool measurement or bypass any numerical assertion.

The focused checks passed 95 cases. The broader targeted root run passed
194 cases and failed its existing frozen/offline vendor launcher because
the locked Torch 2.7.1 CUDA 12.8 wheel was not cached. The same launcher
failure was reproduced on the untouched parent. Running its underlying
real-model matrix directly with the installed vendor environment passed all
90 cases. All four real baseline GPU cases passed on RTX 4090 (195.57 s),
including the independent full numerical reference and normal generation.
No full repository suite was run, and these overlapping test runs are not
an additive count of distinct tests.

The new tests compare the frozen normalization reference on 14 valid inputs,
14 explicit malformed inputs and 172 bounded container variants. They cover
mapping/tensor/nested forms, exact error classes and messages, Boolean values,
large and negative integers, and list subclasses with overridden indexing or
length. Seven isolated mutations fail their intended assertions. Coverage of
the complete `_ids` helper reaches all 13 statements and 12 branches. Ruff
and type checking of the changed tests pass. Configured worker type checking
retains the parent's existing optional-`logits` diagnostic, reproduced on
both revisions.

The same investigation measured two alternate orders for copying the full
embedding snapshot to CPU. All 84 measured comparisons were exact and both
alternatives avoided a temporary 1,050,673,152-byte CUDA allocation. However,
their paired speedups were only 0.637–0.643× with deterministic execution
enabled, versus 1.175–1.195× when disabled. Neither alternative was promoted;
the performance tradeoff reverses with the execution setting. These are
component observations, not a qualification of complete preparation on
other hosts.

No Hopper or Blackwell rerun was performed for this pure Python normalization
change. The prior GPU qualifications remain recorded above; this iteration
adds no CUDA kernel or model-forward change. Profiles, frozen source,
mutation and coverage records, paired pool samples, the rejected copy-order
probe and exact search traces are under
`out/axbench-gcg-performance/loop16/`.

## ASCII checks in both canonical token pools

The preparation profile also identified per-character Unicode checks in
token-pool construction. The baseline helper now handles only exact built-in
ASCII strings with `isascii()` and `isprintable()`. ASCII is already NFC;
within that range the printability test rejects the same controls and keeps
ordinary space. Non-ASCII strings and string subclasses continue through the
original policy. Python's general Unicode printability predicate would also
reject separators, private-use and unassigned characters that this benchmark
allows, so it must not replace the full policy. See the
[Python string-method definitions](https://docs.python.org/3.12/library/stdtypes.html#str.isprintable).

The project preparation loop uses the same narrow ASCII shortcut while
retaining its own Unicode checks. In particular, the project pool does not
require NFC, whereas the baseline pool does. Strict UTF-8 validation, special
IDs, token order, round-trip encoding and the per-preparation lifetime remain
unchanged. Both existing sites are edited directly; there is no new shared
policy layer, cache, setting or tensor operation.

A paired cProfile probe of the actual baseline helper reduces Unicode
category calls from 729,942 to 102,919 and NFC normalization calls from
127,999 to 30,154. Both arms yield the same 124,259 eligible `(ID, text)`
pairs, of which 94,396 contain ASCII text. These are call counts; instrumented
timings are not used for the performance claim.

The accepted pool benchmarks freeze the preceding implementation from
`52d49aa6c00f70a05e1ded97f1eadc1c65ed8853` and use the actual new source.
The baseline benchmark freezes the complete eligibility helper; the project
benchmark extracts the exact pool loop from each preparation implementation.
Each has three warm pairs and 21 alternating-order measured pairs on the
local Ryzen 9950X host, with explicit garbage collection outside the intervals.

| Complete pool | Parent median (s) | ASCII path median (s) | Median paired speedup | Faster pairs |
|---|---:|---:|---:|---:|
| SmolLM3 baseline, Python 3.12.11 | 0.522090 | 0.476411 | 1.09231× | 21/21 |
| Project RustBPE, Python 3.13.2 | 0.060328 | 0.042938 | 1.40304× | 21/21 |

All 42 measured comparisons preserve the ordered pools exactly: 122,908
baseline IDs and 32,461 project IDs. The baseline uses model/tokenizer commit
`cfb32d505f5025ec9be4e704f70cfbf5bdf8da94`; the project uses the cached
32,768-entry RustBPE tokenizer used by the representative checkpoint.
Their pool JSON SHA-256 values are respectively
`17e9e9568a43dd0ce16e79366d96ce0391c1c07074771858ff74ce8c90b2b184` and
`1a8750f08365ed8b2a8f7dcbe6dbd155851519b2e0b005046329fb694c1395ca`.
Ratios are medians of paired ratios. These are CPU pool-construction gains,
not claimed whole-search speedups; earlier prototypes are retained separately.

The local targeted root suite passed 249 cases (two deselected), and the
shared SelfConditioning/NL runtime checks passed 127 cases. The focused text
suite passed 111 cases, while the baseline pool tests also passed all 81 cases
in Python 3.12. These runs overlap and are not additive counts. No full
repository suite was run.

Each pool is checked on all 16,513 ASCII strings of length zero, one or two,
with an independent expected set of 9,120 retained strings. Baseline checks
exercise both fast and scalar tokenizers. Another 29 named cases cover
Unicode category boundaries, NFC versus compatibility normalization, long
ASCII text and overridden string methods, iteration and encoding. The project
tests enter public preparation and compare the resulting ordered pool and
stored membership set with a frozen parent loop; its body was verified by AST.

All four real baseline GPU tests passed on RTX 4090 (194.29 s), including
the original numerical oracle, complete pool equality and normal generation.
The representative project checkpoint passed (90.28 s) for SelfConditioning
and SelfConditioningMasked, both objectives and k=1/32. All eight project
preparations compare their complete ordered pool against the frozen reference;
the existing independent score/gradient and generation checks also pass.
This permanent project integration assertion is included in the change.

A separate bounded DiffMean replay freezes the complete preceding baseline
eligibility helper and checks its routing. Both objectives use 72 pairs,
four selected members, physical batch two, k=1, two search steps, 16 proposals,
top-k 32 and normal 128-token sampled generation with a fake judge. Every
optimizer request, gradient hash, all 1,872 raw scores, eight paired search
receipts and sampled generations match exactly. This replay is for numerical
and persisted-output parity; its raw timings are not performance evidence.

Thirteen deliberate faults fail their intended checks, including admitting
ASCII controls, dropping ordinary spaces, applying printability to Unicode or
string subclasses, and changing either pool's normalization policy. The
unmodified project control passes. The initial scratch mutation harness copied
module globals and bypassed a fixture's embedding patch; those unrelated
failures were excluded, the harness was corrected to use live module globals,
and the checks were repeated. Both runs are retained for review.

Coverage reaches all 20 statements and 14 branches of the complete baseline
eligibility helper, and all five statements and six branches in the changed
project text block. Whole-worker or whole-project-preparation coverage is not
claimed. Ruff, project-adapter type checking and all changed-test type checks
pass. Configured baseline-worker type checking retains the preceding source's
optional-`logits` diagnostic. The first GPU launcher invocation used the vendor
environment for the root driver and failed on its missing root dependency;
the corrected root-driver/vendor-worker invocation is the passing run above.

No new Hopper/Blackwell kernel qualification was run for these Python-only
checks. The preceding stack's larger-GPU results remain separate. Optional NL
uses the same project preparation path and has targeted CPU runtime coverage;
no real NL artifact qualification is claimed here.

The source SHA-256 values are
`1b14a009ad9b5788fb7e67e764b210cb72f0d7bf2b36696380a5eac1b2c08d2f`
for the baseline worker and
`4f61b800a127e192afac433c34a162f1d130d787fb861d325faa99e1e936787b`
for the project adapter. Evidence is retained under
`out/axbench-gcg-performance/loop17/`.

## Correctness checks

The test-only rowwise VJP oracle differentiates each original scalar loss and
asserts that all other batch rows have zero gradient. A deliberately coupled
example must fail this precondition. CPU adapter tests also compare separate
scalar forwards, which catches erroneous averaging before backward. Both
objectives, alpha zero/0.01/one, uneven lengths, member permutation, partial
batches, and batch sizes one/two/four/eight are covered.

The vendored integration suite checks actual Llama, Qwen2, SmolLM3, and GPT-NeoX models,
real PyVene additions and PEFT adapters, FP32 and BF16, and both objectives.
The real SmolLM3 CUDA tests check every gradient consumed by the bounded
optimizer against the rowwise oracle inside the worker, including tensor
sidecar descriptors. They exercise three search steps, four proposals, top-k
32, two training pairs, and normal final generation with a fake judge. This is
a bounded regression check, not a judged quality campaign.

State-guard tests cover all seven hook registries, initially empty and
nonempty contents, replaced registry objects, existing removable handles,
hook order, aliased roots, mixed module modes, existing gradient contents,
and exceptions. The CUDA failure check also consumes device RNG state and
replaces an existing gradient before checking restoration. The real-model
reference forwards use the frozen original guard alongside the original
decoder-depth/LM-head/backward paths. The paired timing helper has a
deliberately shared, overwritten result test to ensure that numerical differences cannot be
hidden by a subsequent operation.

Focused CPU/CUDA coverage reached all 69 statements and 34 branches in the
guard, and all 29 statements and six branches in the paired timer. Seven
deliberate mutations failed the intended assertions: leaked new hooks,
lost registry identity, lost existing hooks, leaked training mode, leaked
backward-hook flags, sorted-away hook order, and an omitted independent
numerical snapshot. The current hook-order/timer tests and these mutations
also ran in the frozen Torch 2.7 environment (Slurm job `23003668`).
All four real baseline GPU tests then passed with the extended references
(178.81 s). The real project test passed at its original physical gradient
batch size one and default compiler settings (79.38 s), covering both
SelfConditioning methods, both objectives, suffix lengths one/32, exact
reference comparisons, and generation before and after GCG.

## Cumulative comparison with current main

A separate audit compares the complete GCG worker and state guard from main
`b453e530ea501aeec1de0d41016117633f719b71` with stack revision
`387f8ead1fef920915523a4098b55ff880f882ac`, before the collector-metadata
change below. Both reference modules have independent globals, including the
frozen worker's imported state guard. The runtime, driver and model loader are
unchanged between those revisions. Each implementation prepares its own session
on the same loaded model. Entire preparation responses, allowed pools and every
clean activation match exactly.

The workload is DiffMean on the same SmolLM3 commit and 72-pair source described
above, with both objectives and one/32 suffix tokens. The 4090 uses four members
and physical batches of four; the H200 uses the production defaults of 16 members
and physical batches of 32. Each operation has three warm pairs followed by
21 measured pairs on the 4090 or seven on the H200. Execution order alternates
and every paired result is compared with zero relative and absolute tolerance.
Both use PyTorch 2.5.1/CUDA 12.4 and the same Transformers, PyVene and PEFT versions.

| GPU / members / batch | Objective | Gradient k=1 / k=32 | Scoring k=1 / k=32 |
|---|---|---:|---:|
| RTX 4090 / 4 / 4 | `response_margin` | 2.660× / 2.662× | 1.131× / 1.134× |
| RTX 4090 / 4 / 4 | `activation_l2` | 2.890× / 2.874× | 1.454× / 1.456× |
| H200 / 16 / 32 | `response_margin` | 12.834× / 12.493× | 1.126× / 1.122× |
| H200 / 16 / 32 | `activation_l2` | 17.569× / 16.988× | 1.421× / 1.410× |

All **224 measured gradient/score pairs** match exactly: 168 on the 4090 and
56 in H200 job `23040606` on `node3200`. These are median paired worker-call
speedups. Preparation and complete searches are not timed in this audit. Both
prepared sessions coexist, so the retained peak counters are not isolated
memory-savings measurements. Larger physical batches increase the benefit of
replacing rowwise backward traversals, and these matched workloads do not
constitute a hardware-speed comparison.

Reference source hashes, preparation checks, full numerical tensors and raw
paired timings are retained under `out/axbench-gcg-performance/loop18/`.

## Activation collector dimension on the host

For DiffMean and LinearProbe activation-L2, PyVene 0.1.8 uses the collector's
`interchange_dim` buffer as a Python slice bound. `set_device` moves this scalar
to CUDA along with the other intervention buffers. Each collection then needs
a device-to-host scalar read before constructing the full-width slice.

The worker retains the collector object and sets this fixed bound to an explicit
CPU `int64` tensor after device placement. It uses the model's existing hidden
width. The intervention, full residual tensor, slice extent and differentiation
path are otherwise unchanged. This requires no new runtime option or device
dispatch. The final-source DiffMean profile at four members/batch four/k=1
reduces `cudaStreamSynchronize` calls from 30 to 28 for the gradient and 52 to
48 for scoring. `aten::item` counts remain 18/36 and `aten::mm` counts remain
814/816: the same scalar conversion now reads host memory. Time attributed to
the original Python slice includes GPU waiting, so it is not a savings estimate.

Uninstrumented RTX 4090 measurements compare the complete worker and state guard
from parent `387f8ead1fef920915523a4098b55ff880f882ac` against the final source,
using independent module globals and prepared sessions on the same model. Each
case uses all 72 preparation pairs, four members, physical batch four, three
warm pairs and 21 measured pairs with alternating order. The activation-L2
scoring results are:

| Method | Suffix tokens | Parent median | New median | Median paired speedup | Faster pairs |
|---|---:|---:|---:|---:|---:|
| DiffMean | 1 | 286.867 ms | 285.190 ms | 1.00471× | 21/21 |
| DiffMean | 32 | 306.511 ms | 304.974 ms | 1.00368× | 17/21 |
| LinearProbe | 1 | 290.355 ms | 288.127 ms | 1.00765× | 20/21 |
| LinearProbe | 32 | 305.763 ms | 304.468 ms | 1.00422× | 19/21 |

Gradient effects are smaller and mixed: median paired speedups are 1.00127× /
0.99969× for DiffMean k=1/32 and 1.00328× / 1.00252× for LinearProbe. The unchanged
response-margin control cases range from 0.99851× to 1.00043× across scoring and
gradients. All **336 measured comparisons** are exact, with zero relative and
absolute tolerance. Complete preparation responses and all 144 clean activations
per L2 case also match. Both sessions coexist; their peak-memory counters are
not an isolated memory-savings measurement.

The same source passed **112 exact measured comparisons on each larger GPU**:
H200 job `23041865` (`node3200`, Torch 2.5.1/CUDA 12.4) and RTX PRO 6000 Blackwell
job `23041867` (`node5004`, Torch 2.7.1/CUDA 12.8). These use both additive methods,
both objectives, k=1/32, 16 members and physical batch 32, with three warm pairs
and seven measured pairs per operation. Their small timing changes are mixed;
no larger-GPU speedup is claimed for this increment. Each job also passed the
122-case targeted model suite, including all 32 new collector cases.

The new cases exercise actual Llama, GPT-NeoX, Qwen2 and SmolLM3 models with
DiffMean/LinearProbe, CPU/CUDA and ordinary/inference-mode preparation. They
check the full-width bound, unchanged device placement of other buffers and
parameters, exact clean residuals, exact nonzero suffix VJPs, exception
restoration and session cleanup. The reference independently reconstructs the
parent's model-device bound and also extends the real-worker numerical oracle.
Final-source local checks passed all 32 new cases, 65 focused root tests and all
four real baseline GPU tests (186.17 s). These test sets overlap and are not an
additive coverage total. Eight deliberate mutations fail the intended checks;
tracing reaches all three changed executable locations, with no new branch.
This is changed-code evidence, not whole-file coverage. No full `tests/` run was
used. Scoped `ty` checks reproduce the same seven diagnostics as the untouched
parent, including a missing `pytest` import in the base vendor environment;
pytest is supplied by the test command's uv overlay. There are no new diagnostics.

Four paired complete DiffMean searches also match exactly: one warm pair and
one additional pair for each objective, with all 72 training pairs, four members,
physical batch two, a two-step limit, 16 proposals, top-k 32 and one suffix token.
Every optimizer request and gradient sidecar, all **936 raw score entries**,
the byte-for-byte search receipts and sampled generation results with a 128-token
budget match the frozen parent preparation path. The judge is fake. These are bounded
correctness checks; their retained timings do not establish a whole-search gain.

The final worker SHA-256 is
`fdbff2dc411b187306c1cf1c10d52da64cee704262e17cc19a5174d8b6946e73`.
Frozen reference sources, preparation comparisons, raw timings, numerical
tensors, losslessly compressed profiles, mutation results and test logs are
retained under `out/axbench-gcg-performance/loop18/`. Earlier prototype and
intermediate-setup runs are kept separately from these final-source results.

## Clean-reference copies before the baseline forward

Each activation-L2 physical batch now finishes its ordinary blocking copies of
clean CPU references before the model forward. The loss uses the same references
in the same row order, with unchanged arithmetic, dtype and alignment. The shared
loss still transfers a reference to the captured residual's device if necessary.
The temporary copy list is local to one physical batch; the prepared CPU cache
is neither replaced nor changed. This applies to DiffMean, LinearProbe, LoRA and
PromptSteering. Project SelfConditioning paths are unchanged by this increment.

The parent is `ce08487e7840c61f927a0239e197f0091e301c95`. The only production
function changed is `GCGSession._side_batch`; the final worker SHA-256 is
`e452e6ff5105f871fe74f3cd34e0624ff0d5db34b53286086975520f098be100`.

An initial asynchronous-copy prototype matched all 84 measured comparisons but
showed only 1.00108–1.00312× scoring speedups and nearly flat gradients. Its trace
removed explicit synchronization calls while most host waiting remained inside
the pageable-memory copy call. That variant was not promoted. The selected
prototype keeps blocking copy semantics and moves the copies ahead of forward.
Its whole-worker AST equals the final production AST; production adds two
explanatory comments. At four members/batch four/k=1, its gradient/scoring traces
retain 28/48 `cudaStreamSynchronize` calls, 644/72 `cudaMemcpyAsync` calls and
814/816 `aten::mm` calls, matching the parent. CPU time attributed to clean-copy
calls moves out of the interval between forward and loss dispatch. This
attribution shift is not itself a speedup measurement.

The paired numerical reference loads the complete parent worker and state helper
into independent module globals. Every preparation response and all 144 clean
activations per L2 case are compared exactly. Each operation uses three warm
pairs and alternating execution order. Local final-source measurements use
21 pairs, four members and physical batch four for every baseline, both
objectives and suffix lengths one/32. Both arms use the same tmpfs directory for
large transient tensor sidecars; compact completed evidence is archived between
runs. Memory counters include both prepared sessions, so their absolute peaks
are not isolated per-implementation memory measurements. The early copies stay
live through forward and can increase temporary GPU allocation; the matched
peak deltas are reported with the timings.

All **672 measured local comparisons are exact**. The following ratios are
parent time divided by new time; parentheses count faster pairs. Peak changes
are new minus parent, in MiB, for gradient/scoring respectively.

| Method | Suffix tokens | Gradient ratio | Scoring ratio | Peak increase, MiB |
| --- | ---: | ---: | ---: | ---: |
| DiffMean | 1 | 1.003811× (19/21) | 1.002420× (16/21) | 5.18 / 4.46 |
| DiffMean | 32 | 1.003065× (19/21) | 1.003574× (17/21) | 2.41 / 4.46 |
| LinearProbe | 1 | 1.002994× (20/21) | 1.003978× (19/21) | 5.18 / 4.46 |
| LinearProbe | 32 | 1.003591× (16/21) | 1.002128× (20/21) | 2.41 / 4.46 |
| LoRA | 1 | 1.004881× (20/21) | 1.002651× (19/21) | 4.84 / 4.46 |
| LoRA | 32 | 1.003672× (19/21) | 1.004128× (19/21) | 3.27 / 4.46 |
| PromptSteering | 1 | 1.002857× (21/21) | 1.001970× (18/21) | 4.02 / 3.80 |
| PromptSteering | 32 | 1.003360× (19/21) | 1.002557× (19/21) | 4.71 / 4.46 |

Response-margin control ratios span 0.99845–1.00025×; that path has no transfer
change. The L2 gains are small, with an observed temporary allocation cost of
2.41–5.18 MiB at this batch size. These results do not measure preparation or
establish an end-to-end search speedup.

Hopper job `23045700` and Blackwell job `23045703` both completed successfully.
Each ran the final source through all 122 targeted model tests and 112 exact
paired operation comparisons: DiffMean/LinearProbe, both objectives, suffix
lengths one/32, 16 members and physical batch 32, with three warm and seven
measured pairs. The H200 used Torch 2.5.1/CUDA 12.4; the RTX PRO 6000 Blackwell
used Torch 2.7.1/CUDA 12.8. Blackwell L2 scoring ratios were 1.00391–1.00499×
and gradient ratios 1.00267–1.00453×. H200 scoring ratios were
1.00261–1.00510×, with gradients ranging from effectively flat to 1.00532×.
H200 response-control variation reached a similar magnitude, so these results
qualify the hardware and numerics without establishing a reliable H200 speed
gain for this increment.

The matched L2 peak increases at batch 32 are 25.89–31.83 MiB on H200 and
30.45–36.91 MiB on Blackwell. Local and larger-GPU runs together provide
**896 exact measured operation comparisons**. Remote checkouts are git archives
plus the reviewed overlay, so their report's git commit field is null; the
stage manifest and worker/test file hashes identify their source instead.

The permanent numerical oracle independently retains the parent's activation-L2
row loop and copies after forward. It combines with the existing full decoder,
full vocabulary, rowwise backward, state-restoration and collector references.
The targeted model matrix passed 122 cases. Its 32 preparation cases were then
extended and passed again, covering CPU/CUDA, four model families, both additive
methods, ordinary/inference-mode preparation, scoring and gradients, and an
unchanged CPU reference cache. The focused root subset passed 71 tests, and all
four real baseline GPU tests passed (185.99 s). These test sets overlap.

Seven deliberate faults are rejected: reversed references, reusing the first
reference, omitting the last reference, indexing the first row for every loss,
an inverted objective fence, corrupted reference values and a persistent GPU
reference cache. An additional check bypasses the independent reference: the
value-corruption fault then escapes, demonstrating why the reference is needed.
Tracing reaches all five changed executable lines and both objective outcomes;
this is not whole-file coverage. Scoped `ty` checks reproduce the parent's seven
existing diagnostics, with none added by the new reference helper. No full
repository test suite was run.

Four paired complete DiffMean searches also pass against the frozen parent
batch implementation: one warm pair and one additional pair per objective,
72 training pairs, four members, physical batch two, a two-step limit,
16 proposals, top-k 32 and a one-token suffix. Every optimizer request and
gradient sidecar, all **936 raw score entries**, byte-for-byte search receipts
and sampled generations with a 128-token budget match exactly. The judge is
fake. These bounded searches qualify behavior; their timings are diagnostic
and carry no end-to-end speedup claim.

Initial local test commands exposed launcher setup issues rather than numerical
failures: one command included the root Hydra-dependent conftest, another included
the offline vendor-environment installer, and a real-worker child attempted to
resolve the newer lockfile environment. That resolver was stopped before GPU
testing. The passing runs use the installed vendor runtime explicitly, with
`--confcutdir tests/axbench_adv` for the standalone vendor matrix and
`UV_NO_SYNC=1 UV_OFFLINE=1` for nested real-worker commands. The local runtime
remains Torch 2.5.1/CUDA 12.4; the larger-GPU reports record their own environments.

Source snapshots, prototype decisions, compressed traces, raw paired measurements,
independent numerical tensors, mutation results and test logs are retained under
`out/axbench-gcg-performance/loop19/`.

## Strict token-ID validation

The parent `29f37f7f352a78d3255cb0ae15d6a19f7b7d49ac` repeatedly resumes a
Python generator to validate every token ID in each rendered candidate prompt.
Plain lists now use standard-library iterators for the same strict type-identity
checks. Booleans, integer subclasses and NumPy integers remain invalid.
The singleton fast path, container normalization and exact errors are preserved.
There is no new cache, configuration option, tensor arithmetic or GPU kernel.

A rejected first version also optimized list subclasses. A regression probe
found that it destroyed an ephemeral invalid item before closing its iterator,
whereas the reference closed the iterator first. The final implementation keeps
the literal original generator for subclasses. The new cleanup-order test rejects
that faulty version, and its measurements are archived separately.

A separate final-source filter profile records 290 normalizer calls in each
arm and identical results. Python profile calls fall from 79,754 to 4,932,
accounted for by removing 74,822 generator resumptions for plain lists. These
profiled durations are excluded from the timing claims below.

### RTX 4090 workstation

This final-source comparison uses SmolLM3-3B commit
`cfb32d505f5025ec9be4e704f70cfbf5bdf8da94`, the same 72 prepared pairs as
the preceding experiments, and independent complete parent worker/state modules.
Each complete filter call validates two candidates against all 144 prompts per
candidate. Each case has three warmup pairs and 41 measured pairs in alternating
order. The model-call controls use four members, physical batch four, seven
measured pairs and three warmup pairs. Loading and preparation are excluded.
Python is 3.12.11, Torch 2.5.1+cu124, Transformers 5.16.1, PyVene 0.1.8 and
PEFT 0.13.2. Tokenizer parallelism, Rayon, OMP and MKL thread overrides are unset.
Timings run without overlapping local tests or GPU work.

The table reports host-CPU filtering, in milliseconds. Speedup is the median
of paired parent/new ratios, rather than a ratio of independent medians.

| Method | Objective | Suffix tokens | Parent (ms) | New (ms) | Paired speedup |
|---|---|---:|---:|---:|---:|
| DiffMean | Response margin | 1 | 8.979 | 8.651 | 1.0429× |
| DiffMean | Response margin | 32 | 10.079 | 9.702 | 1.0460× |
| DiffMean | Activation L2 | 1 | 9.264 | 9.021 | 1.0429× |
| DiffMean | Activation L2 | 32 | 10.066 | 9.741 | 1.0381× |
| LinearProbe | Response margin | 1 | 8.941 | 8.620 | 1.0397× |
| LinearProbe | Response margin | 32 | 10.034 | 9.589 | 1.0450× |
| LinearProbe | Activation L2 | 1 | 9.053 | 8.687 | 1.0535× |
| LinearProbe | Activation L2 | 32 | 10.157 | 9.530 | 1.0663× |
| LoRA | Response margin | 1 | 9.178 | 8.771 | 1.0400× |
| LoRA | Response margin | 32 | 10.013 | 9.682 | 1.0315× |
| LoRA | Activation L2 | 1 | 9.026 | 8.731 | 1.0365× |
| LoRA | Activation L2 | 32 | 9.975 | 9.631 | 1.0414× |
| PromptSteering | Response margin | 1 | 12.596 | 11.983 | 1.0612× |
| PromptSteering | Response margin | 32 | 13.843 | 13.183 | 1.0478× |
| PromptSteering | Activation L2 | 1 | 12.475 | 12.182 | 1.0253× |
| PromptSteering | Activation L2 | 32 | 13.890 | 13.745 | 1.0304× |

All **656 measured filter responses** match exactly; 530 pairs favor the new
implementation. All **224 measured score/gradient comparisons** are also exact.
Model-call ratios range from 0.9977× to 1.0032×, so no model-call speedup is
claimed. GPU peak counters match in every pair. Both prepared sessions coexist;
these counters do not measure isolated per-implementation process memory.

### Hopper and Blackwell

The same final worker and 137 targeted tests passed in H200 job `23053533`
and RTX PRO 6000 Blackwell job `23053534`, both on the preemptible queue.
Both compare DiffMean and PromptSteering, both objectives, k=1/32, 16 members
and physical batch 32. Each has **328 exact measured filter comparisons** and
**112 exact measured score/gradient comparisons**. H200 uses Python 3.12.1 /
Torch 2.5.1+cu124; Blackwell uses Python 3.12.1 / Torch 2.7.1+cu128.

Filtering ratios are **0.9825–1.0016× on the H200 host** and
**0.9882–1.0187× on the Blackwell host**. The H200 results include a regression
of up to about 1.8%; neither remote run establishes a filtering speedup.
Model-call ratios are 0.9957–1.0037× and 0.9986–1.0010× respectively, with
matching peak counters. The local CPU gain is not a hardware-independent claim.

### Correctness and full-search scope

The 137 focused unit tests cover normalization, candidate batching and worker
contracts. Eleven added cases check strict builtin-int identity, custom metaclass
equality/hashing traps, short-circuiting, iterator exceptions, and cleanup order
against the independent frozen normalizer. Forty selected normalization cases
execute all 16 executable lines in `_ids`, including the subclass generator.
Eight deliberate faults are detected, including the rejected subclass variant.
A deliberately coupled-reference control confirms that the independent oracle
is necessary to reject the generated-container drop-last fault.

Four paired complete DiffMean searches, including warmups, use 72 training
pairs, four sampled pairs, physical batch two, k=1, two steps, 16 proposals and
top-k 32. Every optimizer request, filter result, gradient sidecar and all
**936 raw score entries** match exactly. Search receipts and final sampled
generations match byte for byte. The judge is fake. Observed total-time ratios
are 0.9853× for response margin and 0.9984× for activation L2; **no whole-search
speedup is claimed**. An initial attempt ran out of disk space during preparation
and is excluded; the unchanged source and harness passed with `TMPDIR` in
shared memory. All four real-GPU integration tests passed (186.63 s), covering
all baseline methods, both objectives, k=1/32, independent model-math references,
IPC sidecars, bounded searches, and normal sampled generation.

The final worker SHA-256 is
`d7242ca10f8858a73d1b9e4f2818e12ef63b286168ac7573a3e4835606c4fea1`.
Its complete AST differs from the parent only in the imports and the guarded
validation expression. Ruff passes. Root-environment typing has the same four
diagnostics as the parent, with none added. The full `tests/` suite was not run.
Reports, tensors, profiles, mutation logs and frozen harnesses are retained under
`out/axbench-gcg-performance/loop23/`; `v1-general-map/` contains the rejected
earlier variant and is excluded from the final-source results above.

## BF16 MLP gating during candidate scoring

The merged-main RTX 4090 trace attributes about 25 ms of response-margin GPU
work to separate SiLU and gate-multiply kernels. The first fusion prototype
allocated an output tensor. Deterministic Torch execution filled that allocation,
adding one GPU fill for each fused call; at physical batch two this eliminated
the expected launch reduction and regressed the bounded search. That prototype
was rejected.

The implementation overwrites the fresh native gate projection instead. It uses
one elementwise Triton kernel, preserves the BF16 rounding between SiLU and
multiplication, and keeps the up projection unchanged. Explicit non-FTZ
`div.rn.f32` is required: the earlier FTZ version disagreed with eager SiLU for
508 gate bit patterns. Index arithmetic is widened before multiplication so
large tensors do not wrap at 2**31 elements. A real test crosses that boundary.

Fusion applies inside Transformers candidate scoring to supported native Llama,
Qwen2 and SmolLM3 MLPs with BF16 CUDA inputs on Ampere or newer. The scoped
dispatcher restores instance forwards in `finally`. Native Linear projections
and Parameters, known non-inplace SiLU activations, no hooks or forward overrides,
no Torch modes, no tracing, and no autocast are required. Missing Triton, a compiler
or Python headers keeps scoring on the eager path. Other compilation failures
remain visible.

Correctness qualification checks all 4,294,967,296 BF16 operand pairs. Non-NaN
outputs must match bit for bit, including signed zero; both paths must produce
NaN together, without requiring equal NaN payload bits. The focused tests cover
all three native MLP families, biases, input gradients, hooks, custom projections
and parameters, restoration after failures, modes, autocast, operand rejection,
owned storage, tails, and offsets beyond signed int32. Python executable-line
coverage is 91/91 for the seven helper/wrapper functions; device arithmetic is
qualified by actual CUDA tests. Two mutation controls passed and all 16 intentional
faults were detected.

All four final-source real-model integration tests passed on the RTX 4090
(217.68 s). They cover DiffMean, LinearProbe, LoRA and PromptSteering, both
objectives, suffix lengths 1 and 32, gradient/score reference comparisons, IPC,
bounded optimizer searches and ordinary sampled generation with a fake judge.
The optimization/reference comparisons use exact equality at each fixed batch
shape. Cross-batch comparisons retain the existing BF16 tolerances; changing
GEMM shapes can change eager numerical results. A candidate-only corruption
of `+0.125` in the final kernel was rejected specifically by the independent
`checked_score` equality assertion. This confirms that the reference does not
accidentally share the candidate fusion.

The source comparison freezes the complete worker and model-state helper from
main `5b99abc88a7309ed0495284c91cc02b1a602f953`, in independent module globals.
Those GCG files are byte-identical to the merged 17-PR stack and the rebased
main `afa1f4f0a8d99283d75a5e3e07651434092fe0f3`. Both sessions share one unchanged
model; exact prepared payloads and clean hidden states are checked before
alternating timed calls. Tensor snapshots, comparisons and dispatch counting
are outside measured intervals. Candidate scoring must actually invoke the new
kernel; the reference and gradients must not. Peak memory figures from two
co-resident sessions are not isolated memory claims.

### Blackwell qualification

RTX PRO 6000 Blackwell job `23061811` completed with the final 64-bit-indexed
kernel. It uses SmolLM3-3B commit `cfb32d505f5025ec9be4e704f70cfbf5bdf8da94`,
72 prepared training pairs, 16 sampled members, physical batch 32, two candidates,
and suffix lengths 1 and 32. Each case has three warmup pairs and seven measured
pairs. Python is 3.12.11, Torch 2.7.1+cu128, Triton 3.3.1, Transformers 5.16.1,
PyVene 0.1.8 and PEFT 0.13.2. OMP and MKL threads are set to eight. Deterministic
Torch execution and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are enabled. Loading,
preparation, compilation and numerical comparisons are excluded from these warm
component timings.

Durations are medians in milliseconds. Speedup is the median of paired
reference/new ratios, so it need not equal the ratio of the displayed medians.

| Method | Objective | Suffix tokens | Reference score (ms) | New score (ms) | Paired speedup |
|---|---|---:|---:|---:|---:|
| DiffMean | Response margin | 1 | 792.023 | 762.505 | 1.0388× |
| DiffMean | Response margin | 32 | 865.150 | 837.937 | 1.0334× |
| DiffMean | Activation L2 | 1 | 625.573 | 605.285 | 1.0328× |
| DiffMean | Activation L2 | 32 | 682.902 | 657.880 | 1.0402× |
| PromptSteering | Response margin | 1 | 969.888 | 935.241 | 1.0371× |
| PromptSteering | Response margin | 32 | 1053.744 | 1016.749 | 1.0364× |
| PromptSteering | Activation L2 | 1 | 768.848 | 742.106 | 1.0333× |
| PromptSteering | Activation L2 | 32 | 833.718 | 804.555 | 1.0329× |

All 56 measured score pairs favor the new implementation; all 112 measured
score/gradient comparisons match exactly. Per-case median gradient ratios range
from 0.9976× to 1.0036×; no gradient speedup is claimed. All 93 targeted tests passed
on this GPU, including every BF16 operand pair and the large-index test. The same
93 targeted tests also passed in the local vendor environment on the RTX 4090.
The root environment passed the 66 CPU cases, with 27 GPU cases deselected.

The original remote Python installation has no development headers. Its eager
fallback passed 66 tests with 27 CUDA-fusion tests skipped, and a real-model probe
verified zero fused calls and exact scores/gradients for both objectives. Fusion
was qualified using an isolated Python installation with headers and the same
original Torch, Transformers and Triton packages. The shared environment was
not modified. Fallback timing from the single-pair probe is not a speed claim.

The final kernel SHA-256 is
`113bde18adf0c882ba4c281bbbde8e17e2218591d396eca233241071dc26255b`.
Blackwell reports, numerical tensors, source hashes and logs are retained under
`out/axbench-gcg-performance/loop24/owned-v4/blackwell/`. The collected archive
SHA-256 is `2d244db73b22b72d2e95c4cabc6e3a6414939cddfe9183ad99804b7a516a25e4`.

### Hopper qualification

H200 job `23061810` completed with the same final source, model, inputs, case
matrix and paired protocol. It uses Python 3.12.11, Torch 2.5.1+cu124 and Triton
3.1.0, with OMP and MKL threads set to four. All 93 targeted tests passed
(14.71 s), including every BF16 operand pair and the large-index test. All 112
measured score/gradient comparisons match exactly. Separate dispatch probes
verify that candidate scores use fusion while reference calls and gradients
remain eager. The original headerless Python environment passed 66 tests with
27 skips and the two-objective real-model fallback probe with zero fused calls.

No reliable Hopper speedup is claimed from this run. Individual paired ratios
range from 0.9336× to 1.2356× even for unchanged gradient code; score ratios range
from 0.7684× to 1.6260×. Although 43 of 56 score pairs favor the candidate, the
large timing variation prevents attributing these swings to the optimization.
All measurements are retained, including slower candidate pairs, without
assigning an unverified cause to the noise.

The reports, numerical tensors, dispatch evidence, source hashes, logs and
validated summary are retained under
`out/axbench-gcg-performance/loop24/owned-v4/hopper/`. The collected archive
SHA-256 is `44ba12f1da1557c69f6d498c5b8e52191090988308bcfc5982224b1a02ac7964`.

### RTX 4090 qualification

The component matrix uses four sampled members, physical batch four and three
warmup pairs per case. DiffMean has 21 measured pairs per case; LinearProbe,
LoRA and PromptSteering each have seven. It uses Python 3.12.11, Torch
2.5.1+cu124, Triton 3.1.0 and the same Transformers, PyVene and PEFT versions as
the remote qualification. OMP and MKL thread overrides are unset. The model,
training population, frozen reference, deterministic settings and numerical
checks follow the protocol above.

Durations are medians in milliseconds; speedup is the median paired ratio.

| Method | Objective | Suffix tokens | Reference score (ms) | New score (ms) | Paired speedup |
|---|---|---:|---:|---:|---:|
| DiffMean | Response margin | 1 | 373.988 | 365.488 | 1.0234× |
| DiffMean | Response margin | 32 | 400.268 | 391.186 | 1.0231× |
| DiffMean | Activation L2 | 1 | 283.561 | 276.704 | 1.0243× |
| DiffMean | Activation L2 | 32 | 303.141 | 296.132 | 1.0240× |
| LinearProbe | Response margin | 1 | 373.764 | 364.468 | 1.0243× |
| LinearProbe | Response margin | 32 | 399.949 | 390.209 | 1.0260× |
| LinearProbe | Activation L2 | 1 | 283.144 | 277.396 | 1.0202× |
| LinearProbe | Activation L2 | 32 | 303.780 | 296.368 | 1.0257× |
| LoRA | Response margin | 1 | 376.575 | 366.772 | 1.0266× |
| LoRA | Response margin | 32 | 400.082 | 391.653 | 1.0214× |
| LoRA | Activation L2 | 1 | 284.308 | 277.777 | 1.0240× |
| LoRA | Activation L2 | 32 | 304.023 | 296.163 | 1.0267× |
| PromptSteering | Response margin | 1 | 453.839 | 442.795 | 1.0245× |
| PromptSteering | Response margin | 32 | 483.298 | 471.341 | 1.0254× |
| PromptSteering | Activation L2 | 1 | 345.349 | 336.252 | 1.0264× |
| PromptSteering | Activation L2 | 32 | 366.185 | 356.359 | 1.0280× |

All 336 measured score/gradient comparisons match exactly. The candidate is
faster in 167 of 168 measured score pairs. Per-case median paired scoring-time
reductions are 1.98%–2.73%; median gradient ratios are 0.9979×–1.0009×, with no
gradient speedup claimed. The slower LoRA score pair remains in the report and
summary.

Each method runs in a separate, independently prepared subprocess. The DiffMean
subprocess returned successfully, with its final reference evidence written
29.8 seconds before a foreign pytest process interrupted the subsequent
LinearProbe run. The one-second resource audit observed no foreign GPU or
pytest processes throughout the completed DiffMean block. Its complete results
are retained; the incomplete LinearProbe block was excluded and restarted with
the full warmup protocol. LinearProbe, LoRA and PromptSteering then completed
with no observed foreign GPU or pytest process during that separate interval.

### Default execution settings

A separate DiffMean probe disables deterministic algorithms and unsets
`CUBLAS_WORKSPACE_CONFIG`, with four members, physical batch four, one suffix
token, three warmup pairs and seven measured pairs per objective. Its initial
exact gradient comparison failed. An independent control then routes every
operation through the complete frozen reference worker and raises if the new
kernel is called. All eight repeat-versus-first gradient comparisons differ,
across both objectives; the largest absolute reference-self difference is
`0.000396728515625`. Reference-only scores remain exact in all 14 measured
pairs. Raw gradients and the original failure are retained.

Exact gradient qualification therefore uses the deterministic matrix above.
The normal-settings scoring probe retains `rtol=0, atol=0`, verifies zero fused
gradient calls, and makes no gradient parity or timing claim. All 14 measured
candidate/reference score comparisons and both separate dispatch probes match
exactly; every measured score pair favors the candidate.

| Objective | Reference score (ms) | New score (ms) | Median paired speedup |
|---|---:|---:|---:|
| Response margin | 365.054 | 355.616 | 1.0267× |
| Activation L2 | 276.821 | 270.863 | 1.0221× |

### Complete searches and physical batch two

The paired complete-search check uses the common configuration in the
[stack progress table](axbench-adv-gcg-stack.md): 72 prepared pairs, four
sampled pairs, physical batch two, one suffix token, two steps, 16 proposals
and top-k 32. Each objective has one warmup pair and three measured pairs.
All eight reference/candidate search pairs match exactly in requests, filters,
gradients, scores, persisted search receipts and normal sampled generation
with a fake judge. The trace comparison includes 1,872 exact score entries
across measured and warmup searches.

These are bounded-cell measurements, including preparation and generation,
with model loading excluded. They do not establish campaign throughput.
Unchanged preparation and filtering also vary between arms, so the complete
phase means and all pairs are retained in the progress-table evidence.

| Objective | Reference cell median (s) | New cell median (s) | Median paired ratio |
|---|---:|---:|---:|
| Response margin | 17.111 | 17.009 | 1.0063× |
| Activation L2 | 15.706 | 15.620 | 1.0055× |

A separate physical-batch-two component probe has seven measured pairs after
three warmup pairs per objective. All 28 measured gradient/score comparisons
match exactly; all 14 score pairs favor the candidate. Score medians are
379.436 → 377.407 ms for response margin and 290.715 → 288.904 ms for L2, with
median paired ratios of 1.0079× and 1.0070×. These smaller gains are specific
to physical batch two; the batch-four and batch-32 results above are separate
measurements.

Profiler captures run after the paired timings. Response-margin scoring
replaces 288 SiLU/multiply pairs with 288 fused calls, and L2 replaces 232
pairs with 232 fused calls. Total CUDA kernel counts fall by exactly 288 and
232, respectively. Both the shape-indexed `aten::fill_` counts and GPU fill
kernel counts are unchanged. Profiled SiLU-plus-multiply device time changes
from 11.866 to 8.248 ms for response margin and 9.525 to 6.694 ms for L2.
These captures explain the launch reduction; the paired, unprofiled calls
provide the performance measurements.

The fresh-cache kernel probe uses shape `(4, 459, 11008)`. Its first call takes
0.5503 s, including Triton compilation and runtime-driver initialization; the
ten warm calls have a 0.0975 ms median. All 11 calls match eager arithmetic
exactly. CUDA initialization, imports, operand allocation/copies and comparisons
are outside these intervals, so this is not a complete GCG cold-start estimate.

Local reports, numerical tensors, compressed raw traces, resource audits,
interruption records and the frozen harnesses are retained under
`out/axbench-gcg-performance/loop24/owned-v4/rtx4090/`. The collection manifest
records original and retained hashes, including decompression verification for
the traces. Interrupted or incomplete methods are excluded from accepted results.
The collection manifest SHA-256 is
`9394cc3c549f4ca1fccf702a291501d9f96a5548716e4408234c341e9328f379`.
Ruff and formatting pass for all five changed Python files; the scoped type
check passes for the two new helpers and their unit tests. The complete
`tests/` suite was not run.

## Rejected project compiler-budget experiment

A separate SelfConditioning response-margin control reproduced FlexAttention's
recompile-limit fallback without a profiler. Raising the compiler budget from
8 to 32 retained exact gradients but changed all eight score entries: maximum
absolute error was 0.0132551 and RMSE was 0.0074335, failing the diagnostic
`rtol=atol=1e-5` comparison. The candidate was rejected. Separate-process score
medians of 87.355 and 46.982 ms are diagnostic observations, not an accepted
paired speedup. Reports, tensors, logs and harnesses are retained under
`out/axbench-gcg-performance/loop25/`; its collection manifest SHA-256 is
`43eca17d0831349623e1b8974a3a0d9634f11889d387a54c1528b5047601b47b`.

## BF16 RMSNorm postscaling during candidate scoring

The next change targets the separate normalization multiply, FP32-to-BF16
conversion and learned-weight multiply in native RMSNorm. One Triton kernel
replaces those three operations. PyTorch still performs the original FP32
conversion, square, mean, epsilon addition and reciprocal square root. The
kernel preserves the BF16 rounding before the weight multiplication, uses
non-FTZ `mul.rn.f32` for both products, and widens indices before multiplying
by the block size. Inputs and weights retain their values and storage.

It extends the existing scoring context for native Llama, Qwen2 and SmolLM3
norms, sharing its hook, mode, tracing and build-prerequisite checks.
Contiguous BF16 CUDA inputs and weights with matching device and native shape
are required. Custom Parameters, forward overrides and epsilon values other
than Python integers or floats retain the native forward. Autograd, CUDA
autocast and unsupported hardware also retain that path. The context restores
forwards on exit, including exceptions and nested scopes. No additional
configuration or persistent model cache is introduced. A prototype that reused
temporary reduction storage had mixed timings; this implementation uses an
ordinary output allocation.

Review caught a wrapper-interface bug before publication: all three native
classes accept `hidden_states=`, whereas the first wrapper named its argument
`x`. Three new regression cases failed before the parameter rename. They now
pass, and the CUDA parity cases also invoke the native keyword. The rename
leaves bytecode instructions and the arithmetic kernel unchanged, but all
paired model measurements below were repeated with the corrected source. Earlier v1 evidence
and the failed regression remain retained separately.

The complete worker, state helper, MLP dispatcher and SiLU kernel from
[#1712](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1712)
commit `1b850ea4d73cc791210601ea1dbf1af77ef1b475` form the frozen reference.
Their module globals are independent, and MLP fusion remains enabled in both
arms. Prepared payloads and clean hidden states must match exactly. Separate
probes require positive RMS dispatch counts for candidate scoring and zero
for the reference and gradients. Both sessions share an unchanged loaded model;
reported peak allocations therefore do not establish isolated memory savings.
Candidate source hashes identify the tested overlay independently of the
parent Git SHA reported by the benchmark's staging checkout.

### RTX 4090 measurements

SmolLM3-3B commit `cfb32d505f5025ec9be4e704f70cfbf5bdf8da94`, 72 prepared
pairs, four selected pairs, physical batch four and two scoring candidates are
held fixed. Each case has three warmup pairs, then 21 measured pairs for DiffMean
and seven for each other method. The environment is Python 3.12.11, Torch
2.5.1+cu124, Triton 3.1.0, Transformers 5.16.1, PyVene 0.1.8 and PEFT 0.13.2.
Deterministic algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are enabled;
OMP and MKL overrides are unset. Ratios are medians of paired reference/new
measurements and need not equal ratios of the displayed duration medians.

| Method | Objective | Suffix tokens | Reference score (ms) | New score (ms) | Median paired ratio |
|---|---|---:|---:|---:|---:|
| DiffMean | Response margin | 1 | 364.780 | 359.506 | 1.0153× |
| DiffMean | Response margin | 32 | 390.774 | 386.295 | 1.0119× |
| DiffMean | Activation L2 | 1 | 276.591 | 273.270 | 1.0126× |
| DiffMean | Activation L2 | 32 | 295.648 | 292.035 | 1.0125× |
| LinearProbe | Response margin | 1 | 366.332 | 360.631 | 1.0158× |
| LinearProbe | Response margin | 32 | 391.366 | 386.349 | 1.0133× |
| LinearProbe | Activation L2 | 1 | 276.748 | 273.246 | 1.0129× |
| LinearProbe | Activation L2 | 32 | 295.955 | 292.091 | 1.0124× |
| LoRA | Response margin | 1 | 366.349 | 360.710 | 1.0158× |
| LoRA | Response margin | 32 | 391.589 | 387.320 | 1.0118× |
| LoRA | Activation L2 | 1 | 278.120 | 274.336 | 1.0138× |
| LoRA | Activation L2 | 32 | 296.582 | 293.021 | 1.0119× |
| PromptSteering | Response margin | 1 | 444.189 | 439.155 | 1.0121× |
| PromptSteering | Response margin | 32 | 471.412 | 465.677 | 1.0122× |
| PromptSteering | Activation L2 | 1 | 337.377 | 333.724 | 1.0106× |
| PromptSteering | Activation L2 | 32 | 357.228 | 353.414 | 1.0109× |

All 336 measured score/gradient comparisons are exact. The candidate is faster
in 166 of 168 score pairs, with median paired scoring-time reductions of
1.05–1.56%. Per-case median gradient ratios span
0.9993–1.0007×; no gradient speedup is claimed.
Individual score ratios span 0.5798–1.5721×;
outliers are retained. One-second resource audits observed no foreign CUDA-compute or
pytest processes throughout the accepted local runs.

With normal backend settings, all 14 score comparisons remain exact and
13 pairs favor the candidate. A fresh frozen-reference-only gradient
control differs in 4 of four repeat-versus-first comparisons, with maximum
absolute difference 0.00025177002. Exact gradient qualification uses the
deterministic matrix; no gradient tolerance is relaxed. The normal-settings
probe also verifies zero RMS fusion calls during gradients.

| Method | Objective | Suffix tokens | Reference score (ms) | New score (ms) | Median paired ratio |
|---|---|---:|---:|---:|---:|
| DiffMean | Response margin | 1 | 357.399 | 353.847 | 1.0093× |
| DiffMean | Activation L2 | 1 | 270.090 | 268.611 | 1.0070× |

### Complete searches, smaller batches and launch attribution

The complete-search configuration and phase means are recorded in the
[stack progress table](axbench-adv-gcg-stack.md), which adds this change as a
new column. Each objective has one warmup pair and three measured pairs.
All eight search pairs have exact optimizer requests, filtering, gradients,
1,872 score entries, persisted search receipts and sampled generations with a
fake judge. These bounded-cell observations include preparation and generation
but exclude model loading; they do not establish campaign throughput.

| Objective | Reference cell median (s) | New cell median (s) | Median paired ratio |
|---|---:|---:|---:|
| Response margin | 17.008 | 16.837 | 1.0101× |
| Activation L2 | 15.612 | 15.497 | 1.0075× |

A separate physical-batch-two check retains exact equality in all 28 measured
gradient/score comparisons; 14 of its 14 score pairs favor the candidate.

| Method | Objective | Suffix tokens | Reference score (ms) | New score (ms) | Median paired ratio |
|---|---|---:|---:|---:|---:|
| DiffMean | Response margin | 1 | 377.319 | 371.387 | 1.0157× |
| DiffMean | Activation L2 | 1 | 289.904 | 285.366 | 1.0156× |

Separate profiler captures replace 584 normalization/cast/weight-product
sequences with 584 RMS kernels for response margin, and 472 with 472 for L2.
Total CUDA kernel counts fall by exactly 1,168 and 944, respectively. Native
square, mean and reciprocal-square-root counts remain unchanged, as do GPU
fill-kernel and `aten::fill_` counts. The traces support attribution; performance
claims use the unprofiled paired calls.

For shape `(4, 459, 2048)`, a fresh-cache RMS wrapper call takes 0.4066 s,
including Triton compilation and driver initialization. Ten warm calls have
a 0.05185 ms median, and all 11 outputs match exactly. This probe and the PTX
index audit predate the parameter rename and use the identical kernel hash.
Imports, initial CUDA setup, operand allocation and comparisons are excluded;
this is not a full GCG startup measurement.

### RTX PRO 6000 Blackwell qualification

Job `23087425` completed successfully on `node5202`. It uses Python
3.12.11, Torch 2.7.1+cu128 and Triton 3.3.1, 16 selected pairs, physical batch
32, two candidates, three warmup pairs and seven measured pairs per case.
The model, prepared population and deterministic protocol match the local
qualification; OMP and MKL threads are set to 8.

| Method | Objective | Suffix tokens | Reference score (ms) | New score (ms) | Median paired ratio |
|---|---|---:|---:|---:|---:|
| DiffMean | Response margin | 1 | 759.188 | 726.298 | 1.0469× |
| DiffMean | Response margin | 32 | 833.344 | 798.571 | 1.0452× |
| DiffMean | Activation L2 | 1 | 595.898 | 569.822 | 1.0458× |
| DiffMean | Activation L2 | 32 | 651.845 | 623.954 | 1.0476× |
| PromptSteering | Response margin | 1 | 930.866 | 889.846 | 1.0476× |
| PromptSteering | Response margin | 32 | 1011.420 | 965.616 | 1.0482× |
| PromptSteering | Activation L2 | 1 | 726.681 | 693.047 | 1.0482× |
| PromptSteering | Activation L2 | 32 | 788.938 | 753.988 | 1.0459× |

All 112 measured gradient/score comparisons are exact; 56
of 56 score pairs favor the candidate. Median paired scoring-time reductions
are 4.32–4.60%.
Gradient median ratios span 0.9908–1.0037×.
Individual gradient ratios span 0.8550–1.2319×
and score ratios span 1.0335–1.5599×. These raw
observations remain in the evidence; no gradient speedup is claimed.

All 157 targeted tests pass, including the native keyword interface, every
BF16 weight-product pair and real offsets beyond 2**31 elements. A separate
real-model probe with an unavailable C compiler retains exact scores and
gradients for both objectives and verifies zero fused calls. Its timings
are not used to claim a fusion speedup. The collected archive SHA-256 is
`2725b8db031cb1932fec60ce0e5f0618bc3a2a171808f218601be6d7d161e52b`.

### Hopper unit qualification; full worker run pending

The corrected-source H200 qualification, job `23087424`, timed out on node3300
after 25 minutes 24 seconds. Source checks passed, but the first Python probe
stalled before producing a numerical result; a process snapshot observed it
in uninterruptible wait. No result from this attempt is accepted. A bounded
retry, job `23122550`, retained the same archive and numerical protocol,
excluded node3300, and staged the source on verified local storage. It also
timed out, on node3401-c1 after 25 minutes 10 seconds. Source checks passed,
but its first Python probe produced no numerical or timing report. A process
snapshot observed page-I/O wait with a shared-storage pandas dependency open;
the installed dependencies had not been staged locally. The retained terminal
evidence is under `out/axbench-gcg-performance/loop26/hopper-local-timeout-23122550/`,
collection SHA-256
`6061345545c0a210603eb8815fbb331942f3b407a13203d8b99f18c1c9ba4076`.
CPU-only dependency-staging job `23148249` subsequently timed out after
30 minutes 20 seconds on node1618. Inventory completed, but its last periodic
archive record covered only 8,001 of 38,785 files. It produced no complete
dependency archive, verified extraction, or CPU worker-import result. Slurm
denied the requested time-limit extension. The terminal evidence is retained
under `out/axbench-gcg-performance/loop26/hopper-dependency-timeout-23148249/`,
collection SHA-256
`a5a7043246ef4e959d7bae0205151c30a8f521af0fa41e705a8a259c4a869710`.
No further full-worker GPU retry has been submitted for #1713. Full worker
qualification remains outstanding; this CPU timeout is infrastructure evidence.

CPU-only retry `23165617` failed on node3113 after 2 hours 4 minutes 10 seconds.
It used the same dependencies and frozen source archive with a three-hour
allocation. The job reported successful local verification of all 38,785
package files (6,217,579,898 source bytes), then failed during the first Torch
import. The staged package directory was on `PYTHONPATH`, where its `asyncio`
package shadowed Python's standard library. The traceback reaches
`packages/asyncio/base_events.py:296`, whose `tasks.async(...)` expression is
invalid syntax in Python 3.12. This is a verified import-resolution failure;
the worker and package-identity assertions were not reached. No validated
dependency archive was published, no GPU was requested, and no further retry
was submitted before pausing. Full-worker Hopper qualification for #1713
remains outstanding. Terminal scripts, scheduler state and import logs are
retained under `loop26/hopper-dependency-failed-23165617/`, collection SHA-256
`fb7d96476066ba5919b76901ddc60edc20789fe490a0ce23ac2e3de42eaadb9e`.

The separate correctness-only job `23089351` completed on H200 node3200 in
4 minutes 56 seconds. All 64 RMSNorm tests passed, including the exhaustive
BF16 weight-product grid and real wide-offset indexing. Its verified sources,
test log, hardware identity and scheduler status are retained in
`out/axbench-gcg-performance/loop26/hopper-unit/`; the collection manifest
SHA-256 is `3e73f108f01e8a9eefeac27d1a7c726e8964ac352cd632dbf6acb991c24dd3c3`.
This qualifies the postscale kernel's unit contracts. It does not qualify the
later square-fusion change, clean-reference preparation, or full worker
behavior, and provides no Hopper performance measurement.

### Correctness scope and retained evidence

The RMS tests cover all three native norm families, positional and keyword
calls, forward/backward fallback, custom epsilon precision, hooks and overrides,
tensor/Parameter subclasses, strides, dtype/device/shape rejection, empty and
scalar inputs, aliases, inference mode, extreme and nonfinite values, signed
zeros, and restoration after body or entry failures. The low-level weight-product
check enumerates all 4,294,967,296 BF16 operand pairs. Non-NaN outputs match bit
for bit; NaN classification must match without requiring identical NaN payloads.
PTX and Triton IR retain two rounding steps, non-FTZ multiplies and wide indexing.

The RTX 4090 vendor subset passes 156 tests with the large RMS index test
skipped by its free-memory guard; that test runs on Blackwell. The root
environment passes all 85 CPU cases, with 72 GPU cases deselected. All 147
executable Python body lines across ten dispatch/wrapper functions are covered;
device arithmetic is qualified by CUDA execution instead of Python tracing.
Two mutation controls pass and all 14 deliberate RMS faults are detected.

All four real-model integration tests pass, covering the four baselines, both
objectives, suffix lengths 1/32, IPC, full reference comparisons, bounded
searches and ordinary sampled generation. Fixed-shape reference checks remain
exact; existing cross-batch BF16 tolerances are retained. A candidate-only
`+0.125` output corruption fails specifically at the independent `checked_score`
assertion, proving that the reference does not share RMS fusion.

The corrected dispatcher SHA-256 is
`b8bb7942e7c9409ca2232a6d9d5e1f030fffb86e8d88b148da69db02f659b9db`;
the RMS kernel SHA-256 is
`bac69776c58a97cbeda18193edb716b7c2a863a34d4c4b850c45cb4a25dc0dc7`.
Reports, tensors, compressed traces, audits, frozen references, mutations and
source snapshots are retained under
`out/axbench-gcg-performance/loop26/final-v2/`. The local collection contains
210 files and 61,589,205 retained bytes, with hashes and decompression
verification; its manifest SHA-256 is
`242ca7fad1981995ff5d21b82d23a7bcfc768d1096f53e9fd2aa002ede55b7a0`.

The prior local/Blackwell qualification is retained under `loop26/rtx4090/`
and `loop26/blackwell/`. The prior Hopper attempt `23086239` was retired after
remaining in an older environment's startup without test output; its logs and
ownership-checked cancellation are retained. It supplies no RMS correctness
or performance result. Ruff, formatting and scoped type checks pass. The full
`tests/` suite was not run. The combined selected-factor qualification after
PR #1716 is recorded above; this optimization leaves objective construction
untouched.

## BF16 RMSNorm conversion and square during candidate scoring

The preceding RMS postscale profile still launches a separate BF16-to-FP32
copy and FP32 square for each native norm. This change combines those two
operations in one pointwise Triton kernel. PyTorch still performs the original
FP32 mean, epsilon addition and reciprocal square root, and the existing
postscale kernel still rounds to BF16 before multiplying by the learned weight.
The postscale kernel now reads the unchanged BF16 input directly. Private FP32
squares are released after the queued mean; no caller-owned storage is reused.
The existing scoring eligibility, native fallback and restoration are unchanged.
There are no new configuration keys, worker operations or persistent caches.

The square uses explicit non-FTZ `mul.rn.f32`, with a 64-bit element offset
before the block multiplication. Exhaustive BF16 input checks establish exact
FP32 squares, including subnormal results. The retained generated IR accepts
large counts directly as i64 and preserves wide pointer arithmetic. An initial
audit-script assertion incorrectly expected an unsigned-32 count conversion;
inspection corrected the audit metadata, with no production-code change.

An alternative using native `float().pow_(2)` remained exact but improved only
18 of 28 score pairs, with per-case median ratios of 0.9999–1.0036×; it was
rejected. Preliminary component-only timings excluded some parent guards and
are retained as screening, not performance claims. A separate repeated
projection-window probe was closed after checking the earlier rejection and
reproducing a score mismatch; it contributes no production change.

The complete worker, runtime-state helper, MLP dispatcher, SiLU and RMS modules
from [#1713](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1713)
commit `08cb0c7a664dfbb5a0dcd2b8c5b709ed5947941e` form the frozen reference. Both arms keep parent MLP and RMS
postscale fusion. Separate dispatch probes require positive RMS counts during
scoring, zero during gradients, and exact prepared payloads and clean states.
The two implementations have separate module globals; the independent
real-model oracle rejects candidate-only `+0.125` score corruption at its
`checked_score` assertion. Shared model/session allocations are not an isolated
memory-savings benchmark.

### RTX 4090 paired measurements

SmolLM3-3B commit `cfb32d505f5025ec9be4e704f70cfbf5bdf8da94`, 72 prepared
pairs, four selected pairs, physical batch four and two candidates are fixed.
DiffMean has three warmup pairs and 21 measured pairs per case; the other three
methods use three warmups and seven measured pairs. Python 3.12.11, Torch
2.5.1+cu124, Triton 3.1.0 and Transformers 5.16.1 are unchanged. Deterministic
algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are enabled; OMP/MKL overrides
are unset. Ratios are medians of paired reference/new calls, not ratios of
the separately displayed duration medians.

| Method | Objective | Suffix tokens | Reference score (ms) | New score (ms) | Median paired ratio |
|---|---|---:|---:|---:|---:|
| DiffMean | Response margin | 1 | 359.448 | 355.767 | 1.0100× |
| DiffMean | Response margin | 32 | 385.723 | 381.858 | 1.0099× |
| DiffMean | Activation L2 | 1 | 272.795 | 269.928 | 1.0111× |
| DiffMean | Activation L2 | 32 | 291.574 | 288.725 | 1.0096× |
| LinearProbe | Response margin | 1 | 360.083 | 356.559 | 1.0103× |
| LinearProbe | Response margin | 32 | 385.895 | 381.737 | 1.0098× |
| LinearProbe | Activation L2 | 1 | 272.365 | 269.742 | 1.0106× |
| LinearProbe | Activation L2 | 32 | 291.395 | 288.917 | 1.0080× |
| LoRA | Response margin | 1 | 360.105 | 356.299 | 1.0112× |
| LoRA | Response margin | 32 | 386.950 | 382.349 | 1.0113× |
| LoRA | Activation L2 | 1 | 273.441 | 270.225 | 1.0117× |
| LoRA | Activation L2 | 32 | 292.054 | 289.348 | 1.0091× |
| PromptSteering | Response margin | 1 | 438.980 | 433.482 | 1.0134× |
| PromptSteering | Response margin | 32 | 464.618 | 460.414 | 1.0087× |
| PromptSteering | Activation L2 | 1 | 332.206 | 327.907 | 1.0135× |
| PromptSteering | Activation L2 | 32 | 352.719 | 349.424 | 1.0110× |

All 336 measured score/gradient comparisons are exact. The candidate is faster
in 167 of 168 score pairs; median paired scoring-time reductions span
0.79–1.33%. Gradient median ratios span
0.9977–1.0008×. Raw score ratios span
0.6022–1.0171× and raw gradient ratios span
0.9889–1.0074×; the slower outlier remains retained.
No gradient speedup is claimed. One-second resource audits observed no foreign
CUDA-compute or pytest processes during the accepted local runs.

Normal backend settings retain 14 exact score comparisons, all favoring the
candidate. A frozen-reference-only gradient control differs in all four
repeat-versus-first comparisons, with maximum absolute difference 0.0003051758.
Exact gradient qualification therefore uses the deterministic matrix; no
gradient tolerance is relaxed and gradient RMS fusion remains disabled.

| Method | Objective | Suffix tokens | Reference score (ms) | New score (ms) | Median paired ratio |
|---|---|---:|---:|---:|---:|
| DiffMean | Response margin | 1 | 350.545 | 347.190 | 1.0098× |
| DiffMean | Activation L2 | 1 | 266.022 | 261.942 | 1.0152× |

### Complete searches and profile attribution

All eight bounded whole-search pairs (including two warmups) retain exact
optimizer requests, filtering, gradients, 1,872 raw scores, persisted search
receipts and sampled generations. The [stack progress table](axbench-adv-gcg-stack.md)
adds a column with phase means for three measured runs per objective, using
the same receipt-verified configuration as prior columns. Model loading and
paid judging are excluded; these observations do not establish campaign
throughput.

| Objective | Reference cell median (s) | New cell median (s) | Median paired ratio |
|---|---:|---:|---:|
| Response margin | 16.775 | 16.676 | 1.0057× |
| Activation L2 | 15.462 | 15.372 | 1.0064× |

Physical batch two retains exact equality in all 28 measured score/gradient
comparisons; all 14 score pairs favor the candidate.

| Method | Objective | Suffix tokens | Reference score (ms) | New score (ms) | Median paired ratio |
|---|---|---:|---:|---:|---:|
| DiffMean | Response margin | 1 | 370.345 | 366.332 | 1.0103× |
| DiffMean | Activation L2 | 1 | 283.712 | 280.955 | 1.0105× |

Separate profiler captures show 584 copy/square sequences replaced with 584
square kernels for response margin, and 472 with 472 for L2. Total CUDA kernel
counts fall from 15,968 to 15,384 and from 13,000 to 12,528 respectively.
Native mean, reciprocal-square-root and fill counts remain unchanged. Profiles
explain launch removal; performance claims use unprofiled paired calls.

A fresh Triton cache at shape `(4, 459, 2048)` gives a first wrapper call of
0.5074 s including compiler/driver initialization, with a
0.04440 ms median over ten warm calls. All 11 outputs are exact.
Imports, initial CUDA setup, allocation and comparisons are outside that timer;
it is not a whole-GCG cold-start measurement.

### RTX PRO 6000 Blackwell qualification

Job `23090341` completed on `node5202` with Python
3.12.11, Torch 2.7.1+cu128, Triton 3.3.1 and 8 OMP/MKL threads. It uses
the same model and deterministic protocol, 16 selected pairs, physical batch
32, two candidates, three warmup pairs and seven measured pairs per case.

| Method | Objective | Suffix tokens | Reference score (ms) | New score (ms) | Median paired ratio |
|---|---|---:|---:|---:|---:|
| DiffMean | Response margin | 1 | 725.343 | 691.934 | 1.0467× |
| DiffMean | Response margin | 32 | 797.939 | 761.283 | 1.0484× |
| DiffMean | Activation L2 | 1 | 566.242 | 540.062 | 1.0476× |
| DiffMean | Activation L2 | 32 | 621.224 | 592.348 | 1.0457× |
| PromptSteering | Response margin | 1 | 885.788 | 849.078 | 1.0432× |
| PromptSteering | Response margin | 32 | 963.161 | 923.942 | 1.0428× |
| PromptSteering | Activation L2 | 1 | 691.334 | 660.609 | 1.0477× |
| PromptSteering | Activation L2 | 32 | 751.114 | 719.998 | 1.0432× |

All 112 measured score/gradient comparisons are exact; 56 of 56
score pairs favor the candidate. Median paired scoring-time reductions span
4.10–4.62%.
Gradient median ratios span 0.9742–1.0301×;
raw gradient ratios span 0.7632–1.3005× and raw
score ratios span 1.0197–1.0572×. All observations
are retained; gradient timing controls are noisy and no gradient speedup is
claimed.

All 161 targeted MLP, RMS and worker tests pass. A missing-C-compiler
real-model check retains four exact score/gradient comparisons and verifies
zero fused calls in both arms. The collected archive SHA-256 is
`5674d2ccb79e62d53474c80cdab8739059bcedf2f21f4d41afe8e24e6ca9d59e`.

The first Blackwell attempt, `23089698` on `node4004`, waited in startup I/O
for thirteen minutes before any test or model output. Its diagnostics and
ownership-checked cancellation are retained in `loop28/final/review/`. The
successful retry excluded that node and used the identical source archive.
The cancelled attempt supplies no correctness or performance result.


### H200 Hopper qualification

Job `23089697` completed on `node3200` in 9 minutes 50 seconds using
Python 3.12.11, Torch 2.5.1+cu124, four OMP/MKL threads and the immutable
source archive with SHA-256
`b1f7ce7054fae0c38c2806ea7ba8f366d580969823d1afdd048d8d9c35d03e38`.
The protocol uses 16 selected pairs, physical batch 32, two candidates,
three warmup pairs and seven measured pairs per case.

| Method | Objective | Suffix tokens | Reference score (ms) | New score (ms) | Median paired ratio |
|---|---|---:|---:|---:|---:|
| DiffMean | Response margin | 1 | 464.495 | 449.734 | 1.0331× |
| DiffMean | Response margin | 32 | 495.362 | 479.988 | 1.0350× |
| DiffMean | Activation L2 | 1 | 362.590 | 348.942 | 1.0363× |
| DiffMean | Activation L2 | 32 | 383.650 | 371.952 | 1.0326× |
| PromptSteering | Response margin | 1 | 565.039 | 552.008 | 1.0240× |
| PromptSteering | Response margin | 32 | 601.209 | 583.460 | 1.0305× |
| PromptSteering | Activation L2 | 1 | 440.302 | 426.687 | 1.0326× |
| PromptSteering | Activation L2 | 32 | 467.933 | 450.816 | 1.0375× |

All 112 measured score/gradient comparisons are exact; 54 of 56 score pairs
favor the candidate. Median paired scoring-time reductions span 2.35–3.62%.
Gradient median ratios span 0.9699–1.1040×; raw gradient ratios span
0.6459–1.6374× and raw score ratios span 0.6091–1.0911×. All observations
are retained, including the two slower score pairs. Gradient timing controls
show substantial variation; no gradient or whole-search speedup is claimed.

All 161 targeted MLP, RMS and worker tests pass. The separate missing-C-compiler
real-model check retains four exact score/gradient comparisons and verifies
zero fused calls in both arms. Prepared payloads and all 144 L2 clean rows also
match exactly in each applicable case. The sources and every collected file
were hash-verified. The complete evidence archive is retained in
`out/axbench-gcg-performance/loop28/final/hopper/qualification.tar.gz`, with
SHA-256 `abf9229c80180744cf6b21c8805255b96a601e79f7e872a4ccdb5f610bb9141e`.

These results use the original archived worker, before selected-factor L2 and
clean-reference fusion. They qualify the square-fusion change on H200;
combined selected-factor runtime verification remains the separate 4090
qualification documented above.


### Correctness and retained evidence

All 68 final RMS tests pass on the RTX 4090. They include every one of 65,536
BF16 square inputs, offset pointers and tail guards, every one of 2**32 BF16
weight-product pairs for both FP32 and BF16 kernel inputs, nondefault-stream
buffer reuse, and an actual allocation extending beyond 2**31 elements. The
existing native norm families, keyword interface, empty/nonfinite cases,
aliasing, fallback and exception-restoration checks remain covered. Non-NaN
kernel outputs match bit for bit; NaN classification must match, without
requiring identical payload bits.

Before adding the stream test, the RMS/worker subset passed 81 tests. The final
root CPU subset passes 85 with 76 GPU cases deselected, and all four real-model
integration tests pass. All 43 executable Python wrapper lines are covered;
Triton arithmetic is verified by device tests and retained IR/PTX. Two mutation
controls pass and all seven injected faults are detected. Existing cross-batch
BF16 tolerances stay unchanged. Ruff, formatting and scoped type checks pass;
the full `tests/` suite was not run.

The tested RMS module SHA-256 is
`2860072c0c747e8015ec70bedd941a413b9dde9a6a21088dc3ff66cc5927d7af`.
Raw reports, tensors, compressed traces, resource audits, test logs, frozen
references, mutation cases and source snapshots are retained under
`out/axbench-gcg-performance/loop28/final/`. The local collection has
236 files and 72,663,004 retained bytes;
its manifest SHA-256 is
`690dd471a1199ca8381c01a0a00f830cea07ac1b937457083880d8f7ac6a49f0`.

This optimization leaves objective construction unchanged. The combined
selected-factor qualification after PR #1716 is recorded above. The original
timing matrix uses factor one and does not qualify arbitrary checkpoints or
distributed GPU execution.

## Rejected reciprocal-square-root fusion

The next RTX 4090 experiment retained native FP32 mean and epsilon addition,
then moved reciprocal square root into the postscale kernel. The Triton
libdevice version failed on all 8,388,607 positive FP32 subnormal inputs:
its generated `rsqrt.approx.ftz.f32` returned infinity where native Torch
returned finite values. This variant was rejected before model timing.

Explicit `rsqrt.approx.f32` matched native Torch across all 2**32 FP32 input
patterns, requiring exact non-NaN bits and matching NaN classification.
The complete wrapper passed 56 shape/epsilon cases and 126 paired component
comparisons. Against frozen PR #1714 (`9754f3c1b`), all 56 measured model
gradient/score comparisons were exact, but only 11/28 score pairs improved;
the four median paired score ratios ranged from 0.99572 to 1.00152. That
version was also rejected.

A further prototype reused the row value when a complete kernel block lay
within one row. Retained PTX and machine code confirm that this reduces
eight reciprocal-square-root instructions to one for width 2048. It passed
99 additional shape/epsilon cases, including offset operands and widths
around the block boundary. The component screen compared both vector and
scalar variants with the parent: all 252 paired comparisons per variant
were exact. Large component shapes still had mixed or regressive timings.
The model comparison retained 56 exact gradient/score pairs; 19/28 score
pairs improved, with median scoring-time reductions of 0.25–0.29% for
response margin and 0.01–0.07% for activation L2. These small, uneven gains
do not justify adding the kernel logic, so this prototype was rejected too.

Both model probes used SmolLM3-3B revision `it-soup-APO`, DiffMean, 72 prepared
pairs, four selected pairs, physical batch four, suffix lengths 1/32 and
seven measured alternating pairs after warmup. Resource audits observed no
foreign GPU process or pytest run during the accepted probes. These are
local screening results; no Hopper or Blackwell result is claimed. Production
continues to use native Torch reciprocal square root. Frozen sources, raw
timings, tensors, instruction dumps, resource audits and rejection decisions
are retained under `out/axbench-gcg-performance/loop29/`.

## Rejected packed MLP projections

The retained PR #1714 traces attribute about 82% of summed scoring-kernel
duration to matrix multiplication. A follow-up screen combined each native
MLP's gate and up projections into one wider multiplication. It used the
current stack at `f7d8f0a5a`, including the selected-factor L2 change.

The RTX 4090 component screen used real SmolLM3 weights from layers 0 and 28,
four profiled row counts and three larger cases, both deterministic and normal
execution, and 21 alternating measured pairs per case. The first version used
native SiLU/multiply and was slower in 24/28 cases.
A layout-aware version retained fused SiLU/multiply and improved all 28
component medians, with paired ratios of 1.004–1.083 times the reference.
These component measurements exclude packing the weights; packing adds
3,246,391,296 bytes across this model's 36 MLPs. Eight cases per variant
changed outputs; a component gain did not establish worker correctness.

The real-worker screen included weight packing in the measured operation and
used complete independent frozen parent modules. DiffMean response margin
with four selected pairs, physical batch two and one suffix token failed the
existing, predeclared score budget (`rtol=1e-4`, `atol=1e-6`): 4/8 scores
differed beyond tolerance, with maximum absolute error 0.0257706642 and RMSE
0.0095420489. The three measured gradient pairs and their warmup were exact.

Three-repeat controls reproduced the scoring error exactly. Separate gate/up
projections with the same strided activation matched the reference exactly.
Keeping the packed projection and making its activation contiguous retained
the original error. This isolates the combined projection as the source of
the observed score differences. The experiment was rejected without widening
the budget; later cells, complete searches and remote qualification were not
run. No production optimization or worker speedup is claimed.

The sources, paired component timings, failing score tensors, controls and
resource audits are retained under `out/axbench-gcg-performance/loop30/`.
All four audits observed no foreign GPU process or pytest run. The collection
contains 40 files and 551,365 bytes; its manifest SHA-256 is
`8f4fea481fa8d1ca432d9afd6ceeabdbaa13371a7b8aca3d346ee253717485d3`.

## Native decoding during canonical token-pool construction

The complete parent worker at `13ebb9d60f6ceca755bb644a596bf7915ad51d6d`
visits 128,000 non-excluded vocabulary entries when preparing the pinned
SmolLM3-3B tokenizer. Its CPU profile identified repeated public `decode`
input conversion as avoidable work: every input already comes from `range`
and contains one plain Python integer.

The pool iterator now passes `[token_id]` to the native tokenizer's `_decode`
with the same explicit `skip_special_tokens=False` and
`clean_up_tokenization_spaces=False` options. It retains `_text` validation,
Unicode policy, exclusions, ordered round-trip checks, and the original
per-token exception boundary. The guard requires the exact
`PreTrainedTokenizerFast` type, no instance `decode` override, and the public
method captured from `PreTrainedTokenizerBase` at worker import. It checks
these conditions for every token because encoding runs between generator
yields and can replace a decoder. Other tokenizer types and public overrides
use the existing general decoder. This adds no persistent cache or setting.

The final production worker was measured against an independent, complete
frozen parent worker in the vendor Python 3.12.11 / Transformers 5.16.1 /
tokenizers 0.23.1 environment on the Ryzen 9950X workstation. Each arm executes
four complete pool builds per block; seven alternating measured blocks follow
two warmup blocks. Garbage collection remains enabled, and all collection
time stays in the measurements. Temporary validation strings are released
before timing. No profiling callbacks run during these measurements.

| Component | Parent median (s/call) | New median (s/call) | Median paired speedup | Faster blocks |
|---|---:|---:|---:|---:|
| Complete canonical pool | 0.567116 | 0.484255 | 1.17360× | 6/7 |

The medians describe the seven block means. A reference-versus-itself control
returned a median paired ratio of 0.99485×; raw block variation is retained.
These are pool-construction measurements, not a whole-search speed claim.
All 28 measured call pairs produced the same ordered 122,908 IDs, and all
124,259 eligible `(id, text)` pairs matched exactly. The vocabulary has
128,256 embedding rows. Separate, post-timing profiles recorded 128,000
low-level decodes in both arms and reduced public decode calls from 128,000
to zero, directly confirming which work was removed.

Earlier public and backend batch-decoding screens preserved every output but
regressed, with median paired ratios of 0.75955× and 0.84850× respectively.
Initial scalar measurements were inconclusive: garbage-collection pauses
around 0.2 seconds distorted single-call medians, including a
reference-versus-itself comparison. The prospective block protocol above was
chosen before the final production measurement. All earlier samples remain
in the evidence directory; no slow samples were removed.

The final focused suites passed 265 root-environment tests and 128 vendor
environment tests. The pool iterator covered all 22 statements and 16 branches;
the pool builder covered all 20 statements and 10 branches. Eleven deliberate
faults failed their intended checks, including subclass/instance/class override
bypasses, a cached guard, changed decode flags, lost keyword delegation,
omitted text validation, an incorrect token ID, and a disabled native path.
The remaining fault exercises subclass iterator cleanup. A control passed,
and the subclass fault was rechecked after a test-only signature correction.

Four targeted real-model cases passed: DiffMean at factor 0.6 and
LinearProbe, LoRA, and PromptSteering at factor 1. Each exercises both objectives,
k=1/32, complete pool equality, the independent score and gradient oracle,
repeated gradients, search, and generation-state restoration. DiffMean and
LinearProbe completed before another pytest run started; the resource guard
stopped the remaining work, and LoRA/PromptSteering passed after the machine
became available again. The completed pool timing also predates that contention.

A separate complete-parent replay used DiffMean at factor one, 72 prepared
pairs, four sampled pairs, physical batch two, k=1, two steps, 16 proposals,
top-k 32, and 128-token sampled final generation with the fake judge. It
retains the selected-factor policy introduced by #1716 in both arms. All eight
pairs, including warmups, matched complete preparation payloads, optimizer
requests, filters, gradients, all 1,872 scored losses, byte-for-byte saved
receipts, and generated text exactly. The resumed resource audit observed no
foreign GPU or pytest process.

| Objective | Parent mean preparation (s) | New mean preparation (s) | Parent mean cell (s) | New mean cell (s) |
|---|---:|---:|---:|---:|
| Response margin | 2.093621 | 2.023788 | 16.678393 | 16.611151 |
| Activation L2 | 4.615178 | 4.545724 | 15.376889 | 15.299283 |

These three-pair workflow means are descriptive; the controlled pool component
is the accepted performance result. The [workflow tables](axbench-adv-gcg-stack.md#selected-factor-receipt-cohort)
retain every measured phase and keep the new receipt cohort separate from the
historical stack. The paired timing interval includes preparation, search,
generation, and trace auditing, with model loading excluded.

An existing cleanup test assumed Python 3.12's temporary-object destruction
order. Unmodified parent runs showed that Python 3.13 uses a different order,
with implementation and frozen reference still exactly equal on both
interpreters. The assertion now requires that exact reference order and every
cleanup event. Production ID normalization is unchanged. Ruff and formatting
checks passed; the changed tests passed `ty`. The worker retains the one
existing optional-`logits` type diagnostic, reproduced on its unchanged parent.

The candidate worker SHA-256 is
`501d604253c5b6db38012a13ec27e5dce04ed811f39af142664a92a8a5a019ab`;
the frozen parent's is
`9c35a2abbfa4b5bc32f1514276b25fc1706228449bceac673393d817a34904a4`.
Only the existing `_eligible_token_texts` definition changes; the general
`_decode`, `_ids`, and pool-builder definitions retain their parent ASTs.
Source snapshots, raw timings, profiles, mutation logs, interpreter diagnostics,
and resource audits are retained under `out/axbench-gcg-performance/loop31/`.
The final local collection contains 90 files and 727,552 bytes; its manifest
SHA-256 is `9058ff177aef0f128b7949814646c644107739b3e6e30d7b6cecca26befc3ac4`.
The qualified code commit is `f600ecffb098b70524e94d1d67c2d0698aef907e`.

This change executes on the host CPU and leaves the parent's CUDA code
unchanged. Its real-model qualification used the RTX 4090; no new Hopper or
Blackwell run is claimed. The earlier RMSNorm PRs' Hopper qualification remains
pending at publication. A fresh review of main at
`91a7604ecf28500fcf2b0df6ccaeb96d901e57b7` found only unrelated training and
documentation changes since the selected-factor integration used here.

## Fusion during clean-reference preparation

**Rejected candidate (#1718).** Exact numerical checks pass, and the 4090
component improves, but DiffMean at physical batch two has a 15–16% median
paired slowdown on two Blackwell nodes. The experiment is retained for review;
subsequent optimization work starts from the complete #1717 parent without this scope
extension. No device-specific workaround or new dispatch cache is added.

Profiling the complete #1717 worker at
`3eb74edd89227415a50f39fed6d2413edf03cd4c` identified clean-reference forwards
as 2.617 seconds of a 4.488-second instrumented activation-L2 preparation
(58.31%). The pinned 72 training pairs require 144 clean response activations:
72 forwards at physical batch two. This path still used eager MLP gates and
RMSNorm throughout. The trace recorded 2,088 SiLU calls and 4,248 calls each to
power, mean, and reciprocal square root, with no fused kernels. These are
profile-attribution numbers; instrumentation was removed for all timings below.

The proposal makes `GCGSession._compute_clean_hidden` enter the existing
`fused_scoring_mlps` scope once around its physical-batch loop, after runtime
state preservation and disabled autograd take effect. It reuses the existing
native Llama/Qwen2/SmolLM3 BF16 CUDA guards and kernels. Unsupported model
implementations, user instrumentation, and missing build prerequisites retain
the helper's eager fallback. There is no new kernel, persistent cache, setting,
batching rule, calibration arithmetic, or reference tensor representation.
Every other worker definition retains its parent AST.

The final production code at `bea1f71ae50b9c734058618960747477d5192314`
was compared with complete, independent frozen parent worker, state, scoring,
SiLU, and RMSNorm modules. The RTX 4090 qualification used the pinned
SmolLM3-3B `it-soup-APO` model revision
`cfb32d505f5025ec9be4e704f70cfbf5bdf8da94`, DiffMean at factor one, 72
prepared pairs, and one suffix token in vendor Python 3.12.11 /
Torch 2.5.1+cu124 / Transformers 5.16.1. Each batch size had one untimed full
preparation per arm, two warmup pairs, and seven alternating measured pairs.
The timed region covers the entire clean-reference computation, including
scope entry/exit and CUDA synchronization. A separate reference-versus-itself
control used one warmup pair and three measured pairs.

| Physical batch | Parent median (s) | New median (s) | Median paired speedup | Faster pairs | Reference/self median ratio |
|---|---:|---:|---:|---:|---:|
| 2 | 2.469186 | 2.369701 | 1.04118× | 7/7 | 1.00447× |
| 4 | 2.428727 | 2.304481 | 1.05431× | 7/7 | 0.99812× |

All 144 clean tensors matched exactly after every arm, including warmups and
controls. Complete prepared payloads, subsequent eight-score comparisons,
four-member gradients, and gradient-sidecar descriptors also matched exactly
for each batch size. State snapshots checked module modes, forward bindings,
hook registries, parameter gradient flags and objects/values, tensor versions,
padding side, Python/NumPy/Torch CPU/CUDA random states, and autograd mode.
Untimed dispatch instrumentation confirmed 109 replaced module forwards and
2,088 MLP / 4,248 RMS calls at batch two (half as many at batch four).
No dispatch counters or profiler callbacks ran inside measured intervals.

The preliminary scoped prototype also passed exact preparation, activation,
score, and gradient comparisons, with median paired ratios of 1.03858× and
1.05379×. Its method AST and the committed production method are identical.
The table above uses only the subsequent committed-code qualification.
These are warm clean-reference component results; cold compilation, complete
preparation, model loading, and campaign throughput are not represented by
those speedup ratios.

Focused tests passed **232 cases in the root environment and 95 in the vendor
environment**. Ten new cases cover both PromptSteering and LoRA, multiple
physical batches, exact eager-reference values, caller-state preservation,
and failures on entry, capture, a later capture after partial caching,
alignment validation, and exit. Failed preparation clears the earlier
prepared state, clean cache, direction and sidecars, then rejects later use.
Response-margin preparation never enters the clean-reference scope. The
changed method covered all 14 statements and all six branches. Three isolated
mutations—omitting the scope, entering it before disabling gradients, and
entering once per physical batch—failed their intended checks; the control
passed. The real-model clean-activation oracle now explicitly disables this
fusion while independently recomputing reference values.

Four targeted real-model integration cases passed: DiffMean at factor 0.6,
and LinearProbe, LoRA, and PromptSteering at factor one. They exercise both
objectives, k=1/32, independent score and gradient oracles, repeated gradients,
search, and generation-state restoration. A separate complete-parent replay
used DiffMean at factor one, all 72 prepared pairs, four selected pairs,
physical batch two, k=1, two steps, 16 proposals, top-k 32, and normal sampled
generation with a 128-token budget and fake judge. Both arms retain #1716's
selected-factor L2 policy. All eight paired searches, including warmups,
matched complete preparation payloads, optimizer requests, filters,
gradients, all 1,872 scored losses, byte-for-byte saved receipts, and generated
text exactly.

| Objective | Fresh #1717 mean preparation (s) | New mean preparation (s) | Fresh #1717 mean cell (s) | New mean cell (s) |
|---|---:|---:|---:|---:|
| Response margin | 2.001277 | 2.007676 | 16.593147 | 16.584450 |
| Activation L2 | 4.475893 | 4.390488 | 15.227168 | 15.139835 |

These three-pair workflow means are descriptive; the controlled clean-reference
component establishes the local 4090 improvement. It does not establish
cross-hardware acceptance. The
[workflow tables](axbench-adv-gcg-stack.md#selected-factor-receipt-cohort)
append this change while preserving earlier measurements. The measurement
JSON also retains the fresh complete #1717 reference and verifies that the
receipt contracts match the preceding selected-factor cohort. The local
qualification's resource audit observed no foreign GPU or pytest process.

Ruff and formatting checks passed. The changed tests and independent oracle
passed `ty`; the worker retains the same pre-existing optional-`logits`
diagnostic as its frozen parent, with no new diagnostics. The candidate worker
SHA-256 is
`be326ac8da135983580e4d131806cebac2261293e160ec43cf58cfa00ff8a4d0`;
the frozen parent's is
`501d604253c5b6db38012a13ec27e5dce04ed811f39af142664a92a8a5a019ab`.
Profiles, full source snapshots, paired timings, state and numerical checks,
mutation logs, interpreter diagnostics, and resource audits are retained under
`out/axbench-gcg-performance/loop32/`.
The final local qualification collection contains 71 files and 603,504 bytes;
its manifest SHA-256 is
`0d0fb15a331a56787c615270a72fb4824b3e7fa11942ccf13ef45e46cd29817d`.

The production-caller ratchet passed four checks and failed its existing
test-only-symbol check. Running that failing check on the clean, unmodified
#1717 parent reproduced the same four symbols: `_loader_health_reset`,
`conditioned_loss_forward`, `enumerate_core_task_contexts`, and
`enumerate_core_target_contexts`. The matching diagnostics and source identities
are retained in `loop32/production-callers/`; this change does not repair those
unrelated failures.

The clean-reference change is qualified locally on RTX 4090. Its first
Blackwell attempt, job `23118724`, timed out after shared-filesystem checksum
preflight consumed almost its entire allocation. A process snapshot observed
`sha256sum` in uninterruptible wait while reading a repository test file.
Slurm denied extending the time limit. No numerical result from that attempt
is accepted.

### Blackwell parity passes; one performance case regresses

Retry `23120740` completed on node5101 in 5 minutes 21 seconds. It used the
identical verified archive and staged source under `/tmp` on an `xfs`
filesystem, retaining every source-file checksum before and after execution.
The GPU was an RTX PRO 6000 Blackwell Server Edition (compute capability 12.0),
using Python 3.12.11, Torch 2.7.1+cu128, Triton 3.3.1 and eight OMP/MKL threads.
The candidate, frozen parent and numerical protocol were unchanged.

| Method | Physical batch | Parent median (s) | New median (s) | Median paired ratio | Faster pairs | Reference/self median ratio |
|---|---:|---:|---:|---:|---:|---:|
| DiffMean | 2 | 1.820611 | 2.094378 | 0.86876× | 0/7 | 0.99983× |
| DiffMean | 4 | 1.500713 | 1.432888 | 1.04725× | 7/7 | 0.99958× |
| PromptSteering | 2 | 2.095701 | 2.091649 | 1.00179× | 6/7 | 1.00025× |
| PromptSteering | 4 | 1.721249 | 1.630471 | 1.05591× | 7/7 | 0.99993× |

All 24 targeted tests passed. Complete prepared payloads, all 144 clean
tensors after every timed arm, and subsequent scores, gradients and sidecar
descriptors matched exactly in all four cases. The two warmup pairs, seven
measured pairs and three measured reference/self pairs per case are retained.

DiffMean at batch two is approximately **15.1% slower**, consistently across
all seven measured pairs. Both batch-four cases improve; the 0.18% paired
change for PromptSteering at batch two does not establish a useful speedup.
These results fail the cross-hardware performance gate. The repeat below
confirms the failure, and the unconditional scope extension is rejected.
All observations are retained in
`out/axbench-gcg-performance/loop32/blackwell/`, whose collection manifest
SHA-256 is `43917d64fd19902ab85a1cbcc45b78cfe4f60cbb0366ce896297daf0272224d5`.
The immutable source archive SHA-256 is
`507c463c8af09346407e76690f6320945013cc87d39a0a92b0d8680086e7e873`.

No Hopper result is claimed for this preparation scope. The separate H200
square-fusion qualification above does not qualify clean-reference fusion.

Diagnostic job `23122750` completed on node5004 in 12 minutes 45 seconds.
It retained the same production archive and environment, reversed method and
batch order, and repeated the paired comparisons and exact numerical/state
checks. Its complete source checks and all 24 targeted tests passed.

| Method | Physical batch | Parent median (s) | New median (s) | Median paired ratio | Faster pairs | Reference/self median ratio |
|---|---:|---:|---:|---:|---:|---:|
| PromptSteering | 4 | 1.716888 | 1.625810 | 1.05607× | 7/7 | 1.00009× |
| PromptSteering | 2 | 2.073990 | 2.079194 | 0.99732× | 0/7 | 0.99991× |
| DiffMean | 4 | 1.498908 | 1.431400 | 1.04730× | 7/7 | 0.99993× |
| DiffMean | 2 | 1.830927 | 2.130156 | 0.86129× | 0/7 | 1.00019× |

The DiffMean batch-two regression repeats in all seven measured pairs; its
median paired slowdown is 16.1%. Across both jobs, all 56 measured clean
preparation pairs match exactly. Prepared payloads, scores, gradients and
sidecar descriptors also match in each case. Clean-tensor and gradient
digests additionally agree between the two nodes for all four cases.

Separate DiffMean batch-two ablations used two warmup pairs and five measured
alternating pairs each. Eligibility overrides were installed outside timed
calls; every arm checked all 144 clean tensors and caller state exactly.

| Enabled fusion | Parent median (s) | Ablation median (s) | Median paired ratio | Faster pairs |
|---|---:|---:|---:|---:|
| MLP only | 1.824511 | 1.947907 | 0.93665× | 0/5 |
| RMSNorm only | 1.825442 | 2.071461 | 0.88184× | 0/5 |

Both isolated variants regress. Separate CPU/CUDA profiles verify exactly
2,088 SiLU launches for MLP-only fusion and 4,248 launches each of square and
postscale for RMS-only fusion. The complete candidate launches all 10,584;
the eager parent launches none of these fused kernels. All four profiles
retain exact clean tensors and caller state. The profiled function-call count
rises from 1,242,150 to 2,446,554 with full fusion, including Triton dispatch
and specialization work. This directly establishes added host work; it does
not isolate every contributor to the uninstrumented regression.

Profiler wall times reverse the apparent ranking (6.478 seconds for the
parent versus 4.046 seconds for the candidate), demonstrating substantial
instrumentation effects. They are attribution evidence only. Rejection is
based on the uninstrumented paired comparisons, with every observation
retained. No new launcher, cache, or hardware-specific policy is introduced
to rescue this candidate. The complete diagnostic archive, including all
four losslessly compressed traces, the committed candidate and frozen parent
modules, is retained in
`out/axbench-gcg-performance/loop32/blackwell-diagnostic/qualification.tar.gz`.
Its SHA-256 is `b52153ca220f3f3ff807d062c8908cc3811c1bbfdc2b36725f315704946ac588`.

A further check of the #1718 diagnostic traces found 34,132–34,140 fill kernels
and approximately 43–45 ms of fill-kernel device time in each arm. Extra
allocation filling therefore does not explain the measured regression by
itself. Added dispatch work remains a supported hypothesis, not an isolated
causal result: profiling perturbs timing, and the 4090 and Blackwell environments
use different Torch/Triton versions. The uninstrumented two-node regression is
the reason for rejecting the change.

## Native-rounded rotary arithmetic during candidate scoring

This change starts from PR #1717, `3eb74edd89227415a50f39fed6d2413edf03cd4c`,
including #1716's selected-factor L2 objective. The rejected #1718 clean-reference
fusion is excluded. An intervening input-packing experiment was also rejected:
all outputs matched, but every median scoring difference was below 0.3% with
comparable control noise. Its complete evidence is retained under
`out/axbench-gcg-performance/loop33/packed-inputs-screen/`.

The parent profile identified repeated four-dimensional rotary multiplies,
negations, additions and concatenations. The optimization replaces each query
or key rotary expression with one kernel during candidate scoring. It computes
each product with an explicit FP32 round-to-nearest instruction, converts each
product to BF16, adds the rounded products with another explicit FP32 instruction,
then stores BF16. FP32 subnormals are preserved and multiply/add contraction is
disabled. Two retained compiled PTX variants contain the intended rounded
instructions and BF16 conversions, with no floating-point FMA instruction.

The existing scoring context binds a function with the **identical native
attention code object**, native defaults and metadata, and a fresh globals
mapping whose sole replacement is the rotary callable. It modifies selected
attention instances and restores them in the existing `finally` block.
Transformers module globals and class methods remain native. The change adds
no configuration, persistent cache or alternative attention implementation.

Eligibility is restricted to exact, unmodified Llama, Qwen2 and SmolLM3
attention classes with ordinary linear projections. Custom projections,
external rotary wrappers, subclasses, hooks, compiled calls, missing compiler
prerequisites and Torch modes retain native execution. The arithmetic helper
also checks plain BF16 CUDA tensors, device agreement, disabled gradients and
autocast, projection strides, coefficient shape/contiguity, even head widths
and non-singleton dimensions. Outputs are fresh allocations; inputs may alias
and are preserved. Flat element offsets use 64 bits, with a separate bound on
the intermediate token index.

The first prototype failed exact output-stride equality for a singleton key
head. The accepted implementation narrows eligibility to non-singleton shapes;
the metadata gate was not weakened. That failed screen is retained under
`loop34/v1-layout-rejection/`. Gradient computation, clean preparation and final
generation continue through their existing native paths.

### Matched production scoring measurements

The measured production code is commit
`857333f1d773966911e78f6faaa8e7344f2aacbb`; subsequent changes extend tests and
documentation only. The reference consists of independently loaded complete
PR #1717 worker, state, scoring, SiLU and RMS modules, each checked against its
recorded source hash. Both arms use the same pinned SmolLM3-3B commit
`cfb32d505f5025ec9be4e704f70cfbf5bdf8da94`, DiffMean factor one, 72 prepared
pairs, four selected members and two scoring candidates. Suffix lengths 1/32
and physical scoring batches 2/8 give eight cases per device.

Each case has two warmup and seven measured alternating pairs, plus one warmup
and three measured reference/self pairs. Timing includes the complete
`session.score` call, guards, input construction, scalar materialization and
device synchronization. Profiling is disabled during these measurements.
The prospective gate requires at least two cases to reduce median paired
elapsed time by at least 1%, exceed reference/self noise, and no case to regress
by 1% or more. Exact numerical parity is mandatory.

All 56 measured pairs and 24 reference/self pairs per device retain exact
scores and caller state. The qualification also checks prepared payloads,
288 clean-reference tensors, 4,608 input rows, eight gradients and descriptors,
and native rotary output bits and strides. Local runs have recorded quiet
preflights and no observed foreign GPU or pytest processes. Measurements cover
warm workers; they do not establish compilation-startup or campaign throughput.

#### RTX 4090

| Objective | k | Batch | Parent median (ms) | Candidate median (ms) | Median paired elapsed reduction | Faster pairs |
|---|---:|---:|---:|---:|---:|---:|
| Response margin | 1 | 2 | 366.263 | 360.136 | 1.71% | 7/7 |
| Response margin | 1 | 8 | 353.385 | 349.856 | 1.10% | 7/7 |
| Activation L2 | 1 | 2 | 279.358 | 274.506 | 1.91% | 7/7 |
| Activation L2 | 1 | 8 | 267.605 | 263.459 | 1.49% | 7/7 |
| Response margin | 32 | 2 | 382.620 | 375.871 | 1.79% | 7/7 |
| Response margin | 32 | 8 | 377.230 | 372.214 | 1.30% | 7/7 |
| Activation L2 | 32 | 2 | 291.100 | 286.293 | 1.63% | 7/7 |
| Activation L2 | 32 | 8 | 284.582 | 279.942 | 1.42% | 7/7 |

#### RTX PRO 6000 Blackwell

| Objective | k | Batch | Parent median (ms) | Candidate median (ms) | Median paired elapsed reduction | Faster pairs |
|---|---:|---:|---:|---:|---:|---:|
| Response margin | 1 | 2 | 324.601 | 317.906 | 1.64% | 6/7 |
| Response margin | 1 | 8 | 203.518 | 198.183 | 2.53% | 7/7 |
| Activation L2 | 1 | 2 | 260.276 | 258.945 | 0.50% | 7/7 |
| Activation L2 | 1 | 8 | 162.344 | 158.968 | 2.03% | 7/7 |
| Response margin | 32 | 2 | 323.338 | 321.004 | 0.72% | 7/7 |
| Response margin | 32 | 8 | 226.317 | 222.316 | 1.65% | 6/7 |
| Activation L2 | 32 | 2 | 263.135 | 261.314 | 0.55% | 6/7 |
| Activation L2 | 32 | 8 | 180.361 | 177.125 | 1.96% | 7/7 |


The 4090 passes the prospective gate with 56/56 measured pairs faster and
1.10–1.91% median paired elapsed reductions. Its environment is Python 3.12.11,
Torch 2.5.1+cu124 and Triton 3.1.0, with the recorded thread overrides unset.
Blackwell production job **23128838** completed successfully on node5004 with
Python 3.12.11, Torch 2.7.1+cu128, Triton 3.3.1 and eight OMP/MKL threads.
It passes the same gate with 53/56 pairs faster. Batch-eight reductions are
1.65–2.53%. Some batch-two differences are below the 1% improvement threshold;
L2 k=1 batch-two is also within its reference/self noise. These small cases
are not claimed as demonstrated improvements. Both use Transformers 5.16.1.

The production H100 job **23128839** timed out on node2906 after 20 minutes
10 seconds. Its 121 CPU tests passed, but the CUDA tests started too late to
finish and no model benchmark ran. Process snapshots recorded prolonged I/O
waiting during dependency import and after the CPU test report. This attempt
does not establish Hopper GPU correctness or performance. Its retained
collection SHA-256 is
`676febd857f0b24143851f4d555be99c139cbd123eb9eb67b5e140b3188415bd`.

#### H200 Hopper

H200 retry **23133103** completed on node3401-c1 in 19 minutes 47 seconds.
It passed **121 CPU tests and 70 CUDA tests**, all 56 measured scoring pairs
and all 24 reference/self pairs. Prepared payloads, clean tensors, gradients,
descriptors, scores, caller state and rotary output bits/strides match exactly.
The unchanged prospective performance gate passes.

| Objective | k | Batch | Parent median (ms) | Candidate median (ms) | Median paired elapsed reduction | Faster pairs |
|---|---:|---:|---:|---:|---:|---:|
| Response margin | 1 | 2 | 399.861 | 399.193 | 0.34% | 6/7 |
| Response margin | 1 | 8 | 155.397 | 149.576 | 3.79% | 7/7 |
| Activation L2 | 1 | 2 | 360.848 | 360.554 | 0.08% | 4/7 |
| Activation L2 | 1 | 8 | 143.457 | 138.836 | 3.26% | 7/7 |
| Response margin | 32 | 2 | 404.066 | 404.775 | -0.09% | 3/7 |
| Response margin | 32 | 8 | 168.680 | 162.025 | 3.82% | 6/7 |
| Activation L2 | 32 | 2 | 368.088 | 367.444 | 0.30% | 7/7 |
| Activation L2 | 32 | 8 | 154.721 | 149.126 | 3.62% | 7/7 |

Batch-eight reductions are **3.26–3.82%**, with 27/28 measured pairs faster;
each case exceeds its reference/self median variation. Batch-two reductions
range from **-0.09% to 0.34%** and do not reach the 1% improvement threshold.
No batch-two improvement is claimed. No case regresses by 1% or more.
These are complete warm scoring-call measurements, excluding model loading
and compilation startup; the earlier local full-search results remain separate.

The environment is Python 3.12.11, Torch 2.7.1+cu128, Triton 3.3.1 and
Transformers 5.16.1, with four OMP/MKL threads. GPU and environment differences
are not controlled across the 4090, Blackwell and Hopper measurements.

This retry excluded the earlier timed-out node and staged 2,872 byte-verified
Transformers files on local storage before importing them. The dependency
archive SHA-256 is
`4884fe5f2284b9b953b6f566ca34699a9f4e248f77d2de852d77e99b17be7bcf`.
The tests, model, numerical/performance gates and qualification source archive
were unchanged. The source archive shared with Blackwell has SHA-256
`8daf49e65efa3ac5de9b522abb637be05477230aea0c7b971da52ad2904a358b`.
Both source and import manifests were verified in the completed job. Startup
observations still recorded a shared Torch bytecode read waiting on filesystem
I/O; that startup delay is outside the measured scoring intervals.

### Independent attribution and correctness

Separate 4090 profiles compare the complete parent and candidate for both
objectives at batches two/eight. Every profiled score and caller-state snapshot
matches exactly. Each rotary pair adds two deterministic fill kernels but
removes six net launches. Response-margin totals fall from 15,384 to 14,088
launches at batch two and 4,002 to 3,678 at batch eight. L2 totals fall from
12,528 to 11,472 and 3,252 to 2,988. Profiled host call counts increase, including
347,762 to 402,636 for response-margin batch two. These traces establish the
launch tradeoff; the separate uninstrumented paired runs establish performance.

The final focused validation includes:

- 261 adjacent scoring/worker tests in the root environment, followed by the
  final 108 rotary cases in **each** of the root and AxBench environments.
  There are 264 distinct passing root cases across those two runs. Blackwell
  and H200 each separately pass 121 CPU and 70 CUDA cases from the archived
  production qualification. The final two CPU cases and stronger
  coefficient-offset assertions were added afterward; production code is
  identical.
- Every BF16 encoding under eight coefficient patterns, independent random
  operand bit patterns, signed zero, subnormals, infinities, NaN classification,
  ragged widths, grouped heads, coefficient batch broadcasting, independent
  query/key/cosine/sine storage offsets, aliases, nondefault streams and offsets
  exceeding 2**31. Non-NaN BF16 bits and output strides match native arithmetic.
  These tests do not enumerate all possible four-operand tuples or require
  identical NaN payload bits.
- Native Llama, Qwen2 and SmolLM3 attention checks for eager/SDPA execution,
  cache updates, input gradients, unsupported inputs, overrides, optional
  imports, nested contexts and failure restoration. The new path preserves
  global bindings and unrelated model instances.
- Two passing unmodified mutation controls and detection of all 16 deliberately
  broken variants, covering both product roundings, rotation sign/offset,
  head/coefficient indexing, wide offsets, autograd/autocast/negation guards,
  singleton layout, Boolean dimensions, global isolation, defaults, restoration
  and external overrides. The initial report-parser interruption is retained
  alongside its corrected completion; it did not change a numerical gate.
- 116/116 Python statements and 38/38 reported branches in the scoring module,
  plus all reported Python lines/branches in the rotary wrapper functions.
  Coverage does not measure the compiled Triton body or independently enumerate
  every Boolean subcondition; GPU arithmetic tests, mutations and retained PTX
  provide separate evidence for those paths.
- Four existing real-model IPC cases: DiffMean factor 0.6, LinearProbe, LoRA and
  PromptSteering, both objectives and suffix lengths 1/32. Fixed-shape scores
  and gradients use exact independent native oracles. The existing tolerance
  for comparing different BF16 physical batch sizes remains unchanged.
- A further per-method comparison against the complete frozen parent verifies
  exact scores and state for both objectives, with actual kernel counts.
  DiffMean, LinearProbe and PromptSteering have 36 eligible attention modules.
  The saved LoRA adapter changes only `o_proj` in layers 7, 14, 21 and 28; those
  four attention modules retain native execution while the other 32 qualify.
  Its response-margin/L2 calls launch 96/76 rotary kernels versus 108/88 for
  the other methods. The first probe's incorrect all-LoRA-fallback assertion
  is preserved; the completed probe verifies the saved and loaded adapter
  configuration before checking this mixed dispatch.
- Eight full DiffMean search pairs, including warmups, with exact optimizer
  requests, filtering, all 1,872 scores, gradients, prepared payloads, receipt
  bytes and generated text. The [workflow tables](axbench-adv-gcg-stack.md)
  preserve historical PR columns and add both the current column and a fresh
  paired #1717 reference. Three measured pairs per objective are descriptive
  whole-search observations, not a campaign-throughput claim.

Scoped Ruff formatting/lint, type checking and whitespace checks pass. The
full `tests/` suite was not run. No paid judges or Hugging Face uploads were
used. Rejected #1718's implementation remains outside this branch; only its
reviewed performance history and completed earlier Hopper evidence are carried
forward in the documentation.

All raw pairs, reference/self controls, complete-search receipts, original
failed probes, source hashes, final tests, audit logs and compressed profiles
are retained in `out/axbench-gcg-performance/loop34/`. The following collection
hashes identify the accepted local, Blackwell and Hopper evidence and validation.

| Evidence directory under `loop34/` | Collection SHA-256 |
|---|---|
| `production-screen/` | `ccd474034d333397422f1059c96d0047265a1f0c7946a75bd503103400a6c719` |
| `production-blackwell/` | `74b1f7a8301d77594e09e2702f134dc214b1fd81a40c56ef791711337382eb4e` |
| `production-hopper-local-import-retry/` | `69b20288fe9d886b27afdb734e5ba20ca8f07f8d639135e5a2c548b8e18fd4ea` |
| `production-full/` | `a0a2b0c0fd93811b471de4bf6a8b5c39e26d0ae146726fb6276b71ebb7a928b2` |
| `production-profile/` | `d1f0addccd570fd68e9c0bc867e2998890cfb845add8d98c4d049cab68225b49` |
| `production-mutations/` | `eaeddb4fc1b347eff13d78e7e96ceb5f5f5b60feeffc7eb8b72ecef6804de8f6` |
| `production-methods/` | `eaa47042a7d09281da3abd5bea8cadca56eb7d6912dab526a84eb9ac61f08eee` |
| `production-coverage/` | `39b4a08cab6757b29ba77f9fbbb7af0a68fa4cc7e8736088193c1c6130f456b1` |

## Rejected empty-hook cleanup shortcut

The rotary-scoring profiles also attribute host work to PyVene's repeated
hook cleanup. A scratch prototype skips `list(registry.keys())` only for an
empty, exact `dict` or `OrderedDict`, retaining native iteration and deletion
for other registries. It introduces no cache. The experiment temporarily
replaces the helper during candidate scoring; no production integration is
accepted.

The complete parent is rotary-scoring commit
`3f3dec366c64b68dfbd77a8e2c252163b0e20950`. The same pinned SmolLM3 model,
72-pair input, four members, two candidates, exact numerical gate, alternating
two warm/seven measured pairs, and three reference/self controls are used.
The timer includes the temporary replacement and restoration. A separate
untimed probe checks hook-registry identities and contents against native
cleanup; the observed registries are all empty, so this is not qualification
of nonempty or custom registries.

| Objective | Suffix tokens | Physical batch | Median paired elapsed reduction |
|---|---:|---:|---:|
| Response margin | 1 | 2 | 0.21% |
| Response margin | 1 | 8 | 0.05% |
| Activation L2 | 1 | 2 | 0.06% |
| Activation L2 | 1 | 8 | 0.21% |
| Response margin | 32 | 2 | 0.48% |
| Response margin | 32 | 8 | 0.29% |
| Activation L2 | 32 | 2 | 0.40% |
| Activation L2 | 32 | 8 | -0.03% |

All 56 measured score pairs and 24 reference/self pairs match exactly, as do
prepared payloads, 288 clean rows, 4,608 constructed input rows, eight gradient
descriptors/values and caller state. The resource audit reports no foreign GPU
or pytest activity. Although 40/56 candidate pairs are faster, every case falls
below the predeclared 1% improvement threshold. The prototype is rejected;
profiled cleanup time did not translate into a sufficient measured gain.

The prospective plan, prototype, frozen parent, raw pairs, audits and analysis
are retained under `out/axbench-gcg-performance/loop35/`. The collection SHA-256
is `460739423e28c9dba06a96bcb57ea74da91f14759a69380e72799212d53af420`
and analysis SHA-256 is
`c5ca6031c4e4a46739b93bea36d6788926126f46cfa9c7051c475aee864a584e`.

## Rejected joint query/key rotary launch

A second scratch experiment combines the query and key rotary kernels into
one launch, assigning separate program ranges to each operand. An AST comparison
confirms unchanged eligibility checks, projection-layout checks and per-element
arithmetic after the program-index calculation. Both output allocations and
their native strides remain independent. The reference is complete rotary PR
commit `987cd49446a13140dca3a1439a56d6c7d3de5062`.

All 70 adapted CUDA contracts pass on the 4090, including arithmetic bit checks,
native attention/gradient/cache behavior, fallbacks, aliasing and both output
tensors beyond signed 32-bit offsets. Separate model probes observe one kernel
request per rotary pair and two exact native outputs. The response-margin
batch-two case therefore requests 216 rotary kernels instead of 432; no
instrumentation runs during the paired timing.

The same eight-case protocol and prospective gate from the preceding experiment
give the following results:

| Objective | Suffix tokens | Physical batch | Median paired elapsed reduction |
|---|---:|---:|---:|
| Response margin | 1 | 2 | 0.22% |
| Response margin | 1 | 8 | -0.05% |
| Activation L2 | 1 | 2 | 0.27% |
| Activation L2 | 1 | 8 | 0.12% |
| Response margin | 32 | 2 | -0.02% |
| Response margin | 32 | 8 | -0.05% |
| Activation L2 | 32 | 2 | 0.25% |
| Activation L2 | 32 | 8 | 0.16% |

All 56 measured score pairs, 24 reference/self pairs, prepared payloads, 288
clean rows, 4,608 constructed input rows, eight gradients/descriptors and caller
states match exactly. The resource audit reports no foreign GPU or pytest
activity. Only 33/56 pairs are faster and no case reaches the predeclared 1%
improvement threshold. The extra kernel branch and helper are not added to
production.

Artifacts are retained under `out/axbench-gcg-performance/loop36/`. Collection
SHA-256 is `8c5d31c3edc2608eddb5e410509c6a82bd9a4085600676fd8d619c605b00eaae`;
analysis SHA-256 is
`c2371d0d34c8a3c00f48d04acb270b7db91969c9fff4c39b945a5c398acd349d`.

## Native-order RMS square and mean

Candidate scoring previously allocated a full FP32 squared-hidden tensor,
then launched native mean for every eligible RMSNorm. The width-2048 path now
computes the same variance directly from BF16 inputs. It removes that large
intermediate and one kernel launch per norm. The epsilon addition, native
`rsqrt`, BF16 rounding before weight multiplication, and fresh output allocation
retain their existing operations.

This is one specialization inside the existing scoring helper. It accepts
contiguous BF16 operands, width 2048, at least 16 flattened batch/token rows,
and fewer than `2**29` input elements. All existing caller eligibility checks
still apply. Other shapes and unqualified Torch revisions use the preceding
helper. Preparation, gradients, selected-factor L2 semantics, and generation
continue through their existing contexts.

### Reduction order and revision fence

The qualified Torch revisions are 2.5.1
`a8d6afb511a69687bbb2b7e88a3cf67917e1697e` and 2.7.1
`e2d141dbde55c2a4370fac5165b0561b6af4798b`. For this shape,
[`Reduce.cuh`](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/Reduce.cuh)
selects 32 lanes per row, four sequential accumulators per lane, their
left-associated combination, and ascending shuffle distances 1, 2, 4, 8, 16.
The kernel writes out that order with explicit round-to-nearest FP32 additions
and multiplications. The native FP32 squares allocation would be aligned;
unlike the BF16 input, it has no unaligned vector-load head. The element limit
conservatively keeps its byte span below the
[TensorIterator indexing bound](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/TensorIterator.cpp).
At least 16 rows keeps the native block at 32 lanes by 16 rows, with neither
an inter-warp nor an inter-block reduction for each output.

An early variant passed the vendor Torch 2.5.1 tests but failed 37 of 68
checks under root Torch 2.13. Its CUDA reduction uses descending shuffle
distances. A controlled diagnostic reused identical BF16 input bytes in both
environments: each native result exactly matched its corresponding forced
order, while the opposite order differed in 48/128 rows by at most one ULP.
Both forced orders produced identical bits across the environments. The
production revision fence therefore preserves the old helper on Torch 2.13;
it does not add another reduction tree or relax the numerical gate.

### Measured scoring performance

The complete independent reference is PR #1720 at
`506ba8e1d1051f961cdc805a7d617446602c685d`. The measured production code is
`e93fb76651afc8bb2269361425644f056dc85bbf`; its RMS kernel SHA-256 is
`5c05bf2b8bf3d85a1122e336c81a86c64819059fa301a884f8064c395023c779`.
The later documentation and exact-cutoff test extension leave those production
bytes unchanged. The reference includes the complete worker, state, scoring,
SiLU, RMS and rotary modules with independent globals. It excludes rejected
#1718's clean-reference scope extension.

Each case uses pinned SmolLM3-3B, DiffMean factor one, 72 prepared pairs, four
selected members and two candidates. Two alternating warmup pairs precede
seven measured pairs; three reference/self pairs measure local noise. Timers
cover the complete scoring call and final synchronization, with no profiler.
The predeclared gate requires at least two cases with 1% median paired elapsed
reduction beyond reference/self median noise, and no 1% regression.

| Objective | Suffix tokens | Physical batch | 4090 elapsed reduction | Blackwell elapsed reduction | H200 |
|---|---:|---:|---:|---:|---:|
| response_margin | 1 | 2 | 1.24% | 1.88% | 1.03% |
| response_margin | 1 | 8 | 1.47% | 1.67% | 1.75% |
| activation_l2 | 1 | 2 | 1.57% | 1.22% | 1.04% |
| activation_l2 | 1 | 8 | 1.33% | 1.36% | 0.81% |
| response_margin | 32 | 2 | 1.40% | 1.90% | 1.25% |
| response_margin | 32 | 8 | 1.19% | 1.58% | 1.82% |
| activation_l2 | 32 | 2 | 1.36% | 1.69% | 1.13% |
| activation_l2 | 32 | 8 | 1.81% | 1.52% | 1.90% |

All eight cases pass on the 4090 and Blackwell. Every measured score and
runtime state matches exactly; 56/56 pairs are faster on the 4090 and 55/56 on Blackwell.
The 4090 uses Torch 2.5.1+cu124/Triton 3.1; Blackwell uses
Torch 2.7.1+cu128/Triton 3.3.1 with eight OMP/MKL threads. Blackwell job
23138110 completed on node5101 in 3:39. These are scoring-component results.

H200 job `23138111` completed on node3401-c1 in 4 minutes 54 seconds, using
Torch 2.7.1+cu128, Triton 3.3.1, Transformers 5.16.1 and four OMP/MKL threads.
All 56 measured score pairs and 24 measured reference/self pairs preserve
scores and runtime state exactly; preparation, clean tensors, gradients,
descriptors and native variance/output bits also match. Seven cases exceed
the 1% gate and their controls, with reductions of 1.03–1.90%; 53/56 pairs are
faster overall. The k=1/batch-eight L2 row improves only 0.81% and its control
ratio is 0.96909x, so that row is not claimed as an improvement. The overall
prospective gate passes with no 1% regression. These are complete warm score
calls, not whole-search or cross-environment speed comparisons.

Separate prototype profiles, whose reduction AST matches production, reduce
response-margin total kernel counts from 14,088 to 13,504 at batch two and
3,678 to 3,532 at batch eight. L2 counts fall from 11,472 to 11,000 and
2,988 to 2,870. L2 retains 16 terminal loss-mean calls. Fill-kernel counts do
not change; the large squared-hidden allocation is replaced with a small
variance allocation. Host profile call counts increase slightly, so these
profiles do not establish a host-dispatch reduction. The retained PTX uses
explicit rounded operations, no FMA/FTZ instruction, 38 registers, no spills,
and 1,024 shared bytes. Profile timings are not performance acceptance data.

### Correctness and qualification limits

The focused subsets cover 205 distinct root-runtime cases and 139 distinct
vendor-runtime cases, including 71 new CUDA cases. They check intermediate
FP32 variance and native RMS output, all BF16 encodings in constant/isolated
rows, mixed magnitudes, storage offsets, streams, aliasing, epsilon extremes,
revision/shape fallbacks, and real allocations below, at and above the cutoff.
The existing exhaustive BF16 weight-pair tests and wider-than-int32 element
indices also pass in both environments. Synthetic non-NaN values match by
bits; NaN classification matches without requiring payload equality.

Both Python wrappers have full line coverage and all 12 recorded branches
covered. Python coverage does not measure Triton arithmetic or independently
cover every Boolean subcondition. Eleven isolated deliberate faults in loads,
reduction association/order, divisor, row mapping, subnormal handling and
eligibility fences fail the intended assertions; the unmodified control passes.

Four real-model IPC cases cover DiffMean at factor 0.6 and LinearProbe, LoRA
and PromptSteering at factor one, both objectives, suffix lengths 1/32,
native scores/gradients, bounded searches and generation. Existing documented
cross-batch BF16 tolerances remain separate from the exact fixed-shape gates.
Further untimed method probes exercise the new kernel through all four methods
and match 1,056 intermediate variances and RMS outputs exactly.

Eight complete paired searches match prepared payloads, optimizer requests,
filters, gradients, all 1,872 scores, receipt bytes and generated text exactly.
The [progress tables](axbench-adv-gcg-stack.md#matched-native-rms-mean-comparison)
include each workflow phase, a new historical column, and the freshly
remeasured #1720 reference. Three measured pairs per objective are descriptive
workflow observations, not a campaign-throughput claim.

Blackwell and H200 each pass 85 CPU and 119 CUDA cases from the frozen
production snapshot. The one-case exact-cutoff extension was checked later
in both local runtimes and is outside those remote test counts; production
kernel bytes are identical. The full `tests/` suite was not run. Scoped Ruff,
formatting, type and whitespace checks are recorded with the final source review.

All plans, frozen references, paired controls, tests, original failures,
profiles and audit logs are retained under `out/axbench-gcg-performance/loop37/`.
The first prototype used `tl.gather`, unavailable in Triton 3.1, and failed
before arithmetic or model timing. A later local report mislabeled the rotary
reference path; `reference-path-metadata-correction.json` verifies all six
loaded frozen files against the parent commit. The original report is preserved;
the reference code, numerical comparisons and timings were unaffected.

| Evidence directory under `loop37/` | Collection SHA-256 |
|---|---|
| `production-screen/` | `347cdb2d39982fab968bfbbc7d710ecf77f7257778941a8c0235fedbef832eb0` |
| `production-blackwell/` | `4ffa3070b359ac72dc13003d886d5e7ea445f1acbbde21a3d05e2ac68c52f581` |
| `production-hopper/` | `c8c7587f965d717a4cd509fa202ece296f2410d2e676890c47b382993df74440` |
| `production-full/` | `9608b2de1236d931aadce21f5f8edde798b2353f760824dae89ae9968821043d` |
| `production-methods/` | `c511d5bd558cea8d3bb58d101842f5b9282cc4014cfdc2de0f486dd1d6405bfc` |
| `production-mutations/` | `75732ae4767f54d13eb854c7720612528f8ea7904909a230830f0b827696d884` |
| `production-coverage/` | `e6a1465a3b8874fe488cccbbe40bacb43e31b955ebfdd730be6bd785e93ffb57` |
| `prototype-profile/` | `1b0d923db0bd14d61bb282508081ca04d9ace2fdf4417f7daa0676ab17b32800` |

### Follow-up experiments on input staging and native masks

The following scratch experiments compare complete scoring calls against the
six frozen #1721 modules at `bcc2619bfec5d470f93abd4c5188fafa05d49a9e`.
They use the same pinned SmolLM3 model, DiffMean factor one, 72 prepared pairs,
four selected pairs, two candidates, both objectives, suffix lengths 1/32,
and physical batches 2/8. They are not production changes or whole-workflow
measurements.

Pinned CPU token/mask construction with asynchronous CUDA copies alone
preserved exact inputs, clean activations, gradients, scores and caller state,
but reduced elapsed time by only **0.05–0.98%** across the eight cases. None
passed the predefined 1% screening threshold. Each environment passed 126
focused cases, including delayed-stream checks that release the pinned Python
tensors and exercise allocator reuse before the copy completes. Separate
profiles showed input-copy waits moving to native mask scalar checks for
response-margin, and to clean-reference transfers for L2. These waits include
queued GPU work and cannot be interpreted as CPU construction costs.

The combined prototype changes only eligible scoring batches: it stages the
integer inputs asynchronously and asks Transformers' native mask factory to
materialize the mask without checking whether it can be omitted. Unequal,
positive sequence lengths already establish that the right-padded mask has a
zero within the query length, so the ordinary skip checks would return false.
The prototype preserves equal-length batches and other unsupported cases on
the original path. It is limited to the inspected causal SmolLM3 SDPA path
without sliding attention; the older root Transformers API also falls back.

All 56 measured pairs pass exact checks, including 40 direct comparisons of
native mask values, shapes, dtypes and strides, 288 clean tensors, 4,608 input
rows and eight gradient arrays/descriptors. Both environments pass 34 focused
mask/fallback cases; the vendor environment additionally exercises native
mask preparation and tiny-model outputs. Initial test-collection and fixture
failures are retained with the corrected runs.

A fresh four-arm ablation rotates execution order and retains two warm and
seven measured groups plus reference/self controls for every case. All 224
measured score calls preserve exact outputs and state. The two response-margin
batch-two cases reproduce the useful improvement:

| Complete scoring case | Staging only | Native mask only | Combined |
|---|---:|---:|---:|
| Response margin, k=1, batch 2 | 1.076% | 0.374% | 2.141% |
| Response margin, k=32, batch 2 | 0.841% | 0.212% | 1.938% |

Values are median paired elapsed-time reductions against the complete parent.
The combined change exceeds each single mechanism in both cases. L2 and
batch-eight reductions stay below 1% in this ablation. Separate profiles
verify identical matrix/attention kernel counts and removal of input-copy and
mask-check synchronizations for response-margin. L2 retains its 16
clean-reference copy synchronizations. Profile durations are attribution only.

These prototypes motivated the response-margin-only production qualification
below. The production change preserves the original preparation, gradient and
L2 paths. The earlier evidence and workflow measurements remain unchanged.

| Evidence directory under `out/axbench-gcg-performance/` | Collection SHA-256 |
|---|---|
| `loop42/prototype-v2-screen/` | `21ec32f0473db7919007cf1e793d3e372d86f6d1dc26950cbabb4b501b904909` |
| `loop42/prototype-v2-profile/` | `3264226ec877b09379b7fffa743f3e23a28e77727bda6ee1afb7ff39d005a5f1` |
| `loop43/prototype-v2-screen/` | `fe77357ad3336bcb50ec15e3686b9dbf3ccbd941196982cc08b1641c4b1b2fb7` |
| `loop43/prototype-ablation/` | `95a34a3117818d93d9acc7823f900dc2aec50b2c5085144ba65512b261fc3e13` |
| `loop43/prototype-profile/` | `11c9231c75cdeafc7bc08547e68f7a08b420873325a900eedd78ff7c07180223` |

## Native input staging during response-margin scoring

This change starts from complete PR #1721 at
`3909cac4efa86f8788b8ac478c4baa9017e8a9dc`. Production code was frozen at
`8c78b0ddc5ba37bebd3316395b8c04ddbabb9ee1`; the subsequent five review cases
and documentation leave those production bytes unchanged. Main was refreshed
to `91a7604ecf28500fcf2b0df6ccaeb96d901e57b7`, with no main-only changes to
the relevant GCG paths. The selected-factor L2 behavior from #1716 is retained.

The input and native-mask profiles above identified synchronization during
row-by-row GPU integer construction and during native causal-mask skip checks.
For eligible response-margin scoring batches, `_padded_embeds` constructs the
same right-padded token IDs and attention flags in two pinned CPU tensors and
copies them on the current CUDA stream with `non_blocking=True`. The ordinary
embedding lookup, detached leaf behavior, physical batches, row order and loss
calculations are unchanged.

Unequal positive sequence lengths prove that the ordinary native skip checks
would encounter a padding zero within the query length. The worker asks the
same Transformers `create_causal_mask` factory to materialize that mask with
`allow_is_causal_skip=False`, then passes its native boolean four-dimensional
result through the existing forward. Mask shape, dtype, strides and values
are checked against the ordinary two-dimensional-input route. The inspected
SmolLM3 forward creates the same default position IDs for attention; there is
no KV cache or packed-position input on this path.

Eligibility requires native causal SmolLM3 SDPA without sliding attention,
native model/backbone/embedding forwards, and the original mask factory and
registry entry. DiffMean and LinearProbe use their native PyVene wrapper;
LoRA uses its native PEFT wrapper and tuner without prompt learning or replaced
input-hook contexts; PromptSteering uses the base model directly. Root,
backbone, embedding and wrapper observers, changed input getters or padding
methods, global hooks, Torch modes, autocast, tracing and compilation preserve
the original path. Equal-length, single-row and unsupported model/API batches
also preserve it. The Transformers 5.14 root API lacks the explicit skip flag
and falls back; the installed 5.16 vendor API exercises the optimization.

Preparation, gradient calls and activation-L2 scoring never enter the new
eligibility helper. Their original method bodies remain equal as ASTs after
removing the added response-margin branch and optional padding branch. Actual
method probes separately fail if those operations enter staging. The change
uses PyTorch's existing pinned allocator and native mask factory; it adds no
configuration, custom kernel, persistent input buffer or cross-call cache.

### Complete scoring measurements

The model is the pinned SmolLM3-3B checkpoint at revision
`cfb32d505f5025ec9be4e704f70cfbf5bdf8da94`. The DiffMean factor-one cohort has
72 prepared pairs, four selected pairs and two candidates, both objectives,
suffix lengths 1/32 and physical batches 2/8. Each case retains two warmup and
seven measured alternating pairs, plus one warmup and three measured
reference/self controls. Timing covers the complete score call and device
synchronization. No profiler or dispatch observer runs inside those intervals.

The complete frozen parent worker, state helper, scoring dispatcher and
SiLU/RMS/rotary modules are loaded independently. Every measured score and
caller-state comparison is exact, as are complete preparation payloads, 288
clean tensors, 4,608 directly constructed input rows and eight gradient arrays
and descriptors per architecture. Each architecture also checks 20 native
response-margin masks directly. Explicit staging in the input-helper probes
does not imply that production gradients or L2 use it.

Values below are median paired elapsed-time reductions. Positive means less
time. The prospective gate requires at least two cases to improve by 1% or
more beyond their reference/self median noise, with no case regressing by 1%
or more. Exact parity is mandatory.

| Objective | Suffix tokens | Physical batch | RTX 4090 | Blackwell | H200 |
|---|---:|---:|---:|---:|---:|
| response_margin | 1 | 2 | 2.002% | 2.680% | 0.271% |
| response_margin | 1 | 8 | 0.459% | 1.217% | 1.531% |
| activation_l2 | 1 | 2 | -0.012% | 0.054% | 0.012% |
| activation_l2 | 1 | 8 | -0.334% | -0.411% | -0.177% |
| response_margin | 32 | 2 | 1.999% | 1.865% | 0.513% |
| response_margin | 32 | 8 | 0.236% | 1.015% | 1.850% |
| activation_l2 | 32 | 2 | -0.060% | 0.231% | -0.137% |
| activation_l2 | 32 | 8 | 0.177% | -0.095% | 0.229% |

Both response-margin batch-two cases pass on the 4090; all four response-margin
cases pass on Blackwell. Subpercent variations, including every L2 row, are not
claimed as demonstrated improvements. The 4090 uses Python 3.12.11,
Torch 2.5.1+cu124 and Triton 3.1.0 with OMP/MKL thread overrides unset.
Blackwell job `23151427` completed on node5101 in 2 minutes 42 seconds, using
Torch 2.7.1+cu128, Triton 3.3.1 and eight OMP/MKL threads. Both use Transformers
5.16.1, PyVene 0.1.8 and PEFT 0.13.2. These are within-environment comparisons;
they do not isolate the effects of GPU architecture from framework versions.

H200 job `23151429` completed on node3401-c1 in 2 minutes 44 seconds with
Torch 2.7.1+cu128, Triton 3.3.1 and four OMP/MKL threads. Both response-margin
batch-eight cases pass the same gate, reducing complete scoring time by
1.531% and 1.850% beyond their reference/self controls. Batch-two differences
and L2 controls are subpercent and are not claimed as improvements. All 56
measured score pairs and 24 measured reference/self pairs preserve scores and
caller state exactly. The other preparation, clean/input tensor, gradient and
native-mask comparisons above also pass; no case regresses by 1% or more.

The [stack table](axbench-adv-gcg-stack.md#matched-scoring-input-comparison)
adds a column for this change and retains all historical values. A fresh
complete-parent comparison has one warmup and three measured search pairs per
objective, with 128-token generation and a deterministic fake judge. All eight
pairs preserve optimizer requests, filters, gradients, 1,872 scores, complete
prepared payloads, receipt bytes and generated text exactly. These phase means
are descriptive; the separate scoring experiment supplies the performance gate.
The unchanged L2 preparation path also varies between these small samples.

### Correctness and review evidence

The new file contains 234 distinct cases: padding and fallback contracts,
CUDA tensor/leaf/state comparisons, native masks and tiny-model outputs,
wrapper and observer boundaries, and the final inference/getter checks.
Including 75 existing nearby worker, batching, layer-prefix and state cases,
the focused subsets pass 279 root cases with 30 optional-dependency skips,
and all 309 cases in the vendor environment. Every root skip passes against
the actual PyVene/PEFT dependencies in the vendor environment. Blackwell
passes the earlier frozen 203 CPU and 101 CUDA cases; its production source
is identical, but its archive predates the five added review cases.

Delayed-stream tests release both pinned Python tensors while their DMA copies
are still pending, then churn 64 same-sized pinned allocations. Neither
protected pointer may be reused before completion; IDs, masks and embeddings
must remain exact. Both default and nondefault streams are covered. Separate
real-model checks cover all four methods, with DiffMean/LinearProbe at factor
0.6 and LoRA/PromptSteering at factor one: 576 clean tensors, eight gradient
arrays/descriptors and every score match the complete parent. Each method
actually enters the native response-margin mask path. Four real-worker IPC
cases also cover both objectives, suffix lengths 1/32, bounded searches and
generation, using the independent native numerical oracle and fake judging.

Python coverage exercises all 69 added executable lines and all recorded
branches originating in added code. The two eligibility helpers and padding
helper have complete statement and branch coverage. This does not enumerate
every Boolean subcondition or prove DMA correctness by itself. All 13 deliberate
faults fail their intended assertions, while the unchanged control passes all
229 initial cases through the same source-overlay mechanism. Faults cover bad
padding/masks, blocking copies, removed API/mode/registry/forward/getter/PEFT
guards and accidental staging during L2 or gradients. Mutation tests leave
production files untouched.

The final four CUDA review cases score inside inference mode and then request
gradients after leaving it, in BF16/FP32 on both streams. Losses and gradients
remain exact, cached response constants remain ordinary tensors, and parameter
versions, gradients and RNG stay unchanged. A fifth case replaces the embedding
getter at class scope and requires fallback before that observer is invoked.
All five pass in both environments with ordinary settings, including an unset
CuBLAS workspace policy, and with the established deterministic settings.
All pre-existing test AST nodes remain unchanged.

Ruff, formatting, whitespace checks and scoped `ty` pass. The root configuration
excludes the separate-environment worker; the changed test file is type-checked
within that scope. Forcing the excluded worker into the root invocation produced
dependency/API diagnostics and is not counted as a successful check. The full
`tests/` suite was not run.

Initial failures are retained: the first guard attempted `t.device` before
rejecting `TorchFunctionMode`, which the test caught; the corrected ordering
passes. A later vendor test invocation collected unrelated root dependencies;
using the existing `--confcutdir tests/axbench_adv` boundary resolved that
collection failure. Neither attempt supplied accepted performance evidence.

Blackwell and H200 each pass 203 CPU and 101 CUDA cases from the same
immutable production archive. That archive predates the final five local
review cases described above; the measured production code is identical.
This qualifies the remote component protocol, while complete searches remain
the separate local evidence described above. Archive SHA-256:
`12a7fc7ffcc9bfdae888f3182d6109a0ce11b9853db8d309fe185a599844850f`.
The measured production worker SHA-256 is
`57fb344a662baca3476a1bef9fa46814bab94f0a21a8d62ea78ec83a6f807949`.

All plans, six complete parent snapshots, original failures, raw measurements,
environment records, receipts, test logs and resource audits are retained under
`out/axbench-gcg-performance/loop44/`. Local resource audits observed no foreign
GPU or pytest processes during the accepted runs.

| Evidence directory under `loop44/` | Collection SHA-256 |
|---|---|
| `production-tests/` | `a56007a8a976343e07b5941be6a20c4628cddff2d7bb75a02f4eade38f574889` |
| `production-screen/` | `712048b6e063cdd31bccdbc9dc2bbe87549cb78a01ef80eb4e0e219a9c43f2b2` |
| `production-blackwell/` | `6c3a96c42079473e9b7d825860355730faacb8a5e97dd5e29273a53358a4ce63` |
| `production-hopper/` | `9dc8bd004658311fc095ffb15f81fa6dbf9b0dfd9741476c9b5cb3f36c798b45` |
| `production-methods/` | `bc8fc108b52df7cd6676bb4b8dec3695ec37702c5ddd79e0652961b7cf51669d` |
| `production-full/` | `d1c6a3c31eb962df92decb218c79c98ad6b618fc62f0b4c0958d368a428bd726` |
| `production-mutations/` | `cf547299bbcfda7bc860954ef6b64db074d7427e87b584585b6496c60f34a39d` |
| `production-coverage/` | `75802dc05ad159fa828abca12f547d6f7ce4323dfb0bae645343c80b6aca7c59` |
| `review-tests/` | `0f8dcfc0e6ef049979da09ffaac8659f4a71cbb4817bb299125d4f33aae3d430` |

## Pinned clean-reference transfer screen (rejected)

The next experiment starts from complete PR #1723 at
`a673342aad249bd46c5bf186d0128f102ea3fe5d`. Earlier pageable nonblocking
copies had shown little benefit. This scratch prototype instead pins each
existing private CPU clean-reference tensor once during preparation, then uses
`non_blocking=True` for its ordinary copy before the L2 forward. Both copies
and their consumers use the current CUDA stream. PyTorch's
[native CUDA copy implementation](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cuda/Copy.cu)
records the pinned allocation's stream use for asynchronous copies. The
experiment adds no persistent GPU reference cache and changes no model or
objective arithmetic.

The source and prospective gate were frozen before execution. The complete
six-module parent is independently loaded. Both objectives, suffix lengths
1/32 and physical batches 2/8 use 72 prepared pairs, four selected pairs and
two candidates. Each case has two warmup and seven measured alternating
pairs, plus one warmup and three measured reference/self controls. No profiler
or copy observer runs inside the timed complete-score calls.

All 56 measured pairs and 24 measured controls preserve scores and caller state
exactly. Prepared payloads, all 288 clean tensors and eight gradient arrays and
descriptors also match exactly. Separate observation verifies the clean-copy
row order, source pin state and `non_blocking` argument. Each environment passes
48 focused cases, including CPU/CUDA, BF16/FP32, default/nondefault streams,
both objectives, ragged batches and inference scoring followed by gradients.

| L2 suffix tokens | Physical batch | Median paired elapsed-time reduction | Faster pairs |
|---|---:|---:|---:|
| 1 | 2 | 0.697% | 7/7 |
| 1 | 8 | 0.374% | 7/7 |
| 32 | 2 | 0.425% | 6/7 |
| 32 | 8 | 0.450% | 6/7 |

Response-margin controls range from -0.064% to 0.090%. No case reaches the
predefined 1% improvement threshold. The prototype would also pin 147,251,200
bytes (140.43 MiB) of live clean-reference storage per prepared cell in this
cohort; this is not a measurement of peak allocator reservations. The screen
therefore rejects the proposal before further preparation-cost, lifetime,
all-method or architecture qualification. Production source and workflow
timing columns remain unchanged.

The 4090 run completed without observed foreign GPU/pytest contention. Plans,
source snapshots, test logs, raw timings, copy observations and resource audits
are retained under `out/axbench-gcg-performance/loop45/`. Prospective plan
SHA-256: `d2c7197d9e76102c61f486df63fb9e85d30ce42e2a9d9670ffb0be6e4f9fefd8`.
Collection SHA-256:
`f77d77c3548841bffc52af0088b8fcd517115cdc2c238502471ae59de3c703cc`.

## L2 loss-removal diagnostic

Before implementing another loss kernel, a separate experiment measures the
available headroom on complete PR #1723 at
`6a69d922ffcbd21fcf14d0658ca0f994b65f8f89`. Seven complete modules, including
the shared objective math, are independently loaded from that commit. The
diagnostic records the 16 native GPU loss scalars for one fixed payload and
replays them while retaining the full model forward, clean-reference copies,
row handling and final pair-score construction. Replay would be wrong for
other inputs; this is a cost-removal experiment, not a production proposal or
a correctness qualification for a replacement loss.

The pinned DiffMean/4090 cohort uses 72 prepared pairs, four selected pairs,
two candidates, suffix lengths 1/32 and physical batches 2/8. Each case has two
warmup and seven measured alternating pairs, plus one warmup and three measured
reference/self controls. All fixed-input scores and caller/wrapper/RNG state
match in the 28 measured pairs and 12 measured controls. Both arms synchronize
the device inside each complete-call timing interval. All timing finishes
before profiling begins.

| Suffix tokens | Physical batch | Median paired elapsed-time reduction | Reference/self median ratio |
|---|---:|---:|---:|
| 1 | 2 | 0.236% | 0.99487x |
| 1 | 8 | 0.050% | 0.99854x |
| 32 | 2 | 0.049% | 1.01691x |
| 32 | 8 | -0.058% | 0.99893x |

No case reaches the predefined 1% investigation threshold or exceeds its
reference/self median noise. The larger k=32/batch-two control variation is
retained. These measurements do not establish a strict upper bound on all
possible implementations, but they provide no demonstrated headroom sufficient
to justify a new loss fusion for this workload.

Four subsequent profiles at k=32 verify that the native 16 norm and 16 mean
calls become zero in replay. The earlier RMS square/mean and postscale
dispatches remain present; all matrix and attention kernel names/counts match
between arms, totaling 1,864 at batch two and 466 at batch eight. Profile
durations provide attribution only. The first collection attempt counted CPU
and GPU user-annotation entries together because they share a name. Raw trace
categories verify 16 entries on each device; the corrected collector separates
them. The initial checker failure and the category evidence are retained, and
the benchmark/probe sources and raw measurements were not changed.

The resource audit observed no foreign GPU/pytest contention. The experiment
adds no production code or workflow-table column. Frozen sources, complete raw
timings, loss layouts, losslessly compressed profiles and collection checks are
retained under `out/axbench-gcg-performance/loop46/`. Prospective plan SHA-256:
`e02672d8fd5989e9a334abe46db5fcc2ebcdf8b792c455683a7d75d93306e863`.
Collection SHA-256:
`a65eb9106e857e310d39ec2ec97b7187b7882acdfe44fd81d266fae23c6ff26d`.

## Packed encoder index reuse screen (rejected)

The final project-model screen starts from complete PR #1723 at
`37e59de8fb3068b76aa07bddff89737d946d7e94`. A scratch-only prototype changes
one expression in `stack_packed_encoder_outputs`: resolve the validated
Boolean mask with `nonzero().flatten()` once and reuse those integer indices
for the three existing gathers. It introduces no cache, custom kernel,
configuration or compiler-policy change. The helper is also used by batched
self-conditioned generation, so broader caller qualification was conditional
on demonstrating a useful complete-call gain first.

The frozen 4090 protocol uses one project checkpoint, SelfConditioning, both
objectives, one suffix token, physical batch four, 72 prepared pairs, four
selected members and two candidates. Each objective runs in a fresh process.
Each score and gradient comparison has three warmup and seven measured
alternating pairs, followed by one warmup and three measured reference/self
controls. Timings include synchronization and finish before profiling begins.
The prospective gate requires at least one complete-score reduction of 1% or
more beyond its control variation, with no complete-score regression of 1%
or more. Gradient or instrumented component gains cannot satisfy this gate.

All 64 focused CPU/CUDA cases pass across FP32/BF16, empty/full/prefix/gapped
masks, contiguous/strided values and repeated rows. They compare values,
layout fields, strides and input gradients against the frozen parent and an
independent selection oracle, and check source immutability and no aliasing.
For both objectives, every recorded score/gradient pair is byte-identical.
Caller state, including parameter gradients, module modes, hooks and RNG,
is preserved. All 144 clean L2 tensors match with zero tolerance. This includes 28 measured score/gradient pairs and
12 measured reference/self pairs across both objectives, plus all warmups.

| Objective / complete call | Parent median (ms) | Candidate median (ms) | Median paired elapsed reduction | Faster pairs | Reference/self median ratio |
|---|---:|---:|---:|---:|---:|
| response_margin / gradient | 83.981 | 83.934 | 0.056% | 4/7 | 0.99607x |
| response_margin / score | 85.277 | 85.346 | -0.196% | 3/7 | 0.99510x |
| activation_l2 / gradient | 331.023 | 329.859 | -0.248% | 3/7 | 1.00614x |
| activation_l2 / score | 103.222 | 102.534 | 0.517% | 5/7 | 0.99929x |

The paired reduction uses the median of per-pair ratios, which can differ
from the ratio of the two independent medians. One candidate L2 gradient
sample is 0.943 seconds; it remains in the data, and this screen does not
establish its cause. No score case reaches the 1% threshold. The proposal is
rejected before expanded method, generation, full-search or remote testing.
Production source and the existing workflow-table columns remain unchanged.

Separate profiles confirm the intended mechanism for both objectives:
`aten::nonzero` calls inside stacking fall from 24 to 8 for gradients and
48 to 16 for scores. Corresponding kernel launches fall from 232 to 152 and
464 to 304; stream synchronizations fall from 24 to 8 and 48 to 16. Matrix
and attention kernel names/counts remain identical. These operator reductions
do not establish an accepted complete-call speedup.

Response-margin completed before contention interrupted the subsequent L2
load; its independently verified evidence is retained. Interrupted loading
attempts, a stopped preflight and an out-of-space preflight produced no L2
timing data. The final L2 run completed after a 600-second idle preflight,
with no foreign GPU/pytest process observed during the accepted run. Space
recovery preserved every retained archive's bytes; only verified duplicate
storage and raw profiles with verified compressed copies were reclaimed.
The compiler recompile limit remains eight in every recorded phase.

Frozen source, plans, every timing sample, raw numerical tensors, profiles,
resource audits and interruption boundaries are retained under
`out/axbench-gcg-performance/loop47/`. Prospective plan SHA-256:
`72a1579ee0b13dc0e546d110bfcd8b868c02c3ae9e9a5210456e39c393324471`.
Accepted collection SHA-256:
`160443e8ac9fe8d784e7e8da8897b763d8c13f4c4a618bea332e52244261990c`.
The optimization loop pauses after this cycle at the user's request.
