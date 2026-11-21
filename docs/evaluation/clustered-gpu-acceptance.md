# Clustered GPU acceptance (PR #739 H3)

This runbook drives the real l22/f65k clustered contract through a bounded
two-representative AxBench Stage 0a to BPB-Adv acceptance. It is an acceptance
gate, not a benchmark or a research-quality evaluation: the adversarial grids
are reduced to their zero-step/control coordinates and the Stage 0b-3 source
bridge is deterministic. Stage 0a, clustered preflight, extraction, target
models, candidate derivation, scoring, aggregation, and plotting use production
code.

The two concept sources deliberately cover both projection cases:

- source `344` remains representative `344`;
- sibling source `1026` projects to representative `89`, whose allowed-member
  set includes `1026`.

The test fails rather than skips when an asset, GPU, contract, output, or resume
invariant is absent.

## Pinned contract

The committed acceptance config is
`configs/eval/axbench_bpb_adv/axbench_bpb_adv_clustered_gpu_acceptance.yaml`.
It pins:

- decoder-only `d12_base`, step `054600`;
- final encoder-decoder `l22_main_h100_8rank_db16_v1`, step `054600`;
- l22/f65k context-independent, no-BOS feature statistics;
- the exact whitelist and schema-v1 static cluster file;
- producer splice batch size `64`;
- the explicit two-shard validation selection manifest;
- all fourteen clustered BPB-Adv methods;
- BPB class margin, one pair, one-device-row batches, zero optimizer steps,
  and disabled general-validation scoring.

Expected stable identities:

| Identity | Expected value |
|---|---|
| Feature contract, iteration checkpoint | `3b91132728f46ebe06a3d5600c0a7f89a2017503063b978f9fb3069f0a16afda` |
| Feature contract, final checkpoint | `979a087474e0a1b5d44ddb650be732d3754f7efc10d218d4f880113e465fa538` |
| Cluster descriptor | `5738db84ba00c8b7a84cd28e7a6b5ebbd4d8385bc81262cb63d904265ec95039` |
| Canonical cluster content | `8f4d1f3288aec0a59b7126ea5d300060ac1a88b41f365cfb85c780af7c15640b` |
| Selection hash | `e9e395739f1ac4c9498f0935e2e48a407ae9eb13b0134904d70f7ec64a3d5be3` |
| Source projection | `344 -> 344`, `1026 -> 89` |

SAELens currently performs an HTTP range request for a safetensors header even
when the complete SAE file is cached locally. The test replaces only that
shape-only transport with `safe_open` over `hf_hub_download(...,
local_files_only=True)`. Model/SAE resolution and weight loading remain the
production paths, and `HF_HUB_OFFLINE=1` plus `TRANSFORMERS_OFFLINE=1` make an
uncached dependency fail.

## GPU discipline

Run `nvidia-smi` immediately before each pytest invocation. Do not start while
another compute process is present. The test repeats the process query before
its initial and resume passes, ignores only its own PID, and records
`nvidia-smi` before and after each pass.

Do not pipe pytest output. Use a fresh `/tmp` work root when changing the target
checkpoint or any semantic input.

## Iteration checkpoint

```bash
nvidia-smi
PR739_H3_ENCDEC_TAG=i768_pr3_real_cpu_smoke_20260723 \
PR739_H3_ENCDEC_STEP=1 \
PR739_H3_WORK_ROOT=/tmp/pr739-h3-small-YYYYMMDDTHHMMZ \
uv run --offline pytest \
  tests/axbench_bpb_adv/integration/test_clustered_gpu_acceptance.py \
  -m very_slow -vv -s
```

## Final l22 evidence

The final target is the checked-in default, so it needs only a retained work
root:

```bash
nvidia-smi
PR739_H3_WORK_ROOT=/tmp/pr739-h3-l22-YYYYMMDDTHHMMZ \
uv run --offline pytest \
  tests/axbench_bpb_adv/integration/test_clustered_gpu_acceptance.py \
  -m very_slow -vv -s
```

The test invokes the complete stage dispatcher twice. During the first pass it
records the artifact tree after every stage. During the second pass it requires
every stage to leave all committed non-profile bytes unchanged, then compares
the complete final tree to the first pass.

## Mutation gate

The test copies the cluster YAML into
`$PR739_H3_WORK_ROOT/mutations/`, removes one member from one cluster, and
re-runs production contract resolution against the same checkpoint. It requires
rejection at the checkpoint/effective-vocabulary boundary. The canonical
cluster file is never edited.

This is intentionally a content mutation, not a path-only mutation: the
mutated file has its own SHA-256 and its content no longer agrees with the
checkpoint’s authoritative persistent vocabulary and cluster descriptor.

## Evidence

`preflight-evidence.json` is committed to the work root before the first GPU
boundary. It records checkpoint coordinates, feature/cluster/selection/stats
identities, source projection, Stage 0a candidate/provenance hashes, clustered
source-contract hash, and mutation rejection.

On full success, `acceptance-evidence.json` additionally records:

- checkpoint model and metadata SHA-256;
- initial and resume wall-clock;
- `nvidia-smi` snapshots;
- per-stage wall-clock and before/after artifact SHA-256 maps;
- exact fourteen-method table coverage and row counts;
- final artifact SHA-256 map.

The profiler output is under the BPB run’s `profile/` directory. Profile files
are excluded from byte-exact resume comparisons because timings are expected to
change; all scientific and presentation artifacts remain included.

## What this proves

A pass proves that one real clustered checkpoint:

- agrees exactly with the configured mask, cluster content, compact feature
  vocabulary, statistics, SAE identity, and packed selection;
- preserves direct and sibling-evidence concept projection through Stage 0a
  provenance;
- rejects changed cluster content before target execution;
- loads the real extraction Gemma/SAE and decoder/encoder-decoder checkpoints;
- routes, derives, scores, aggregates, and plots every locked clustered method;
- produces schema-correct, feature-contract-stamped attack, candidate, and
  score tables for both representatives;
- resumes every stage without changing committed artifact bytes.

It does not establish research-quality metric values, nonzero-step optimizer
quality, full-scale source synthesis, multi-rank behavior, or performance at a
larger concept count. Those are separate gates.
