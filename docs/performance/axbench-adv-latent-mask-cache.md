# AxBench-Adv project packed decode-mask reuse

This continuation stacks on [#1762](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1762).
Its L40S [stage trace](evidence/axbench-adv-mask-cache-baseline-stage.json)
on source `7daeb05f1719fd7d79268fdc3a1600b8d8a294aa` measured a median
11.649 s generating ten responses and 10.002 s judging them across eight
warmed project method/pressure cells. The median wall time was 12.652 s.
A synchronized [single-response profile](evidence/axbench-adv-mask-cache-generation-profile.yaml)
put 1.143 of 1.174 s in autoregressive decode; prefill took 0.025 s and
engine construction 0.001 s. The checkpoint refreshes its turn-delimited
self-conditioning state every 33 emitted tokens. The
[operator capture](evidence/axbench-adv-mask-cache-operators.json) recorded
1,536 compiled cross-attention calls and 544 sort calls in one 128-token
generation. Profiling affects absolute timing; these counts motivated a
bounded routing-cache experiment.

`project_reuse_single_token_mask=true` reuses the packed cross-attention
BlockMask only while a batch-one eager generation keeps the same encoder
layout object and target slot. The existing per-forward routing validation
still runs. A state refresh builds a fresh mask, and `finally` clears the
cache at the end of each generation. The option defaults to `false` and is
limited to the three project self-conditioned methods. It does not change
token seeds, score policies, counter hooks, or judging.

## Screening replay

Engaging job `23630714` completed on an NVIDIA L40S with the same source,
400M step-1013 checkpoint, ten inputs, and all three project methods under
control, DiffMean, and LinearProbe. A temporary identity-scoped cache kept
11,064 mask hits and 357 misses. All nine generation hashes matched the
unmodified stage replay exactly, and each cell made 20 synthetic judge calls.
Excluding the first cold control cell, generation time across eight cells
fell from 92.529 to 82.446 s (1.122×), and cell wall time fell from 101.125
to 91.196 s (1.109×). This comparison spans separate GPU jobs and is a
screen, not the final paired speedup claim. The
[screen report](evidence/axbench-adv-mask-cache-screen-report.json) and
[cache counts](evidence/axbench-adv-mask-cache-screen-counts.json) preserve
the raw observations.

The five evidence files above have SHA-256 values, in link order:
`bcc63c59735bdce123aa2a2b44864981cb88d89889c9fc479e26e2045b2f9233`,
`5b1bd4d0c1b6536ab2a6f50bc026721caf2d43a513ba28b9618eaee5022b9716`,
`7db2d6719a97f7901d372d1c80ef9a4f06432e7f818915ef41d16119d63b6737`,
`17a0524bc07f4295961deed3ba062ddac898cd7740d3d40a9b52c79ad6bad40c`,
and `2dcc9fc1703fe9dac91ff09087474a5c645c63984a93727dfe2eaaaad20f32bd`.

The screen uses a deterministic in-process judge with a one-second wait per
response. It does not measure live-provider latency, quotas, rate limits, or
the final campaign stages.

## Paired GPU qualification

Engaging job `23631536` completed successfully on an NVIDIA L40S with product
source `1ed472d5139d6b2afe7e80969acccd32a93ef329`. It used the actual
framed judge worker against a local deterministic HTTP endpoint, a fresh SQLite
cache for each replay, ten inputs, all three project methods, and control,
DiffMean, and LinearProbe conditions. Judge-pair concurrency stayed at two.
Each condition ran mask reuse off/on/on/off on the same loaded runtime. The
[raw paired report](evidence/axbench-adv-mask-cache-paired.json) was written
only after all 36 `RawEvaluation` tuples matched exactly within their group;
generation hashes also matched. Each cell made 20 judge calls and 20 synthetic
paid misses, with endpoint concurrency peaking at two.

The final warm off cells totaled 116.317 s; the median of the two on cells per
condition totaled 105.510 s. This saves 10.807 s across nine ten-response
cells, a **1.102x cell speedup**. Every condition improved (1.091x to 1.113x).
The first off replay is excluded from the speedup calculation because it
includes cold effects. Peak CUDA allocation ranged from 9.137 to 9.177 GiB;
peak host RSS was 64.180 GiB. The [500 ms GPU samples](evidence/axbench-adv-mask-cache-gpu.csv)
span startup and all modes, so their 23.56% mean utilization and 11.302 GiB
peak device residency cannot be assigned to one mode. Source validation took
24.096 s and runtime startup took 274.187 s; this replay does not establish a
whole-campaign speedup.

Engaging job `23631871` ran a direct numerical probe on the same product
source, checkpoint, L40S class, and one DiffMean counter response. The
[probe script](../../scripts/benchmark/benchmark_axbench_adv_mask_logits.py)
captured 128 single-token model outputs of shape `1 x 128 x 32768` with reuse
off and on. The [report](evidence/axbench-adv-mask-cache-logits.json) records
exact tensor equality (maximum absolute logit difference 0.0) and identical
generated text. The checked-in probe differs from the executed script only by
binding loop-local values explicitly and casting the fake judge factory for
lint/type checking; these edits do not change the measured calls.

The paired report, GPU samples, logit report, and checked-in probe script have
SHA-256 values, respectively:
`2dd1e9adbffaab531e767415e1f5c3153956acc9ee4e50f7f635c475cf06ec37`,
`31fe3732d33f286ca0ec79053f249db2310982c04ab7ae111d33c64f7f19259d`,
`99c8fa2c909567fb54f6bf2f5d4ad03c419cdb78374b5113290eec5553f4992f`,
and `ad3c08e28ea1aea990bdb2bdfa7ac252031eed3c1cfccf192a880cb972be7c4a`.

This setting remains opt-in and should be enabled only for a fresh qualified
project run after the campaign owner selects its source and replay plan. The
active campaign was not changed. Live-provider latency, rate limits, and the
full final-judge/publication campaign remain outside this benchmark.
