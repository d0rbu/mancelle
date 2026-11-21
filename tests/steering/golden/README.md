# Self-conditioned ChatCORE goldens

`self_conditioning_configs.json` retains the original AxBench config/RNG byte
snapshots from main `ba20a6443608d0a319a00a1f9b1cb396a860a339`. The oracle was
the complete pre-refactor `methods.py`, SHA-256
`6d5b4b676d0f18a5542edd964ad55553a029436a93d5879924b5abc31be607a4`, also
identical at `7b641554f`; it did not call the shared builder.

`chat_self_conditioned.json` was captured by running the two family cases of
`test_all_six_tasks_both_families_main` against a `git archive eb5aec39a` source
snapshot, with dataset acquisition replaced by the existing local task fixtures
and the repository test-owned CPU encoder reference. It records natural greedy
predictions, accuracies, all decoder-input/slot/logit hashes and selected-feature
hashes. CUDA was hidden; OMP_NUM_THREADS=1. The snapshot used the checkout git
metadata only to satisfy receipt construction; receipt identity is not a golden.

Run the tests to compare. Do not regenerate the expected values from the changed
implementation when reviewing an equality failure.
