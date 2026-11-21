"""Edit-space-generic GCG optimization machinery.

Hosts the reusable GCG core extracted from the AxBench-BPB-Adv token attack: the
``EditSpace`` protocol, the ``run_gcg`` loop, and the model-agnostic candidate-search,
edit, and aggregation primitives. The conditional encoder-decoder feature-conditioning
optimizer (the package's eventual purpose) builds on this core.
"""
