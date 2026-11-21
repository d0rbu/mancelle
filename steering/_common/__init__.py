"""Shared helpers consumed by multiple training/eval scripts.

This package hosts small, pure-function helpers that were previously
duplicated across ``scripts/train/``. Hoisting them here keeps the
training scripts focused on orchestration while letting the helpers be
unit-tested in isolation.

Per the 2026-05-05 codebase audit (slice 11 / consolidation manifest
sections 1.8 / 1.9 / 1.10), the design rule is:

* ``nanochat/`` is read-only (CODEOWNERS-gated). Any cross-script
  helper that would otherwise live there ships under ``steering/_common/``.
* Each helper has a focused docstring referencing the originating
  script, so future readers can trace lineage.
* Helpers must be pure (no side effects, no module-level state) so they
  remain trivially testable.
"""
