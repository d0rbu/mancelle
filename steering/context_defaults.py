"""Shared context-length defaults for the feature/training pipeline."""

from __future__ import annotations

# One contract across Gemma extraction, dataset splicing, packed rows, and
# conditional pretraining. Individual experiments may still override this
# explicitly; defaults must not silently create incompatible artifact families.
DEFAULT_CONTEXT_WINDOW = 1024
