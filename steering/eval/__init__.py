"""Evaluation and inference utilities for steered language models.

Re-exports are intentionally omitted so that importing ``steering.eval.<submodule>``
does not eagerly pull in heavy dependencies (e.g. ``safetensors``, which has a
PyO3 module that can conflict with coverage's C tracer / other PyO3 modules).
Import names directly from their defining submodule.
"""
