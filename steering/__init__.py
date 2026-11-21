"""steering — extraction, evaluation, and feature-path utilities.

Importing this package triggers the SAELens registry-injection shim that
adds gemma-scope-2 releases SAELens 6.42 does not register but which exist
on HuggingFace. The side effect is idempotent. See
:mod:`steering.sae_resolver_v2_registry` for details.
"""

from __future__ import annotations

from steering import sae_resolver_v2_registry as _registry_shim  # noqa: F401
