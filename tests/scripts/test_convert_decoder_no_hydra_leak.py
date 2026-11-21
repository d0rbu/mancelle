"""Pin: ``scripts/preprocess/convert_decoder_to_encdec.py`` is import-safe.

Slice-12 audit map P2 #12 finding: the script historically called
``initialize_config_dir(...)`` and ``compose(...)`` at module-import
time WITHOUT a ``with`` block.  This left Hydra's global state in an
"initialized" state for any subsequent importer (e.g., a test that
imports ``scripts.preprocess.convert_decoder_to_encdec`` to call helper
functions).  Worse, the entire body of the script executed at
import-time.

The fix wraps the Hydra block in ``with initialize_config_dir(...)``
and gates the body behind ``def main(): ... if __name__ == "__main__":``.

This test pins both:

1. Importing the module DOES NOT leave Hydra initialised globally.
2. The module exposes a ``main()`` callable; the body is not at module
   level.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from hydra.core.global_hydra import GlobalHydra


@pytest.fixture(autouse=True)
def _clean_hydra_state():
    """Clear any pre-existing GlobalHydra state before / after each test."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


class TestConvertDecoderImportSafety:
    def test_module_import_does_not_initialize_hydra(self) -> None:
        """After importing the script, Hydra global state is NOT initialised."""
        # Force a fresh import of just this submodule. NOTE: do NOT pop
        # the ``scripts.preprocess`` parent — popping the parent breaks
        # later tests that import sibling submodules (the parent becomes
        # a fresh empty namespace package and attribute lookups like
        # ``scripts.preprocess.count_tokens`` raise AttributeError).
        sys.modules.pop("scripts.preprocess.convert_decoder_to_encdec", None)

        importlib.import_module("scripts.preprocess.convert_decoder_to_encdec")

        assert not GlobalHydra.instance().is_initialized(), (
            "Importing scripts.preprocess.convert_decoder_to_encdec leaked "
            "GlobalHydra state. The module's Hydra block must be wrapped in "
            "a ``with initialize_config_dir(...)`` block and the body must "
            "live inside ``def main()``."
        )

    def test_module_exposes_main_callable(self) -> None:
        """The module exposes a public ``main()`` callable (CLI entrypoint)."""
        mod = importlib.import_module("scripts.preprocess.convert_decoder_to_encdec")
        assert hasattr(mod, "main"), (
            "scripts.preprocess.convert_decoder_to_encdec must expose a "
            "``main()`` callable for CLI execution under ``__main__`` guard."
        )
        assert callable(mod.main)
