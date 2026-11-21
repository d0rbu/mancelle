"""Re-export session-scoped fixtures for the integration test directory.

pytest discovers fixtures via conftest.py files in the directory hierarchy.
Importing here avoids each test file having to import + shadow the name
(which causes ruff F811 "redefinition of unused" on function parameters).
"""

from __future__ import annotations

from tests.steering.feature_gcg._equivalence import real_model  # noqa: F401
