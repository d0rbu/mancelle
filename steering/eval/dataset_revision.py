"""Dataset cache provenance shared by pinned evaluation data loaders."""

from pathlib import Path
from typing import Any


def validate_dataset_revision(dataset: Any, revision: str) -> None:
    """Require every cached data file to attest the requested Hub revision."""
    files = dataset.cache_files
    if not files or any(Path(row["filename"]).parent.name != revision for row in files):
        raise ValueError(f"loaded dataset cache does not attest requested revision {revision}")
