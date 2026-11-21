"""Tests for get_base_dir() default path and NANOCHAT_BASE_DIR override."""

import os
from pathlib import Path
from unittest.mock import patch


from nanochat.common import get_base_dir


class TestGetBaseDir:
    """Test get_base_dir() returns correct paths."""

    def test_default_returns_repo_nanochat_cache(self) -> None:
        """Without NANOCHAT_BASE_DIR, default is <repo_root>/nanochat_cache."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NANOCHAT_BASE_DIR", None)
            result = get_base_dir()
        repo_root = Path(__file__).resolve().parent.parent.parent
        expected = str(repo_root / "nanochat_cache")
        assert result == expected

    def test_env_override(self, tmp_path: Path) -> None:
        """NANOCHAT_BASE_DIR overrides the default."""
        custom_dir = str(tmp_path / "custom_cache")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": custom_dir}):
            result = get_base_dir()
        assert result == custom_dir

    def test_creates_directory(self, tmp_path: Path) -> None:
        """get_base_dir() creates the directory if it doesn't exist."""
        custom_dir = str(tmp_path / "nonexistent" / "nested")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": custom_dir}):
            result = get_base_dir()
        assert os.path.isdir(result)
