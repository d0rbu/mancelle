"""Unit tests for ``nanochat.hydra_cli.parse_config_name_and_overrides``.

This helper backs the ``config_name=<name>`` CLI convention shared by
``scripts/chat_sft.py`` and ``scripts/training_progression.py``. The
convention matters because:

1. Scripts launched via ``torchrun`` can't use Hydra's ``@hydra.main``,
   so they use ``compose()`` and parse argv by hand.
2. If ``config_name=`` leaked through as a Hydra override, Hydra would
   try to set a top-level ``config_name`` field on the composed config
   (which doesn't exist), and crash.
3. If ``config_name`` were missing, the scripts would keep using the
   baked-in default and silently ignore user intent.
"""

from __future__ import annotations

import pytest

from nanochat.hydra_cli import parse_config_name_and_overrides


class TestParseConfigNameAndOverrides:
    def test_default_when_absent(self) -> None:
        """No config_name= in argv → fall back to the default."""
        config_name, overrides = parse_config_name_and_overrides(
            ["foo=bar", "baz=qux"], default_config_name="chat_sft"
        )
        assert config_name == "chat_sft"
        assert overrides == ["foo=bar", "baz=qux"]

    def test_explicit_overrides_default(self) -> None:
        """config_name=custom → custom; other args remain as overrides."""
        config_name, overrides = parse_config_name_and_overrides(
            ["config_name=sft_d12_10k_safety", "save_every=500"],
            default_config_name="chat_sft",
        )
        assert config_name == "sft_d12_10k_safety"
        assert overrides == ["save_every=500"]

    def test_config_name_filtered_from_overrides(self) -> None:
        """The ``config_name=`` token MUST be stripped from overrides.

        If it leaked, Hydra's ``compose`` would attempt to set a
        ``config_name`` field on the composed config and crash.
        """
        _, overrides = parse_config_name_and_overrides(
            ["config_name=foo", "model_tag=d12"],
            default_config_name="chat_sft",
        )
        assert "config_name=foo" not in overrides
        assert overrides == ["model_tag=d12"]

    def test_dash_args_ignored(self) -> None:
        """torchrun-style ``--`` separator and long flags aren't overrides."""
        _, overrides = parse_config_name_and_overrides(
            ["--", "foo=bar", "--standalone"],
            default_config_name="chat_sft",
        )
        # The ``--`` and ``--standalone`` are skipped; only ``foo=bar`` remains.
        assert overrides == ["foo=bar"]

    def test_empty_argv(self) -> None:
        config_name, overrides = parse_config_name_and_overrides(
            [], default_config_name="training_progression"
        )
        assert config_name == "training_progression"
        assert overrides == []

    def test_only_config_name(self) -> None:
        config_name, overrides = parse_config_name_and_overrides(
            ["config_name=my_cfg"], default_config_name="training_progression"
        )
        assert config_name == "my_cfg"
        assert overrides == []

    def test_config_name_position_irrelevant(self) -> None:
        """The arg order within argv shouldn't matter — we find the token."""
        a_first, _ = parse_config_name_and_overrides(
            ["config_name=A", "x=1"], default_config_name="D"
        )
        name_last, _ = parse_config_name_and_overrides(
            ["x=1", "config_name=A"], default_config_name="D"
        )
        assert a_first == name_last == "A"

    def test_first_config_name_wins(self) -> None:
        """If the user passes ``config_name=`` twice, the first wins. This
        matches the intent of the helper doc ("at most one should be passed")
        and pins the behavior so it doesn't drift silently."""
        config_name, _ = parse_config_name_and_overrides(
            ["config_name=first", "config_name=second"],
            default_config_name="D",
        )
        assert config_name == "first"

    def test_value_with_equals_sign_preserved(self) -> None:
        """Override values that contain ``=`` must survive split(.., 1)."""
        _, overrides = parse_config_name_and_overrides(
            ["weird=a=b=c"], default_config_name="D"
        )
        assert overrides == ["weird=a=b=c"]

    def test_config_name_with_equals_in_value_preserved(self) -> None:
        """Not a recommended usage, but split("=", 1) should keep trailing
        ``=`` tokens in the name so we don't silently drop characters."""
        config_name, overrides = parse_config_name_and_overrides(
            ["config_name=foo=bar"], default_config_name="D"
        )
        assert config_name == "foo=bar"
        assert overrides == []

    def test_empty_config_name_value_raises(self) -> None:
        """``argv=["config_name="]`` (empty value) used to silently produce
        config_name="", which crashes Hydra's compose() deep in the
        resolver with a confusing MissingConfigException. Reject eagerly
        with a clear error message."""
        with pytest.raises(ValueError, match="config_name= must have a non-empty value"):
            parse_config_name_and_overrides(
                ["config_name="], default_config_name="D"
            )

    def test_empty_config_name_mixed_with_other_args_raises(self) -> None:
        """Defensive: even if the user also passes valid overrides, an
        empty config_name= is still an error."""
        with pytest.raises(ValueError, match="config_name= must have a non-empty value"):
            parse_config_name_and_overrides(
                ["save_every=500", "config_name=", "mixture=safety"],
                default_config_name="D",
            )

    def test_empty_config_name_error_names_offending_arg(self) -> None:
        """The error message must include the offending arg so users can
        spot it in a long argv."""
        with pytest.raises(ValueError) as exc_info:
            parse_config_name_and_overrides(
                ["config_name="], default_config_name="D"
            )
        assert "'config_name='" in str(exc_info.value)

    @pytest.mark.parametrize(
        "argv,expected_cfg,expected_overrides",
        [
            (["save_every=500"], "chat_sft", ["save_every=500"]),
            (["config_name=c1"], "c1", []),
            (
                ["config_name=c1", "foo=1", "bar=2"],
                "c1",
                ["foo=1", "bar=2"],
            ),
            (["foo"], "chat_sft", []),  # no = → dropped from overrides too
        ],
    )
    def test_parametrized_cases(
        self,
        argv: list[str],
        expected_cfg: str,
        expected_overrides: list[str],
    ) -> None:
        cfg, ov = parse_config_name_and_overrides(argv, default_config_name="chat_sft")
        assert cfg == expected_cfg
        assert ov == expected_overrides


class TestScriptsUseTheHelper:
    """Guard against scripts re-introducing the old hand-rolled parsing
    or forgetting to consume the helper. These are source-string tests,
    not full imports, so they stay cheap."""

    def test_chat_sft_imports_helper(self) -> None:
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent.parent
            / "scripts"
            / "train"
            / "chat_sft.py"
        )
        text = src.read_text()
        assert "parse_config_name_and_overrides" in text, (
            "scripts/chat_sft.py must consume parse_config_name_and_overrides"
        )
        # The old hand-rolled parse should be gone.
        assert "a for a in sys.argv[1:] if" not in text, (
            "scripts/chat_sft.py still has the old hand-rolled overrides parser"
        )

    def test_training_progression_imports_helper(self) -> None:
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent.parent
            / "scripts"
            / "train"
            / "training_progression.py"
        )
        text = src.read_text()
        assert "parse_config_name_and_overrides" in text, (
            "scripts/training_progression.py must consume "
            "parse_config_name_and_overrides"
        )
        # The old hand-rolled parse block in main() should be gone.
        assert (
            "overrides = [a for a in sys.argv[1:] if" not in text
        ), "scripts/training_progression.py still has the old inline parser"

    def test_conditional_encdec_sft_imports_helper(self) -> None:
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent.parent
            / "scripts"
            / "train"
            / "conditional_encdec_sft.py"
        )
        text = src.read_text()
        assert "parse_config_name_and_overrides" in text, (
            "scripts/train/conditional_encdec_sft.py must consume "
            "parse_config_name_and_overrides so 4090/profile configs are selectable"
        )
        assert "_overrides = [a for a in sys.argv[1:]" not in text, (
            "scripts/train/conditional_encdec_sft.py still has the old inline parser"
        )
