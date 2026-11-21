"""Tests for the SeedInput widget."""

from __future__ import annotations

from scripts.completion_tui.widgets.seed_input import SeedInput
from tests.test_completion_tui._key_dispatch import dispatch_key


class TestSeedInput:
    def test_initial_seed(self) -> None:
        si = SeedInput(seed=42)
        assert si.seed == 42

    def test_first_digit_overwrites(self) -> None:
        si = SeedInput(seed=42)
        si.handle_digit(7)
        assert si.seed == 7

    def test_subsequent_digits_accumulate(self) -> None:
        si = SeedInput(seed=42)
        si.handle_digit(1)
        si.handle_digit(2)
        si.handle_digit(3)
        assert si.seed == 123

    def test_setting_seed_resets_overwrite(self) -> None:
        si = SeedInput(seed=42)
        si.handle_digit(1)
        si.handle_digit(2)
        assert si.seed == 12
        # Setting seed externally resets first_digit flag
        si.seed = 99
        si.handle_digit(5)
        assert si.seed == 5  # overwrites, not appends to 99

    def test_escape_posts_exited(self) -> None:
        si = SeedInput(seed=42)
        msgs = dispatch_key(si, "escape")
        assert len(msgs) == 1
        assert isinstance(msgs[0], SeedInput.SeedInputExited)

    def test_ctrl_left_posts_exited(self) -> None:
        si = SeedInput(seed=42)
        msgs = dispatch_key(si, "ctrl+left")
        assert len(msgs) == 1
        assert isinstance(msgs[0], SeedInput.SeedInputExited)

    def test_enter_posts_exited(self) -> None:
        si = SeedInput(seed=42)
        msgs = dispatch_key(si, "enter")
        assert len(msgs) == 1
        assert isinstance(msgs[0], SeedInput.SeedInputExited)

    def test_non_exit_key_returns_nothing(self) -> None:
        si = SeedInput(seed=42)
        msgs = dispatch_key(si, "left")
        assert msgs == []

    def test_reset_input_makes_next_digit_overwrite(self) -> None:
        si = SeedInput(seed=42)
        si.handle_digit(1)
        si.handle_digit(2)
        assert si.seed == 12
        si.reset_input()
        si.handle_digit(5)
        assert si.seed == 5

    def test_zero_seed(self) -> None:
        si = SeedInput(seed=0)
        assert si.seed == 0
        si.handle_digit(0)
        assert si.seed == 0

    def test_max_digits_stops_input(self) -> None:
        si = SeedInput(seed=0)
        for d in [1, 2, 3, 4, 5, 6]:
            si.handle_digit(d)
        assert si.seed == 123456
        si.handle_digit(7)  # should be ignored
        assert si.seed == 123456

    def test_backspace_removes_last_digit(self) -> None:
        si = SeedInput(seed=0)
        si.handle_digit(1)
        si.handle_digit(2)
        si.handle_digit(3)
        assert si.seed == 123
        si.handle_backspace()
        assert si.seed == 12
        si.handle_backspace()
        assert si.seed == 1

    def test_backspace_single_digit_goes_to_zero(self) -> None:
        si = SeedInput(seed=0)
        si.handle_digit(5)
        assert si.seed == 5
        si.handle_backspace()
        assert si.seed == 0

    def test_backspace_then_digit_appends(self) -> None:
        si = SeedInput(seed=0)
        si.handle_digit(1)
        si.handle_digit(2)
        si.handle_backspace()
        assert si.seed == 1
        si.handle_digit(9)
        assert si.seed == 19
