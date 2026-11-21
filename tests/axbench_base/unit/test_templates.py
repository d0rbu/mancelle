import pytest

from steering.eval.axbench_base.paths import PROMPT_DIR
from steering.eval.axbench_base.templates import (
    _safe_substitute,
    load_template,
    render_template_with_untrusted_values,
)
from steering.eval.axbench_base.types import AxBenchBaseError


def test_safe_substitute_replaces_uppercase_placeholders():
    tpl = "Hello {{NAME}}, welcome to {{PLACE}}."
    out = _safe_substitute(tpl, NAME="World", PLACE="Earth")
    assert out == "Hello World, welcome to Earth."


def test_safe_substitute_ignores_single_braces():
    tpl = "Here is code: def foo() { return 1; } and {{VALUE}}."
    out = _safe_substitute(tpl, VALUE="42")
    assert "def foo() { return 1; }" in out
    assert "42" in out
    assert "{{VALUE}}" not in out


def test_safe_substitute_ignores_literal_braces_in_content():
    tpl = "Prefix: {{PREFIX_TEXT}}\nDescription: {{DESCRIPTION}}"
    prefix_with_braces = "Some text {UNDEFINED} and more"
    out = _safe_substitute(tpl, PREFIX_TEXT=prefix_with_braces, DESCRIPTION="a thing")
    assert "{UNDEFINED}" in out
    assert "a thing" in out
    assert "{{PREFIX_TEXT}}" not in out
    assert "{{DESCRIPTION}}" not in out


def test_safe_substitute_leaves_unfilled_placeholders_visible():
    tpl = "{{FILLED}} and {{MISSING}}"
    out = _safe_substitute(tpl, FILLED="ok")
    assert "{{MISSING}}" in out
    assert "ok" in out


def test_safe_substitute_does_not_re_substitute_values_as_placeholders():
    """A substituted value that contains a literal ``{{OTHER_KEY}}``
    must NOT become a placeholder for a subsequent pass.

    The old iterative ``str.replace`` implementation was an injection
    vector: feeding untrusted text (Stage 1 synth passes wild
    validation-shard ``seed.prefix_text`` into ``{{PREFIX_TEXT}}``) whose
    content happened to look like ``{{DESCRIPTION}}`` would silently
    get the concept description substituted into the prefix slot and
    corrupt the prompt. Pin single-pass semantics so the regression
    can't sneak back in.
    """
    tpl = "Prefix: {{PREFIX_TEXT}}\nDescription: {{DESCRIPTION}}"
    out = _safe_substitute(
        tpl,
        PREFIX_TEXT="Markdown about {{DESCRIPTION}} templates",
        DESCRIPTION="concept-X",
    )
    # The literal ``{{DESCRIPTION}}`` inside PREFIX_TEXT must survive
    # unchanged into the output — NOT be replaced with "concept-X".
    assert (
        out
        == "Prefix: Markdown about {{DESCRIPTION}} templates\n"
        "Description: concept-X"
    ), out


def test_safe_substitute_preserves_kwarg_iteration_independence():
    """Substitution order must not affect the output — simultaneous
    replacement means kwargs can be supplied in any order."""
    tpl = "{{A}} {{B}}"
    out_ab = _safe_substitute(tpl, A="{{B}}", B="y")
    out_ba = _safe_substitute(tpl, B="y", A="{{B}}")
    # Both orderings yield the same result because substitution is
    # single-pass; the ``{{B}}`` inside A's value isn't re-processed.
    assert out_ab == out_ba == "{{B}} y", (out_ab, out_ba)


def test_untrusted_renderer_preserves_placeholder_shaped_source_data():
    prompt = render_template_with_untrusted_values(
        "synth_continue_with_concept.txt",
        label="test synth",
        BEGIN_MARKER="[[BEGIN]]",
        END_MARKER="[[END]]",
        ECHO_CHARS=6,
        PREFIX_TEXT="Literal {{DESCRIPTION}} from source data",
        PREFIX_SUFFIX="e data",
        DESCRIPTION="ocean tides",
    )

    assert "Literal {{DESCRIPTION}} from source data" in prompt
    assert "ocean tides" in prompt


def test_untrusted_renderer_rejects_template_drift_before_inserting_data():
    with pytest.raises(AxBenchBaseError, match="test synth.*DESCRIPTION"):
        render_template_with_untrusted_values(
            "synth_continue_with_concept.txt",
            label="test synth",
            BEGIN_MARKER="[[BEGIN]]",
            END_MARKER="[[END]]",
            ECHO_CHARS=6,
            PREFIX_TEXT="Literal {{DESCRIPTION}} from source data",
            PREFIX_SUFFIX="e data",
        )


def test_all_four_templates_exist():
    for name in (
        "synth_continue_with_concept.txt",
        "synth_continue_avoiding_concept.txt",
        "judge_concept_relevance.txt",
        "judge_fluency.txt",
    ):
        path = PROMPT_DIR / name
        assert path.exists(), f"missing template {name} at {path}"
        assert path.read_text(encoding="utf-8").strip(), f"empty template {name}"


def test_synth_positive_template_has_placeholders():
    tpl = load_template("synth_continue_with_concept.txt")
    assert "{{PREFIX_TEXT}}" in tpl
    assert "{{DESCRIPTION}}" in tpl


def test_synth_negative_template_has_placeholders():
    tpl = load_template("synth_continue_avoiding_concept.txt")
    assert "{{PREFIX_TEXT}}" in tpl
    assert "{{DESCRIPTION}}" in tpl


def test_judge_concept_template_has_axbench_format():
    tpl = load_template("judge_concept_relevance.txt")
    assert "{{CONCEPT}}" in tpl
    assert "{{SENTENCE}}" in tpl
    assert "Rating: [[score]]" in tpl


def test_judge_fluency_template_has_axbench_format():
    tpl = load_template("judge_fluency.txt")
    assert "{{SENTENCE}}" in tpl
    assert "Rating: [[score]]" in tpl


def test_load_template_is_cached():
    a = load_template("synth_continue_with_concept.txt")
    b = load_template("synth_continue_with_concept.txt")
    assert a is b, "load_template should return the same string object on repeated calls"


def test_load_template_missing_raises_axbenchbase_error():
    with pytest.raises(AxBenchBaseError, match="template not found"):
        load_template("definitely_not_a_real_template_xyzzy.txt")
