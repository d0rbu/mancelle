from pathlib import Path


def test_docs_mention_native_backend():
    fe = Path("docs/pipelines/feature-extraction.md").read_text()
    assert "backend=native" in fe, (
        "feature-extraction.md must document the native backend "
        "(not just any 'native' substring)"
    )
    cfg = Path("docs/reference/configuration.md").read_text()
    assert "backend=native" in cfg, (
        "configuration.md must document the native backend"
    )
