"""Round-trip integration: push a tiny tree to a throwaway bucket, pull it back, verify content.

Skipped unless both env vars are set:
  HF_TOKEN — write access to NANO_GPT_HF_TEST_BUCKET's owner namespace
  NANO_GPT_HF_TEST_BUCKET — owner/bucket form, e.g. 'd0rbu/nano-gpt-test-bucket'
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from steering.artifacts.categories import resolve_category
from steering.artifacts.config import ArtifactsConfig, CategoryConfig
from steering.artifacts.sync import sync_pull, sync_push

pytestmark = [pytest.mark.slow, pytest.mark.api]


@pytest.fixture(scope="module")
def test_bucket():
    bucket_spec = os.environ.get("NANO_GPT_HF_TEST_BUCKET")
    if not bucket_spec or not os.environ.get("HF_TOKEN"):
        pytest.skip("set HF_TOKEN and NANO_GPT_HF_TEST_BUCKET to run integration tests")
    owner, name = bucket_spec.split("/", 1)
    return owner, name


@pytest.fixture
def round_trip_cfg(tmp_path: Path, test_bucket):
    owner, name = test_bucket
    local_root = tmp_path / "local_under_test"
    local_root.mkdir()
    return ArtifactsConfig(
        bucket_owner=owner,
        bucket_name=name,
        categories={
            "test_round_trip": CategoryConfig(
                local_root=str(local_root),
                remote_subpath=f"_test_round_trip_{os.getpid()}",
            )
        },
    )


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_round_trip_single_file(tmp_path: Path, round_trip_cfg: ArtifactsConfig):
    local = resolve_category("test_round_trip", round_trip_cfg).local_root

    # Create tiny file locally
    src_file = local / "hello.txt"
    src_file.write_text("hello, hf bucket")

    # Push to bucket
    sync_push(
        cfg=round_trip_cfg, category_name="test_round_trip",
        user_include=None, user_exclude=None,
        no_default_include=False, no_default_exclude=False,
        dry_run=False, delete=False,
    )

    # Wipe local, pull back
    src_file.unlink()
    assert not src_file.exists()
    sync_pull(
        cfg=round_trip_cfg, category_name="test_round_trip",
        user_include=None, user_exclude=None,
        no_default_include=False, no_default_exclude=False,
        dry_run=False,
    )

    assert src_file.exists()
    assert src_file.read_text() == "hello, hf bucket"


def test_filter_excludes_unwanted_file(tmp_path: Path, round_trip_cfg: ArtifactsConfig):
    local = resolve_category("test_round_trip", round_trip_cfg).local_root

    (local / "wanted.safetensors").write_bytes(b"wanted")
    (local / "ignored.tmp").write_bytes(b"ignored")

    # Push with an explicit exclude
    sync_push(
        cfg=round_trip_cfg, category_name="test_round_trip",
        user_include=None, user_exclude=["*.tmp"],
        no_default_include=False, no_default_exclude=False,
        dry_run=False, delete=False,
    )

    # Wipe local, pull
    for p in local.iterdir():
        p.unlink()
    sync_pull(
        cfg=round_trip_cfg, category_name="test_round_trip",
        user_include=None, user_exclude=None,
        no_default_include=False, no_default_exclude=False,
        dry_run=False,
    )

    assert (local / "wanted.safetensors").exists()
    assert not (local / "ignored.tmp").exists()
