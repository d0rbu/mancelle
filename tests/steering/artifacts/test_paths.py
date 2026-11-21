"""HF storage-bucket URI construction."""

from steering.artifacts.paths import BucketURI


def test_bucket_uri_with_path() -> None:
    assert (
        BucketURI(owner="acme", bucket="bkt", path="checkpoints/base/d12")
        == "hf://buckets/acme/bkt/checkpoints/base/d12"
    )


def test_bucket_uri_without_path_has_no_trailing_slash() -> None:
    assert BucketURI(owner="acme", bucket="bkt") == "hf://buckets/acme/bkt"
