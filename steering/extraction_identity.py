"""Producer-owned identity for ordered extraction documents and SAE coordinates."""

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from steering.feature_io import MetadataKey

EXTRACTION_DOCUMENTS_VERSION = "ordered_utf8_length_prefixed_v1"


def extraction_documents_sha256(texts: Iterable[str]) -> str:
    """Bind document boundaries, order and exact UTF-8 text without normalization."""
    digest = hashlib.sha256(EXTRACTION_DOCUMENTS_VERSION.encode() + b"\0")
    for text in texts:
        payload = text.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExtractionSourceIdentity:
    model_id: str
    sae_release: str
    sae_id: str

    def metadata(self) -> dict[str, str]:
        return {
            MetadataKey.EXTRACTION_MODEL_ID: self.model_id,
            MetadataKey.EXTRACTION_SAE_RELEASE: self.sae_release,
            MetadataKey.EXTRACTION_SAE_ID: self.sae_id,
        }
