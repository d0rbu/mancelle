"""False positives whose names are required by externally visible contracts."""

# ``prepend`` is a keyword-only part of two tokenizer Protocol signatures.
# Their ellipsis bodies cannot reference it, but callers pass it by name.
prepend  # noqa: B018, F821  # ty: ignore[unresolved-reference]
