# `words_alpha.txt` provenance

This repository vendors the exact bytes used by the self-conditioning SFT
startup path so a training node never needs live access to GitHub for the
standard or capabilities mixture.

- Upstream repository: `dwyl/english-words`
- Upstream commit: `20f5cc9b3f0ccc8ce45d814c532b7c2031bba31c`
- Upstream file: `words_alpha.txt`
- Vendored byte count: `4,234,910`
- Vendored line count: `370,105`
- Vendored SHA-256:
  `3ed0c94610d8bcf7c11bbb49c56aa49c7234d32b66824df91f554169e572da48`

The task code continues to prefer an operator-provided
`$NANOCHAT_BASE_DIR/words_alpha.txt` when present. That preserves existing
artifact families whose task mixture was built from a previously cached copy.
If no cache file exists, it uses this pinned copy. A self-conditioning run
fails with an actionable local-path error if neither is available; it does not
fall back to a live fetch.
