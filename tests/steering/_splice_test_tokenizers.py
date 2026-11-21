"""Controllable test tokenizers for the feature-shard splice round-trip tests.

Deliberately tiny, deterministic, and network-free so the round-trip seam
machinery can be exercised across stable AND merge-unstable seams without a real
model download. Not named like any removed splice-only helper.

``CharMergeTokenizer`` is a per-character tokenizer with an optional BOS and an
optional ordered list of 2-character merges. A merge makes a seam unstable: if
chain A ends with ``"a"`` and chain B starts with ``"b"`` and ``("a","b")`` is a
merge, the concatenated text ``"...ab..."`` re-tokenizes to the merged id, so the
naive concatenation of the two segments' ids does NOT round-trip.

HF-ness: when ``expose_vocab=True`` the tokenizer carries an integer
``vocab_size`` attribute, so the seam checker's window-side inference treats it as
HF-like (decodes each id to find the max byte length). When False it has no
``vocab_size`` / ``get_vocab``, so the checker uses its fixed non-HF window.
"""

from __future__ import annotations


class CharMergeTokenizer:
    def __init__(
        self,
        *,
        bos_id: int | None = None,
        merge_pairs: list[tuple[str, str]] | None = None,
        merge_id_base: int = 1000,
        expose_vocab: bool = False,
    ) -> None:
        self.bos_id = bos_id
        if bos_id is not None:
            self.bos_token_id = bos_id
        pairs = list(merge_pairs or [])
        self.merges: dict[tuple[str, str], int] = {
            pair: merge_id_base + i for i, pair in enumerate(pairs)
        }
        self._merge_id_to_str = {mid: c0 + c1 for (c0, c1), mid in self.merges.items()}
        if expose_vocab:
            # HF-like: ids span [0, merge_id_base + n_merges); the checker decodes
            # each to bound the window. Keep merge_id_base small for fast tests.
            self.vocab_size = merge_id_base + len(pairs)

    def _encode_body(self, text: str) -> list[int]:
        ids: list[int] = []
        i, n = 0, len(text)
        while i < n:
            if i + 1 < n and (text[i], text[i + 1]) in self.merges:
                ids.append(self.merges[(text[i], text[i + 1])])
                i += 2
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def encode(self, text: str) -> list[int]:
        return self._encode_body(text)

    def decode(self, token_ids: list[int]) -> str:
        out: list[str] = []
        for tid in token_ids:
            if self.bos_id is not None and tid == self.bos_id:
                continue  # BOS renders empty -> source_bos_char_len == 0
            if tid in self._merge_id_to_str:
                out.append(self._merge_id_to_str[tid])
            else:
                out.append(chr(tid))
        return "".join(out)

    def __call__(self, texts, *, padding: bool = False, add_special_tokens: bool = True):
        def one(text: str) -> list[int]:
            body = self._encode_body(text)
            if self.bos_id is not None and add_special_tokens:
                return [self.bos_id, *body]
            return body

        if isinstance(texts, str):
            return {"input_ids": one(texts)}
        return {"input_ids": [one(text) for text in texts]}

    def get_bos_token_id(self) -> int | None:
        return self.bos_id


class SpanMergeTokenizer:
    """Per-character tokenizer with configurable *multi-character span* merges.

    Unlike :class:`CharMergeTokenizer` (2-char pair merges), this collapses each
    registered N-character ``span`` (N >= 2) into a single id wherever the span
    appears in text (longest-match, left-to-right). A PROPER SUBSTRING of a span
    does NOT merge: with ``spans=["pqrs"]``, the text ``"pqr"`` tokenizes to
    ``[ord('p'), ord('q'), ord('r')]`` and ``"qrs"`` to three ids, but ``"pqrs"``
    collapses to one merged id. This is exactly the tool for exercising the
    recombination engine's run-local seam blind spot: a span whose trigger straddles
    a whole middle run (``prev_tail + middle_run + next_head``) is invisible to every
    run-local seam window — the mid-run's out-stitch prefix is synthetic-BOS-led (so
    the left context is masked) and its in-stitch misses the right context — yet the
    full-document oracle sees the assembled span and rejects it. That gap is what
    drives Phase E (full-doc redraw) and unbreakable-loop (culprit redraw) recovery.

    A span merge is *dormant* in any source doc that does not literally contain the
    span, so source docs stay round-trip stable while cross-run stitching can
    activate the merge. BOS is optional; ``expose_vocab`` mirrors
    :class:`CharMergeTokenizer` (kept non-HF by default so the seam window is the
    fixed non-HF constant).
    """

    def __init__(
        self,
        *,
        bos_id: int | None = None,
        spans: list[str] | None = None,
        merged_id_base: int = 1000,
        expose_vocab: bool = False,
    ) -> None:
        self.bos_id = bos_id
        if bos_id is not None:
            self.bos_token_id = bos_id
        span_list = list(spans or [])
        for span in span_list:
            if len(span) < 2:
                raise ValueError(f"span must be >= 2 chars, got {span!r}")
        self.spans: dict[str, int] = {
            span: merged_id_base + i for i, span in enumerate(span_list)
        }
        self._id_to_span = {mid: span for span, mid in self.spans.items()}
        # Longest-first so a longer span wins over a shorter prefix-overlapping one.
        self._ordered = sorted(self.spans, key=len, reverse=True)
        if expose_vocab:
            self.vocab_size = merged_id_base + len(span_list)

    def _encode_body(self, text: str) -> list[int]:
        ids: list[int] = []
        i, n = 0, len(text)
        while i < n:
            for span in self._ordered:
                if text.startswith(span, i):
                    ids.append(self.spans[span])
                    i += len(span)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def encode(self, text: str) -> list[int]:
        return self._encode_body(text)

    def decode(self, token_ids: list[int]) -> str:
        out: list[str] = []
        for tid in token_ids:
            if self.bos_id is not None and tid == self.bos_id:
                continue  # BOS renders empty (matches CharMergeTokenizer)
            if tid in self._id_to_span:
                out.append(self._id_to_span[tid])
            else:
                out.append(chr(tid))
        return "".join(out)

    def __call__(self, texts, *, padding: bool = False, add_special_tokens: bool = True):
        def one(text: str) -> list[int]:
            body = self._encode_body(text)
            if self.bos_id is not None and add_special_tokens:
                return [self.bos_id, *body]
            return body

        if isinstance(texts, str):
            return {"input_ids": one(texts)}
        return {"input_ids": [one(text) for text in texts]}

    def get_bos_token_id(self) -> int | None:
        return self.bos_id


_U2581 = "▁"  # SentencePiece space marker


class Gemma3LikeMarkerTokenizer:
    """Faithful synthetic model of the gemma-3 U+2581 source-data case.

    Distinct ids for the literal marker (900) and a real space (901), but BOTH
    decode to a normal space and BOTH render as the ``▁`` piece. So tokenizing a
    source doc that contains a literal ``▁`` yields id 900; decoding gives a
    space; re-encoding that space gives id 901 — a length-preserving, piece-equal
    mismatch that is exactly what the normalization is allowed to repair.

    ``byte_fallback_id`` (default ``None``) activates an optional synthetic
    byte-fallback token. When set, the character ``\\xff`` encodes to that id,
    decoding it yields ``"Z"`` (a char that re-encodes to a *different* id so the
    sequence is round-trip-unstable), and ``convert_ids_to_tokens`` returns
    ``"<0xFF>"`` for it so ``_is_byte_fallback_token`` recognises it. Existing
    behaviour is unchanged when the parameter is omitted.
    """

    name_or_path = "gemma-3-1b-pt-marker-test"
    _MARKER_ID = 900
    _SPACE_ID = 901

    def __init__(self, *, byte_fallback_id: int | None = None) -> None:
        self._byte_fallback_id = byte_fallback_id

    def _encode_body(self, text: str) -> list[int]:
        out: list[int] = []
        for ch in text:
            if ch == _U2581:
                out.append(self._MARKER_ID)
            elif ch == " ":
                out.append(self._SPACE_ID)
            elif self._byte_fallback_id is not None and ch == "\xff":
                out.append(self._byte_fallback_id)
            else:
                out.append(ord(ch))
        return out

    def encode(self, text: str) -> list[int]:
        return self._encode_body(text)

    def decode(self, token_ids: list[int]) -> str:
        out: list[str] = []
        for tid in token_ids:
            if tid in (self._MARKER_ID, self._SPACE_ID):
                out.append(" ")  # marker AND space both decode to a normal space
            elif self._byte_fallback_id is not None and tid == self._byte_fallback_id:
                out.append("Z")  # decodes to "Z"; re-encodes to ord("Z")=90 ≠ id → unstable
            else:
                out.append(chr(tid))
        return "".join(out)

    def convert_ids_to_tokens(self, token_ids):
        """Accept a single int OR a list of ints (both call paths exist in production).

        Returns a single ``str`` for a single int, ``list[str]`` for a list — matching
        the dual contract needed by ``_is_byte_fallback_token`` (single-int probe) and
        ``_token_pieces`` (list probe).
        """
        if isinstance(token_ids, int):
            tid = token_ids
            if self._byte_fallback_id is not None and tid == self._byte_fallback_id:
                return "<0xFF>"
            return _U2581 if tid in (self._MARKER_ID, self._SPACE_ID) else chr(tid)
        return [self.convert_ids_to_tokens(tid) for tid in token_ids]

    def __call__(self, texts, *, padding: bool = False, add_special_tokens: bool = True):
        if isinstance(texts, str):
            return {"input_ids": self._encode_body(texts)}
        return {"input_ids": [self._encode_body(text) for text in texts]}
