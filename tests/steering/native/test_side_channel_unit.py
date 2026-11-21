import torch as t

import steering.extract_features as ef


class _SAE:
    def __init__(self, f=16):
        self._p = t.nn.Parameter(t.zeros(1))
        self.f = f

    def parameters(self):
        yield self._p

    def encode(self, x):
        return t.zeros(x.shape[0], self.f)


def test_single_doc_uses_native_side_channel(monkeypatch):
    called = []

    def fake_native(native_model, tokens, layer, activation_type, hook_name):
        called.append("native")
        return t.zeros(1, tokens.shape[1], 8)

    monkeypatch.setattr(ef, "_get_activations_native", fake_native)

    model = t.nn.Linear(2, 2)
    model.native_model = object()  # side-channel, mirrors model.nnterp_model  # ty: ignore[invalid-assignment]
    sae = _SAE()
    toks = t.tensor([1, 2, 3, 4], dtype=t.long)
    ef._get_sae_activations(model, sae, toks, layer=0,  # ty: ignore[invalid-argument-type]
                            hook_name="blocks.0.hook_resid_post")
    assert called == ["native"]


def test_extract_document_features_routes_through_native(monkeypatch):
    """Edit B routing: model.native_model is threaded into _get_sae_activations_batched."""
    captured = {}

    # Stub tokenization: returns non-empty kept_ids so the "if not kept_ids" guard
    # passes, and n_tokens for the precomputed segment shapes.
    monkeypatch.setattr(
        ef, "_truncate_to_training_window",
        lambda *a, **k: ([1, 2, 3, 4], 4),
    )

    # Stub CPU pre-computation: return one segment covering the full 4-token span.
    # Contract: list[tuple[list[tuple[int,int]], int, list[int], list[int]]]
    # i.e. [(segments, n_tokens, char_starts, char_ends)]
    monkeypatch.setattr(
        ef, "_precompute_batch_cpu_work",
        lambda *a, **k: [([(0, 4)], 4, [0], [4])],
    )

    sentinel = object()

    # Stub the batched forward: capture kwargs, return list of one [seq, n_feat] tensor.
    # extract_document_features does [0] on the return value to get sae_acts.
    def fake_batched(*args, **kwargs):
        captured["native_model"] = kwargs.get("native_model")
        captured["called"] = True
        return [t.zeros(4, 8)]

    monkeypatch.setattr(ef, "_get_sae_activations_batched", fake_batched)

    # Stub pooling: sae_acts [4,8] + one segment → [1,8]
    monkeypatch.setattr(
        ef, "pool_segments_batched_gpu",
        lambda *a, **k: t.zeros(1, 8),
    )

    # Stub sparse extraction: one segment → lists of one empty tensor each
    monkeypatch.setattr(
        ef, "extract_sparse_gpu",
        lambda *a, **k: ([t.empty(0, dtype=t.int32)], [t.empty(0, dtype=t.float32)]),
    )

    class _Cfg:
        n_ctx = 2048
        model_name = "gemma-2-2b"

    class _Tok:
        bos_token_id = 2

    class _Model:
        cfg = _Cfg()
        tokenizer = _Tok()

    model = _Model()
    model.native_model = sentinel  # the side-channel under test  # ty: ignore[unresolved-attribute]
    sae = _SAE()

    ef.extract_document_features(
        model, sae, "hello world", layer=0,
        hook_name="blocks.0.hook_resid_post",
    )
    assert captured.get("called") is True
    assert captured.get("native_model") is sentinel, (
        "extract_document_features must thread model.native_model into "
        "_get_sae_activations_batched"
    )
