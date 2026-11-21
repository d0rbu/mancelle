"""Cooldown at real live/packed loader boundaries, independent of prefetch."""

from __future__ import annotations

import copy
import json
from itertools import islice
from pathlib import Path

import pytest
import torch as t

import nanochat.flash_attention as attention
from nanochat.dataset import DatasetSplit
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder import dataloader as dl
from steering.conditional.encoder_decoder.conditioning_cooldown import apply_conditioning_cooldown
from steering.conditional.encoder_decoder.conditioning_splice import ConditioningSpliceIndex
from steering.conditional.encoder_decoder.dataloader import SegFeatures
from steering.conditional.splice_provenance import SpliceProvenanceError
from steering.gradient_overlay import provenance_dir
from steering.overlay_io import load_overlay_shard, save_overlay_shard
from steering.packed_batch_io import load_packed_row_shard, save_packed_row_shard
from steering.refined_types import PositiveInt
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    _make_test_shard,
    build_model_with_feature_prediction,
    build_threshold_pipeline,
)
from tests.steering.test_conditioning_cooldown_provenance import (
    _expected_flags,
)
from tests.steering.test_conditioning_cooldown_provenance import (
    _loader as _spliced_loader,
)
from tests.steering.test_conditioning_cooldown_provenance import (
    spliced_packed as spliced_packed,
)


def _pipeline():
    boundaries = (
        t.linspace(0.25, 0.75, NUM_BUCKETS - 1, dtype=t.float64)
        .unsqueeze(0)
        .expand(N_SAE_FEATURES, -1)
        .contiguous()
    )
    return build_threshold_pipeline(boundaries)


def _packed(tmp_path: Path) -> Path:
    packed = tmp_path / "packed"
    packed.mkdir()
    for index, counts in enumerate(([2, 4, 3], [5, 2, 4])):
        shard = _make_test_shard(n_rows=3, segs_per_row=counts, seed=71 + index)
        offset = 0
        for count in counts:
            shard.seg_doc_id[offset : offset + count] = (
                t.arange(count).div(2, rounding_mode="floor").to(t.uint8)
            )
            offset += count
        save_packed_row_shard(shard, packed / f"shard_{index:05d}")
    (packed / "manifest.json").write_text(json.dumps({"metadata": {}}))
    return packed


def _loader(packed: Path, resume=None):
    return dl.precomputed_data_loader(
        packed_row_dir=packed,
        B=2,
        feature_pipeline=_pipeline(),
        n_sae_features=N_SAE_FEATURES,
        device="cpu",
        label_dropout=0.0,
        resume_state_dict=resume,
    )


def _masked(batch, *, step: int, micro_step: int, granularity: str, fraction: float):
    return apply_conditioning_cooldown(
        batch[4],
        batch[7],
        fraction=fraction,
        granularity=granularity,
        seed=727,
        step=step,
        rank=0,
        micro_step=micro_step,
    )


def _assert_batch_equal(actual, expected):
    for left, right in zip(actual[:9], expected[:9], strict=True):
        if right is None:
            assert left is None
        else:
            assert t.equal(left, right)
    assert actual[9] == expected[9]


@pytest.mark.parametrize("artifact", ["sidecar", "provenance"])
def test_lazy_join_rejects_replaced_inputs_after_recording_identity(
    spliced_packed, artifact: str
) -> None:
    env, packed, paths = spliced_packed
    feature_dir = Path(env["feature_dir"])
    stems = json.loads((packed / "manifest.json").read_text())["metadata"]["base_shard_stems"]
    sidecars = [
        json.loads((feature_dir / "_splice_provenance" / f"{stem}.json").read_text())
        for stem in stems
    ]
    stale = ConditioningSpliceIndex(
        packed, feature_dir, packed_shard_names=[path.name for path in paths]
    )
    # Find a valid replacement that changes actual flags. The lifetime check
    # must catch it even though each new file remains individually well formed.
    for path in paths:
        base = load_packed_row_shard(path)
        prov_path = provenance_dir(packed) / path.name
        prov = load_overlay_shard(prov_path)
        for row in range(base.n_rows):
            expected = _expected_flags(feature_dir, packed, path.name, base, row)
            start, end = base.row_seg_offsets[row : row + 2].tolist()
            replaced = False
            if artifact == "sidecar" and any(expected):
                segment = start + expected.index(True)
                source = int(prov["base_shard_idx"][segment])
                raw = sidecars[source]
                doc = raw["docs"][int(prov["doc_idx"][segment])]
                ordinal = int(prov["seg_ordinal"][segment])
                doc["splices"] = [item for item in doc["splices"] if item["pos"] != ordinal]
                raw["params"]["n_splice_positions"] -= 1
                (feature_dir / "_splice_provenance" / f"{stems[source]}.json").write_text(
                    json.dumps(raw)
                )
                replaced = True
            elif artifact == "provenance":
                for packed_doc in base.seg_doc_id[start:end].unique().tolist():
                    indices = (base.seg_doc_id[start:end] == packed_doc).nonzero().flatten()
                    old_flags = [expected[int(i)] for i in indices]
                    for source, raw in enumerate(sidecars):
                        for doc in raw["docs"]:
                            positions = {item["pos"] for item in doc["splices"]}
                            new_flags = [ordinal in positions for ordinal in range(len(indices))]
                            if doc["n_segments"] < len(indices) or old_flags == new_flags:
                                continue
                            prov["base_shard_idx"][indices + start] = source
                            prov["doc_idx"][indices + start] = doc["doc_id"]
                            save_overlay_shard(prov, prov_path)
                            replaced = True
                            break
                        if replaced:
                            break
                    if replaced:
                        break
            if replaced:
                assert _expected_flags(feature_dir, packed, path.name, base, row) != expected
                with pytest.raises(SpliceProvenanceError, match="changed"):
                    stale.row_flags(path.name, base, row)
                return
    pytest.fail("fixture did not contain a semantically different valid replacement")


@pytest.mark.parametrize("granularity", ["segment", "document"])
def test_real_spliced_prefetch_and_checkpoint_replay_preserve_priority(
    spliced_packed, granularity: str
) -> None:
    env, packed, _paths = spliced_packed
    eager = list(islice(_spliced_loader(env, packed), 6))
    prefetched = dl._prefetch_wrapper(islice(_spliced_loader(env, packed), 6), 3, use_cuda=False)
    masks = []
    for index, (batch, expected) in enumerate(zip(prefetched, eager, strict=True)):
        _assert_batch_equal(batch, expected)
        pristine = copy.deepcopy(batch)
        rng = t.get_rng_state().clone()
        masks.append(
            apply_conditioning_cooldown(
                batch[4],
                batch[7],
                fraction=0.75 if index < 2 else 0.25,
                granularity=granularity,
                seed=727,
                step=index // 2,
                rank=0,
                micro_step=index % 2,
                splice_mask=t.tensor(batch[9]["conditioning_splice_mask"]),
            )
        )
        assert t.equal(t.get_rng_state(), rng)
        _assert_batch_equal(batch, pristine)
    # The loader state denotes its pending batch, including a boundary where
    # the B=4 batch spans two epochs of this six-row corpus.
    pending = eager[1]
    assert len({row[0] for row in pending[9]["batch_rows"]}) == 2
    resumed = _spliced_loader(env, packed, resume=json.loads(json.dumps(pending[9])))
    for index, expected in enumerate(eager[1:], start=1):
        batch = next(resumed)
        _assert_batch_equal(batch, expected)
        mask = apply_conditioning_cooldown(
            batch[4],
            batch[7],
            fraction=0.75 if index < 2 else 0.25,
            granularity=granularity,
            seed=727,
            step=index // 2,
            rank=0,
            micro_step=index % 2,
            splice_mask=t.tensor(batch[9]["conditioning_splice_mask"]),
        )
        assert t.equal(mask, masks[index])


@pytest.mark.parametrize("granularity", ["segment", "document"])
def test_prefetch_and_checkpoint_replay_preserve_masks_and_every_payload(
    tmp_path: Path, granularity: str
) -> None:
    packed = _packed(tmp_path)
    eager = list(islice(_loader(packed), 8))
    # The finite iterator makes the real daemon producer exit normally after
    # crossing a shard and epoch; there is no orphan producer in the test.
    prefetched = dl._prefetch_wrapper(islice(_loader(packed), 8), 3, use_cuda=False)
    observed_masks = []
    for index, (batch, expected) in enumerate(zip(prefetched, eager, strict=True)):
        _assert_batch_equal(batch, expected)
        pristine = copy.deepcopy(batch)
        fraction = (1.0, 0.75, 0.5, 0.0)[index // 2]
        rng_before = t.get_rng_state().clone()
        actual_mask = _masked(
            batch,
            step=index // 2,
            micro_step=index % 2,
            granularity=granularity,
            fraction=fraction,
        )
        assert t.equal(t.get_rng_state(), rng_before)
        _assert_batch_equal(batch, pristine)
        expected_mask = _masked(
            expected,
            step=index // 2,
            micro_step=index % 2,
            granularity=granularity,
            fraction=fraction,
        )
        assert t.equal(actual_mask, expected_mask)
        observed_masks.append(actual_mask)
        t.rand(41 + index)  # unrelated evaluation/model RNG cannot affect masks

    assert eager[1][9]["batch_rows"][0][1] != eager[1][9]["batch_rows"][1][1]
    checkpoint = json.loads(json.dumps(eager[4][9]))
    resumed = _loader(packed, checkpoint)
    for index in range(4, 8):
        batch = next(resumed)
        _assert_batch_equal(batch, eager[index])
        actual_mask = _masked(
            batch,
            step=index // 2,
            micro_step=index % 2,
            granularity=granularity,
            fraction=(1.0, 0.75, 0.5, 0.0)[index // 2],
        )
        assert t.equal(actual_mask, observed_masks[index])
    assert not observed_masks[-1].any()


def _segments(length: int):
    return [
        SegFeatures(
            feature_ids=t.tensor([1 + start, 2 + start]),
            buckets=t.tensor([0, 1]),
            percentiles=t.tensor([0.3, 0.7]),
            char_start=start,
            char_end=start + 2,
            token_start=start,
            token_end=start + 2,
        )
        for start in range(0, length, 2)
    ]


def test_live_document_dropout_groups_actual_packed_and_cropped_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def documents(*_args, **_kwargs):
        while True:
            yield (
                [([2] * 4, _segments(4)), ([3] * 12, _segments(12))],
                {"pq_idx": 0, "rg_idx": 0, "epoch": 1, "ddp_world_size": 1},
            )

    monkeypatch.setattr(dl, "_document_batches_with_features", documents)
    loader = dl._encoder_decoder_data_loader_inner(
        tokenizer=None,
        B=1,
        T=10,
        split=DatasetSplit.TRAIN,
        feature_index=None,
        pipeline=_pipeline(),
        segment_size=PositiveInt.parse(2),
        decoder_segment_size=None,
        device="cpu",
        resume_state_dict=None,
        tokenizer_batch_size=2,
        buffer_size=1,
        data_dir=None,
        prefetch_batches=0,
        label_dropout=0.0,
        n_sae_features=N_SAE_FEATURES,
    )
    batch = next(loader)
    assert batch[0].tolist() == [[2, 2, 2, 2, 3, 3, 3, 3, 3, 3]]
    assert batch[7].tolist() == [[0, 0, 1, 1, 1]]
    assert batch[6].tolist() == [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4]]
    pristine = copy.deepcopy(batch)
    combinations = set()
    for step in range(32):
        mask = _masked(batch, step=step, micro_step=0, granularity="document", fraction=0.5)
        assert t.equal(mask[0, 0], mask[0, 1])
        assert t.equal(mask[0, 2], mask[0, 3])
        assert t.equal(mask[0, 3], mask[0, 4])
        combinations.add((bool(mask[0, 0].any()), bool(mask[0, 2].any())))
    assert combinations == {(False, False), (False, True), (True, False), (True, True)}
    _assert_batch_equal(batch, pristine)
    assert batch[8] is not None and batch[8].count_nonzero() == 10


def _live_batch_with_junctions(
    monkeypatch: pytest.MonkeyPatch, *, junction: int, next_offset: bool
):
    long_segments = _segments(12)
    long_segments[junction].is_splice = True

    def documents(*_args, **_kwargs):
        while True:
            yield (
                [([2] * 4, _segments(4)), ([3] * 12, long_segments)],
                {"pq_idx": 0, "rg_idx": 0, "epoch": 1, "ddp_world_size": 1},
            )

    monkeypatch.setattr(dl, "_document_batches_with_features", documents)
    loader = dl._encoder_decoder_data_loader_inner(
        tokenizer=None,
        B=1,
        T=10,
        split=DatasetSplit.TRAIN,
        feature_index=None,
        pipeline=_pipeline(),
        segment_size=PositiveInt.parse(2),
        decoder_segment_size=None,
        device="cpu",
        resume_state_dict=None,
        tokenizer_batch_size=2,
        buffer_size=1,
        data_dir=None,
        prefetch_batches=0,
        label_dropout=0.0,
        n_sae_features=N_SAE_FEATURES,
        next_segment_conditioning_offset=next_offset,
        include_splice_mask=True,
    )
    return next(loader)


def test_cropped_away_junction_does_not_protect_visible_document_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _live_batch_with_junctions(monkeypatch, junction=5, next_offset=False)
    assert batch[7].tolist() == [[0, 0, 1, 1, 1]]
    flags = t.tensor(batch[9]["conditioning_splice_mask"])
    assert flags.tolist() == [[False, False, False, False, False]]
    expected = _masked(batch, step=4, micro_step=1, granularity="document", fraction=0.3)
    actual = apply_conditioning_cooldown(
        batch[4],
        batch[7],
        fraction=0.3,
        granularity="document",
        seed=727,
        step=4,
        rank=0,
        micro_step=1,
        splice_mask=flags,
    )
    assert t.equal(actual, expected)


def test_live_next_segment_priority_tracks_conditioning_source_without_rerouting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plain = _live_batch_with_junctions(monkeypatch, junction=3, next_offset=False)
    shifted = _live_batch_with_junctions(monkeypatch, junction=3, next_offset=True)
    # The long document's fourth extracted segment is beyond the represented
    # decoder prefix. Offset conditioning exposes its features only in the last
    # represented source slot; protect that slot and keep routing unchanged.
    assert plain[9]["conditioning_splice_mask"] == [[False, False, False, False, False]]
    assert shifted[9]["conditioning_splice_mask"] == [[False, False, False, False, True]]
    assert t.equal(plain[6], shifted[6])
    assert t.equal(plain[7], shifted[7])
    assert shifted[2][0, -1].tolist() == [7, 8]
    assert plain[2][0, -1].tolist() == [5, 6]
    flags = t.tensor(shifted[9]["conditioning_splice_mask"])
    actual = apply_conditioning_cooldown(
        shifted[4],
        shifted[7],
        fraction=0.5,
        granularity="document",
        seed=727,
        step=4,
        rank=0,
        micro_step=1,
        splice_mask=flags,
    )
    assert t.equal(actual[0, 2:], shifted[4][0, 2:])


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_packed_batch_forward_backward_preserves_feature_prediction_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fraction: float
) -> None:
    monkeypatch.setattr(attention, "_override_impl", FlashAttnImpl.SDPA)
    batch = next(_loader(_packed(tmp_path)))
    x, y, ids, mags, original_mask, _pcts, routing, documents, targets, _state = batch
    pristine_targets = targets.clone()
    mask = _masked(batch, step=4, micro_step=1, granularity="segment", fraction=fraction)
    model = build_model_with_feature_prediction(n_layer=2, xattn_frequency=1)
    result = model(x, y, ids, mags, mask, routing, seg_doc_id=documents, fp_targets=targets)
    assert t.isfinite(result.loss)
    assert t.isfinite(result.feature_pred_loss) and result.feature_pred_loss > 0
    result.loss.backward()
    assert model.feature_pred_head is not None
    assert model.feature_pred_head.weight.grad is not None
    assert model.feature_pred_head.weight.grad.count_nonzero()
    assert all(
        parameter.grad is None or t.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert t.equal(targets, pristine_targets)
    assert original_mask.any()
    if fraction == 0.0:
        assert not mask.any(), "legacy fallback may not reinstate conditioning"
        with t.no_grad():
            changed_ids = (ids + 17) % N_SAE_FEATURES
            repeated = model(
                x,
                y,
                changed_ids,
                mags,
                mask,
                routing,
                seg_doc_id=documents,
                fp_targets=targets,
            )
        assert t.equal(repeated.loss, result.loss)


@pytest.mark.parametrize("granularity", ["segment", "document"])
@pytest.mark.parametrize("fraction", [0.1, 0.3, 0.7])
def test_all_original_and_all_protected_match_scalar_threshold_at_float_boundary(
    monkeypatch: pytest.MonkeyPatch, granularity: str, fraction: float
) -> None:
    # Thirteen protected units at q=.7 exposes an extra-rounding regression:
    # float32 (q * 13) / 13 can move the threshold by one representable value.
    mask = t.ones(1, 13, 2, dtype=t.bool)
    documents = t.arange(13).unsqueeze(0)
    threshold = t.tensor(1.0 - fraction)
    draws = t.stack(
        [
            t.nextafter(threshold, t.tensor(float("-inf"))),
            threshold,
            t.nextafter(threshold, t.tensor(float("inf"))),
            *([threshold] * 10),
        ]
    ).unsqueeze(0)

    def fixed_random(shape, **kwargs):
        assert tuple(shape) == (1, 13)
        assert isinstance(kwargs["generator"], t.Generator)
        return draws.clone()

    monkeypatch.setattr(t, "rand", fixed_random)
    expected = mask & ~(draws < (1.0 - fraction)).unsqueeze(-1)
    for splice_mask in (
        None,
        t.zeros_like(documents, dtype=t.bool),
        t.ones_like(documents, dtype=t.bool),
    ):
        actual = apply_conditioning_cooldown(
            mask,
            documents,
            fraction=fraction,
            granularity=granularity,
            seed=727,
            step=9,
            rank=0,
            micro_step=1,
            splice_mask=splice_mask,
        )
        assert t.equal(actual, expected)


@pytest.mark.parametrize("granularity", ["segment", "document"])
def test_priority_counts_empty_real_units_and_excludes_padding(
    monkeypatch: pytest.MonkeyPatch, granularity: str
) -> None:
    if granularity == "segment":
        documents = t.tensor([[0, 1, 2, 3, -1, -1]])
        mask = t.tensor([[[True], [False], [True], [True], [False], [False]]])
        expected = t.tensor([[[True], [False], [True], [False], [False], [False]]])
        draws = t.tensor([[0.0, 0.0, 0.7, 0.62, 0.0, 0.0]])
    else:
        documents = t.tensor([[0, 1, 2, 2, 3, 3, -1, -1]])
        mask = t.tensor([[[True], [False], [True], [True], [True], [True], [False], [False]]])
        expected = t.tensor([[[True], [False], [True], [True], [False], [False], [False], [False]]])
        draws = t.tensor([[0.0, 0.0, 0.7, 0.62, 0.0, 0.0, 0.0, 0.0]])
    junctions = t.zeros_like(documents, dtype=t.bool)
    junctions[0, 0] = True

    def fixed_random(shape, **kwargs):
        assert tuple(shape) == tuple(draws.shape)
        assert isinstance(kwargs["generator"], t.Generator)
        return draws.clone()

    monkeypatch.setattr(t, "rand", fixed_random)
    actual = apply_conditioning_cooldown(
        mask,
        documents,
        fraction=0.5,
        granularity=granularity,
        seed=727,
        step=9,
        rank=0,
        micro_step=1,
        splice_mask=junctions,
    )
    # Four real units, including the empty one: protect one and retain each
    # ordinary unit with probability 1/3. Treating the empty slot as padding or
    # the padding as real changes the witness draws at .7 and .62, respectively.
    assert t.equal(actual, expected)
