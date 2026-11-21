"""
Tests for encoder-decoder SFT pipeline tasks.
Starts with PKU-SafeRLHF-10K safety alignment task.
"""

import random
import tempfile
from typing import Any, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

from errors import DataLoadError
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.gpt import GPTConfig
from scripts.train.conditional_encdec_sft import (
    SFTState,
    make_sft_data_generator,
)
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.dataloader import _pad_segments
from steering.conditional.encoder_decoder.full_sequence_routing import (
    build_full_sequence_segment_routing,
    source_aligned_char_offsets,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.sft_features import load_sft_features
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.refined_types import NonNegativeInt, PositiveInt
from steering.segmentation import SegmentMode, segment_tokens, token_to_char_offsets
from steering.sft_schema import SFTSplit
from tasks.alpaca import Alpaca
from tasks.code_alpaca import CodeAlpaca
from tasks.gsm8k import GSM8K
from tasks.mixtures import STEERED_MIXTURES, SFTMixture, build_task_mixtures
from tasks.pku_saferlhf import PKUSafeRLHF
from tasks.steered import SteeredTaskMixture
from tests.conftest import get_tokenizer_or_skip
from tests.steering.test_encdec_sft_conditioned_integration import (
    build_synthetic_features,
    patch_load_sft_features,
)


def _reset_dynamo_for_muon_tests() -> None:
    t._dynamo.reset()
    t._dynamo.config.cache_size_limit = 128


class _NonEmptyRegistryTask:
    """Minimal task-like object for registry factory plumbing tests."""

    def __len__(self) -> int:
        return 1


class _TinyConversationTask:
    """Minimal task-like wrapper for synthetic SFT generator tests."""

    def __init__(self, conversation: dict[str, Any]) -> None:
        self.conversation = conversation

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        assert index == 0
        return self.conversation


def _make_conditioned_sft_generator(
    *,
    task: Any,
    tokenizer: Any,
    B: int,
    T: int,
    device: str,
    sft_state: SFTState,
    min_loss_tokens: int = 8,
):
    features = build_synthetic_features(
        task,
        tokenizer,
        n_sae_features=1000,
    )
    boundaries = t.linspace(0.5, 9.5, 9).unsqueeze(0).expand(1000, -1).contiguous()
    threshold_cfg = ThresholdConfig(
        threshold_type=ThresholdType.segment_topk,
        threshold=1000,
    )

    # ``make_sft_data_generator`` calls ``load_sft_features`` exactly once,
    # eagerly, before its first yield. Prime the generator under the patch and
    # release the patch BEFORE returning. Holding the patch across ``yield from``
    # (the previous shape) leaked it into later tests whenever the returned
    # generator was created but not exhausted: the suspended ``with`` only runs
    # its ``__exit__`` on garbage collection, which is nondeterministic in a full
    # suite run and left ``conditional_encdec_sft.load_sft_features`` mocked for
    # whichever test ran next.
    with patch_load_sft_features(features):
        inner = make_sft_data_generator(
            task=task,
            tokenizer=tokenizer,
            B=B,
            T=T,
            device=device,
            sft_state=sft_state,
            feature_dir="/dummy/features",
            parquet_dir="/dummy/parquet",
            threshold_cfg=threshold_cfg,
            boundaries=boundaries,
            min_loss_tokens=min_loss_tokens,
            alignment_probe_samples=0,
            expected_split=SFTSplit.TRAIN,
        )
        primed: list[Any] = []
        try:
            primed.append(next(inner))
        except StopIteration:
            inner = None

    def _gen():
        yield from primed
        if inner is not None:
            yield from inner

    return _gen()


class TestPKUSafeRLHF:
    """Tests for the PKUSafeRLHF task class."""

    @pytest.fixture(scope="class")
    def task(self) -> PKUSafeRLHF:
        """Load the PKUSafeRLHF task once for all tests in this class."""
        return PKUSafeRLHF(split="train")

    def test_num_examples_positive(self, task: PKUSafeRLHF) -> None:
        """Verify that the dataset contains a positive number of examples."""
        n = task.num_examples()
        assert isinstance(n, int), f"num_examples() should return int, got {type(n)}"
        assert n > 0, f"num_examples() should be positive, got {n}"

    def test_get_example_format(self, task: PKUSafeRLHF) -> None:
        """Verify that each example has the correct conversation format:
        exactly 2 messages, correct roles, string content, non-empty."""
        example = task.get_example(0)
        assert isinstance(example, dict), "Example must be a dict"
        assert "messages" in example, "Example must have 'messages' key"
        messages = example["messages"]
        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"

        user_msg, assistant_msg = messages
        assert user_msg["role"] == "user", (
            f"First message role should be 'user', got {user_msg['role']}"
        )
        assert assistant_msg["role"] == "assistant", (
            f"Second message role should be 'assistant', got {assistant_msg['role']}"
        )
        assert isinstance(user_msg["content"], str), "User content must be a string"
        assert isinstance(assistant_msg["content"], str), "Assistant content must be a string"
        assert len(user_msg["content"]) > 0, "User content must be non-empty"
        assert len(assistant_msg["content"]) > 0, "Assistant content must be non-empty"

    def test_visual_inspection(self, task: PKUSafeRLHF) -> None:
        """Print 5 random samples for human inspection."""
        rng = random.Random(42)
        n = task.num_examples()
        indices = rng.sample(range(n), k=5)
        for i, idx in enumerate(indices):
            example = task.get_example(idx)
            messages = example["messages"]
            print(f"\n--- Sample {i + 1} (index={idx}) ---")
            print(f"USER: {messages[0]['content'][:200]}")
            print(f"ASSISTANT: {messages[1]['content'][:200]}")


class TestAlpaca:
    """Tests for the Alpaca class (yahma/alpaca-cleaned)."""

    @pytest.fixture(scope="class")
    def task(self) -> Alpaca:
        """Load the Alpaca once for all tests in this class."""
        return Alpaca()

    def test_num_examples_positive(self, task: Alpaca) -> None:
        """Verify that the dataset has more than 50000 examples (~51K expected)."""
        n = task.num_examples()
        assert isinstance(n, int), f"num_examples() should return int, got {type(n)}"
        assert n > 50000, f"num_examples() should be > 50000, got {n}"

    def test_get_example_format(self, task: Alpaca) -> None:
        """Verify that each example has the correct conversation format:
        exactly 2 messages, correct roles, string content, non-empty."""
        example = task.get_example(0)
        assert isinstance(example, dict), "Example must be a dict"
        assert "messages" in example, "Example must have 'messages' key"
        messages = example["messages"]
        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"

        user_msg, assistant_msg = messages
        assert user_msg["role"] == "user", (
            f"First message role should be 'user', got {user_msg['role']}"
        )
        assert assistant_msg["role"] == "assistant", (
            f"Second message role should be 'assistant', got {assistant_msg['role']}"
        )
        assert isinstance(user_msg["content"], str), "User content must be a string"
        assert isinstance(assistant_msg["content"], str), "Assistant content must be a string"
        assert len(user_msg["content"]) > 0, "User content must be non-empty"
        assert len(assistant_msg["content"]) > 0, "Assistant content must be non-empty"

    def test_visual_inspection(self, task: Alpaca) -> None:
        """Print 5 random samples for human inspection."""
        rng = random.Random(42)
        n = task.num_examples()
        indices = rng.sample(range(n), k=5)
        for i, idx in enumerate(indices):
            example = task.get_example(idx)
            messages = example["messages"]
            print(f"\n--- Alpaca Sample {i + 1} (index={idx}) ---")
            print(f"USER: {messages[0]['content'][:200]}")
            print(f"ASSISTANT: {messages[1]['content'][:200]}")


class TestCodeAlpaca:
    """Tests for the CodeAlpaca task class (sahil2801/CodeAlpaca-20k)."""

    @pytest.fixture(scope="class")
    def task(self) -> CodeAlpaca:
        """Load the CodeAlpaca task once for all tests in this class."""
        return CodeAlpaca()

    def test_num_examples_positive(self, task: CodeAlpaca) -> None:
        """Verify that the dataset has more than 19000 examples (~20K expected)."""
        n = task.num_examples()
        assert isinstance(n, int), f"num_examples() should return int, got {type(n)}"
        assert n > 19000, f"num_examples() should be > 19000, got {n}"

    def test_get_example_format(self, task: CodeAlpaca) -> None:
        """Verify that each example has the correct conversation format:
        exactly 2 messages, correct roles, string content, non-empty."""
        example = task.get_example(0)
        assert isinstance(example, dict), "Example must be a dict"
        assert "messages" in example, "Example must have 'messages' key"
        messages = example["messages"]
        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"

        user_msg, assistant_msg = messages
        assert user_msg["role"] == "user", (
            f"First message role should be 'user', got {user_msg['role']}"
        )
        assert assistant_msg["role"] == "assistant", (
            f"Second message role should be 'assistant', got {assistant_msg['role']}"
        )
        assert isinstance(user_msg["content"], str), "User content must be a string"
        assert isinstance(assistant_msg["content"], str), "Assistant content must be a string"
        assert len(user_msg["content"]) > 0, "User content must be non-empty"
        assert len(assistant_msg["content"]) > 0, "Assistant content must be non-empty"

    def test_visual_inspection(self, task: CodeAlpaca) -> None:
        """Print 5 random samples for human inspection."""
        rng = random.Random(42)
        n = task.num_examples()
        indices = rng.sample(range(n), k=5)
        for i, idx in enumerate(indices):
            example = task.get_example(idx)
            messages = example["messages"]
            print(f"\n--- CodeAlpaca Sample {i + 1} (index={idx}) ---")
            print(f"USER: {messages[0]['content'][:300]}")
            print(f"ASSISTANT: {messages[1]['content'][:300]}")


class TestOffsetSegmentation:
    """Tests for the token_offset parameter in segment_tokens()."""

    def test_fixed_offset_zero_is_default(self) -> None:
        """Offset=0 should produce identical results to no offset."""
        token_ids = list(range(200))
        segs_default = segment_tokens(
            token_ids,
            SegmentMode.fixed_tokens,
            segment_size=PositiveInt.parse(128),
        )
        segs_offset0 = segment_tokens(
            token_ids,
            SegmentMode.fixed_tokens,
            segment_size=PositiveInt.parse(128),
            token_offset=NonNegativeInt.parse(0),
        )
        assert segs_default == segs_offset0

    def test_fixed_offset_skips_prefix(self) -> None:
        """Offset=50 on 200 tokens with segment_size=128 should produce segments [(50,178), (178,200)]."""
        token_ids = list(range(200))
        segs = segment_tokens(
            token_ids,
            SegmentMode.fixed_tokens,
            segment_size=PositiveInt.parse(128),
            token_offset=NonNegativeInt.parse(50),
        )
        assert segs == [(50, 178), (178, 200)]

    def test_fixed_offset_at_end(self) -> None:
        """Offset at last token should produce one segment of size 1."""
        token_ids = list(range(100))
        segs = segment_tokens(
            token_ids,
            SegmentMode.fixed_tokens,
            segment_size=PositiveInt.parse(128),
            token_offset=NonNegativeInt.parse(99),
        )
        assert segs == [(99, 100)]

    def test_fixed_offset_equals_length_raises(self) -> None:
        """Offset >= len(tokens) should raise AssertionError."""
        token_ids = list(range(100))
        with pytest.raises(AssertionError):
            segment_tokens(
                token_ids,
                SegmentMode.fixed_tokens,
                segment_size=PositiveInt.parse(128),
                token_offset=NonNegativeInt.parse(100),
            )

    def test_fixed_offset_exceeds_length_raises(self) -> None:
        """Offset > len(tokens) should raise AssertionError."""
        token_ids = list(range(100))
        with pytest.raises(AssertionError):
            segment_tokens(
                token_ids,
                SegmentMode.fixed_tokens,
                segment_size=PositiveInt.parse(128),
                token_offset=NonNegativeInt.parse(200),
            )

    def test_fixed_offset_negative_raises(self) -> None:
        """Negative offsets fail at the refined runtime boundary."""
        token_ids = list(range(100))
        bad_offset: Any = -1
        with pytest.raises(BeartypeCallHintParamViolation, match="token_offset"):
            segment_tokens(
                token_ids,
                SegmentMode.fixed_tokens,
                segment_size=PositiveInt.parse(128),
                token_offset=bad_offset,
            )

    def test_fixed_offset_boolean_raises(self) -> None:
        """Boolean false must not satisfy the strict zero-offset contract."""
        token_ids = list(range(100))
        bad_offset: Any = False
        with pytest.raises(BeartypeCallHintParamViolation, match="token_offset"):
            segment_tokens(
                token_ids,
                SegmentMode.fixed_tokens,
                segment_size=PositiveInt.parse(128),
                token_offset=bad_offset,
            )


class TestSFTDataGenerator:
    """Test the SFT data generator produces correct full-sequence batches."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return get_tokenizer_or_skip()

    def test_conditioned_batch_shapes(self, tokenizer):
        """Conditioned batch should have correct shapes and padding masking."""
        task = Alpaca(stop=100)

        B, T = 2, 256
        gen = _make_conditioned_sft_generator(
            task=task,
            tokenizer=tokenizer,
            B=B,
            T=T,
            device="cpu",
            sft_state=SFTState(),
        )
        batch = next(gen)
        inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, _fp_targets = batch

        # Shape checks
        assert inputs.shape == (B, T), f"inputs shape: {inputs.shape}"
        assert targets.shape == (B, T), f"targets shape: {targets.shape}"
        assert enc_fids.ndim == 3 and enc_fids.shape[0] == B
        assert enc_mags.shape == enc_fids.shape
        assert enc_mask.shape == enc_fids.shape
        assert seg_map.shape == (B, T)

        assert enc_mask.any(), "Conditioned batches should include encoder features"

        # Full-sequence SFT only masks padding positions; a densely packed row
        # may have no padding at all.
        assert (targets != -1).any(), "Expected some supervised positions"
        supervised_inner = targets[:, :-1] != -1
        assert t.equal(
            targets[:, :-1][supervised_inner],
            inputs[:, 1:].to(t.int64)[supervised_inner],
        )

    def test_feature_mask_filters_conditioning_and_fp_targets(self, tokenizer):
        task = Alpaca(stop=2)
        features = build_synthetic_features(task, tokenizer, n_sae_features=1000)
        masked_id = int(features[0][0].feature_ids[0].item())
        boundaries = t.linspace(0.5, 9.5, 9).unsqueeze(0).expand(1000, -1).contiguous()
        percentile_table = (
            t.arange(1, 100, dtype=t.float64).unsqueeze(0).expand(1000, -1).contiguous()
        )
        threshold_cfg = ThresholdConfig(
            threshold_type=ThresholdType.segment_topk,
            threshold=1000,
        )

        with patch_load_sft_features(features):
            gen = make_sft_data_generator(
                task=task,
                tokenizer=tokenizer,
                B=1,
                T=256,
                device="cpu",
                sft_state=SFTState(),
                feature_dir="/dummy/features",
                parquet_dir="/dummy/parquet",
                threshold_cfg=threshold_cfg,
                boundaries=boundaries,
                n_sae_features=1000,
                fp_percentile_table=percentile_table,
                feature_mask=FeatureMask(FeatureMaskMode.blacklist, (masked_id,)),
                min_loss_tokens=1,
                alignment_probe_samples=0,
                expected_split=SFTSplit.TRAIN,
            )
            batch = next(gen)

        _inputs, _targets, enc_fids, _enc_mags, enc_mask, _seg_map, fp_targets, *_ = batch
        assert not (enc_fids[enc_mask] == masked_id).any()
        assert fp_targets is not None
        assert t.all(fp_targets[..., masked_id] == 0)

    def test_visual_inspection_tokenized(self, tokenizer):
        """Decode tokenized samples and print for visual verification."""
        task = PKUSafeRLHF(stop=50)

        gen = _make_conditioned_sft_generator(
            task=task,
            tokenizer=tokenizer,
            B=4,
            T=512,
            device="cpu",
            sft_state=SFTState(),
        )
        batch = next(gen)
        inputs, targets, _, _, _, _, _ = batch

        for i in range(min(4, inputs.shape[0])):
            valid_mask = targets[i] != -1
            n_trained = valid_mask.sum().item()
            decoded = tokenizer.decode(inputs[i].tolist())
            print(f"\n--- Tokenized sample {i} ({n_trained} trained tokens) ---")
            print(decoded[:500])

    def test_full_sequence_loss_masks_only_padding(self, tokenizer):
        """Verify targets != -1 for every non-padding target token."""
        task = Alpaca(stop=50)

        B, T = 1, 512
        gen = _make_conditioned_sft_generator(
            task=task,
            tokenizer=tokenizer,
            B=B,
            T=T,
            device="cpu",
            sft_state=SFTState(),
        )
        batch = next(gen)
        inputs, targets, _, _, _, _, _ = batch

        # The number of trained tokens should be > 0 and may equal T when
        # best-fit packing fills the row exactly.
        n_trained = (targets[0] != -1).sum().item()
        assert 0 < n_trained <= T, f"Expected 0 < n_trained <= {T}, got {n_trained}"

    def test_full_sequence_trains_all_non_padding_tokens(self, tokenizer):
        """Full-sequence SFT should match decoder-only full-token supervision."""
        conversation = {
            "messages": [
                {"role": "user", "content": "Spell abc."},
                {"role": "assistant", "content": "abc"},
            ]
        }
        ids, mask = tokenizer.render_conversation(conversation)
        task = _TinyConversationTask(conversation)

        gen = _make_conditioned_sft_generator(
            task=task,
            tokenizer=tokenizer,
            B=1,
            T=len(ids) - 1,
            device="cpu",
            sft_state=SFTState(),
            min_loss_tokens=1,
        )
        _inputs, targets, *_ = next(gen)

        assert targets.shape == (1, len(ids) - 1)
        assert (targets[0] != -1).sum().item() == len(ids) - 1
        assert sum(mask[1:]) < len(ids) - 1

    def test_full_conv_segments_route_user_and_assistant_tokens(self, tokenizer):
        """Full-conversation feature routing should not drop non-final messages."""
        conversation = {
            "messages": [
                {"role": "user", "content": "First question."},
                {"role": "assistant", "content": "First answer."},
                {"role": "user", "content": "Second question."},
                {"role": "assistant", "content": "Second answer."},
            ]
        }
        ids, _mask = tokenizer.render_conversation(conversation)
        offsets = token_to_char_offsets(tokenizer, ids)
        split = next(i for i in range(1, len(ids)) if 0 < offsets[i] < offsets[-1])

        seg_features = (
            [t.tensor([11]), t.tensor([22])],
            [t.tensor([1]), t.tensor([2])],
            [0, offsets[split]],
            [offsets[split], offsets[-1]],
        )
        routing = build_full_sequence_segment_routing(
            ids=ids,
            tokenizer=tokenizer,
            seg_features=seg_features,
            source_bos_char_len=offsets[1],
        )
        fids = routing.per_seg_fids
        buckets = routing.per_seg_buckets
        seg_map = routing.seg_map

        assert len(fids) == 3
        assert len(buckets) == 3
        assert seg_map.shape == (len(ids),)
        assert seg_map[0].item() > 0
        assert seg_map[-1].item() > 0
        assert {1, 2}.issubset(set(seg_map.tolist()))

    def test_multiple_batches(self, tokenizer):
        """Generator should yield multiple batches without error."""
        task = Alpaca(stop=200)

        gen = _make_conditioned_sft_generator(
            task=task,
            tokenizer=tokenizer,
            B=2,
            T=256,
            device="cpu",
            sft_state=SFTState(),
        )
        for _ in range(5):
            batch = next(gen)
            inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, _fp_targets = batch
            assert inputs.shape == (2, 256)
            assert targets.shape == (2, 256)


class TestSFTStateTracking:
    """Test that SFTState properly tracks consumed conversations."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return get_tokenizer_or_skip()

    def test_sft_state_consumed_increases(self, tokenizer):
        """SFTState.consumed must increase as the generator yields batches."""
        task = Alpaca(stop=100)
        state = SFTState()
        assert state.consumed == 0, "SFTState should start at 0"

        gen = _make_conditioned_sft_generator(
            task=task,
            tokenizer=tokenizer,
            B=2,
            T=256,
            device="cpu",
            sft_state=state,
        )

        prev_consumed = 0
        for _ in range(5):
            next(gen)
            assert state.consumed > prev_consumed, (
                f"state.consumed should increase: was {prev_consumed}, now {state.consumed}"
            )
            prev_consumed = state.consumed

    def test_sft_state_not_shared_between_generators(self, tokenizer):
        """Different generators with separate SFTState track independently."""
        task = Alpaca(stop=100)
        state1 = SFTState()
        state2 = SFTState()

        gen1 = _make_conditioned_sft_generator(
            task=task,
            tokenizer=tokenizer,
            B=2,
            T=256,
            device="cpu",
            sft_state=state1,
        )
        gen2 = _make_conditioned_sft_generator(
            task=task,
            tokenizer=tokenizer,
            B=2,
            T=256,
            device="cpu",
            sft_state=state2,
        )

        # Advance gen1 more than gen2
        for _ in range(5):
            next(gen1)
        for _ in range(2):
            next(gen2)

        assert state1.consumed > state2.consumed, (
            f"state1 ({state1.consumed}) should be ahead of state2 ({state2.consumed})"
        )

    def test_sft_state_missing_raises_type_error(self):
        """Calling make_sft_data_generator without sft_state must raise TypeError."""
        generator_factory = cast(Any, make_sft_data_generator)
        with pytest.raises(TypeError, match="sft_state"):
            generator_factory(
                task=Alpaca(stop=10),
                tokenizer=None,
                B=1,
                T=64,
                device="cpu",
            )


class TestTrainingLoop:
    """Test the full training loop (tiny model, few steps)."""

    @pytest.fixture(scope="class")
    def device(self):
        if not t.cuda.is_available():
            pytest.skip("CUDA required for training test")
        return "cuda"

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return get_tokenizer_or_skip()

    @pytest.fixture(scope="class")
    def tiny_model(self, device, tokenizer):
        """Build a tiny encoder-decoder model for testing."""
        vocab_size = tokenizer.get_vocab_size()
        dec_config = GPTConfig(
            sequence_len=256,
            vocab_size=vocab_size,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=128,
            window_pattern="L",
        )
        enc_config = EncoderConfig(n_embd=64, depth=1, head_dim=32)
        xattn_config = CrossAttentionConfig(frequency=1)
        with t.device("meta"):
            model = EncoderDecoderGPT(
                dec_config,
                enc_config,
                xattn_config,
                n_sae_features=1000,
                num_buckets=10,
            )
        model.to_empty(device=device)
        model.init_weights()
        return model

    def test_loss_decreases(self, tiny_model, tokenizer, device):
        """Training loss should decrease over 20 steps."""
        # Encoder-decoder models have many unique parameter shapes, which causes
        # the Muon optimizer's compiled step to recompile for each shape group.
        _reset_dynamo_for_muon_tests()

        task = Alpaca(stop=200)
        gen = _make_conditioned_sft_generator(
            task=task,
            tokenizer=tokenizer,
            B=2,
            T=256,
            device=device,
            sft_state=SFTState(),
        )

        optimizer = tiny_model.setup_optimizer()
        losses = []

        for _step in range(20):
            inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, _fp_targets = next(gen)
            with t.amp.autocast("cuda", dtype=t.bfloat16):
                loss = tiny_model(inputs, targets, enc_fids, enc_mags, enc_mask, seg_map)
            loss.backward()
            optimizer.step()
            tiny_model.zero_grad(set_to_none=True)
            losses.append(loss.item())

        # Loss should generally decrease
        first_5_avg = sum(losses[:5]) / 5
        last_5_avg = sum(losses[-5:]) / 5
        assert last_5_avg < first_5_avg, (
            f"Loss did not decrease: first 5 avg={first_5_avg:.4f}, last 5 avg={last_5_avg:.4f}"
        )

    def test_gradient_flow_conditioned_features(self, tiny_model, tokenizer, device):
        """Full conditioning should send gradients through decoder and encoder embeddings."""
        task = Alpaca(stop=50)
        gen = _make_conditioned_sft_generator(
            task=task,
            tokenizer=tokenizer,
            B=2,
            T=256,
            device=device,
            sft_state=SFTState(),
        )

        tiny_model.zero_grad(set_to_none=True)
        inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, _fp_targets = next(gen)
        with t.amp.autocast("cuda", dtype=t.bfloat16):
            loss = tiny_model(inputs, targets, enc_fids, enc_mags, enc_mask, seg_map)
        loss.backward()

        # Decoder should have gradients
        dec_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in tiny_model.transformer.parameters()
        )
        assert dec_has_grad, "Decoder should have gradients"

        enc_embed_has_grad = (
            tiny_model.encoder.feature_embed.weight.grad is not None
            and tiny_model.encoder.feature_embed.weight.grad.abs().sum() > 0
        )
        assert enc_embed_has_grad, "Encoder embeddings should have conditioned gradients"


class TestCheckpointSaveLoad:
    """Test checkpoint save and resume produces continuous loss trajectory."""

    @pytest.fixture(scope="class")
    def device(self):
        if not t.cuda.is_available():
            pytest.skip("CUDA required")
        return "cuda"

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return get_tokenizer_or_skip()

    def _build_tiny_model(self, device, tokenizer):
        vocab_size = tokenizer.get_vocab_size()
        dec_config = GPTConfig(
            sequence_len=256,
            vocab_size=vocab_size,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=128,
            window_pattern="L",
        )
        enc_config = EncoderConfig(n_embd=64, depth=1, head_dim=32)
        xattn_config = CrossAttentionConfig(frequency=1)
        with t.device("meta"):
            model = EncoderDecoderGPT(
                dec_config,
                enc_config,
                xattn_config,
                n_sae_features=1000,
                num_buckets=10,
            )
        model.to_empty(device=device)
        model.init_weights()
        return model

    def test_checkpoint_continuity(self, device, tokenizer):
        """Save checkpoint, reload, continue -- loss should be continuous."""
        _reset_dynamo_for_muon_tests()

        task = Alpaca(stop=200)

        # Train for 15 steps, save
        model1 = self._build_tiny_model(device, tokenizer)
        optimizer1 = model1.setup_optimizer()
        gen1 = _make_conditioned_sft_generator(
            task=task,
            tokenizer=tokenizer,
            B=2,
            T=256,
            device=device,
            sft_state=SFTState(),
        )

        losses_phase1 = []
        for _step in range(15):
            batch = next(gen1)
            inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, _fp_targets = batch
            with t.amp.autocast("cuda", dtype=t.bfloat16):
                loss = model1(inputs, targets, enc_fids, enc_mags, enc_mask, seg_map)
            loss.backward()
            optimizer1.step()
            model1.zero_grad(set_to_none=True)
            losses_phase1.append(loss.item())

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(
                tmpdir,
                15,
                model1.state_dict(),
                optimizer1.state_dict(),
                {"step": 15},
            )

            # Load into new model
            model2 = self._build_tiny_model(device, tokenizer)
            model_data, optimizer_data, meta = load_checkpoint(
                tmpdir,
                15,
                device,
                load_optimizer=True,
            )
            model2.load_state_dict(model_data, strict=True, assign=True)
            optimizer2 = model2.setup_optimizer()
            optimizer2.load_state_dict(optimizer_data)

        # Continue training for 10 more steps with same data sequence
        gen2 = _make_conditioned_sft_generator(
            task=task,
            tokenizer=tokenizer,
            B=2,
            T=256,
            device=device,
            sft_state=SFTState(),
        )
        # Advance generator to same position
        for _ in range(15):
            next(gen2)

        losses_phase2 = []
        for _step in range(10):
            batch = next(gen2)
            inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, _fp_targets = batch
            with t.amp.autocast("cuda", dtype=t.bfloat16):
                loss = model2(inputs, targets, enc_fids, enc_mags, enc_mask, seg_map)
            loss.backward()
            optimizer2.step()
            model2.zero_grad(set_to_none=True)
            losses_phase2.append(loss.item())

        # Loss should be continuous (no big jump)
        last_loss_phase1 = losses_phase1[-1]
        first_loss_phase2 = losses_phase2[0]
        jump = abs(first_loss_phase2 - last_loss_phase1)
        assert jump < 1.0, (
            f"Loss jump after checkpoint load: {last_loss_phase1:.4f} -> "
            f"{first_loss_phase2:.4f} (jump={jump:.4f})"
        )


def _build_sft_encoder_inputs(ids, tokenizer, seg_features):
    """Test helper: full-sequence routing + _pad_segments for one sample."""
    routing = build_full_sequence_segment_routing(
        ids=ids,
        tokenizer=tokenizer,
        seg_features=seg_features,
    )
    per_seg_fids = routing.per_seg_fids
    per_seg_buckets = routing.per_seg_buckets
    seg_map = routing.seg_map
    # _pad_segments expects per_seg_pcts; build zero-filled placeholders.
    per_seg_pcts = [t.zeros(len(f), dtype=t.float32) for f in per_seg_fids]
    fids, mags, _pcts, mask, seg_map_out = _pad_segments(
        per_seg_fids,
        per_seg_buckets,
        per_seg_pcts,
        seg_map,
    )
    return fids, mags, mask, seg_map_out


class TestFeatureLabeledSFT:
    """Test feature-labeled SFT data generator with synthetic features."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return get_tokenizer_or_skip()

    def test_segment_alignment_synthetic(self, tokenizer):
        """Build synthetic features and verify full-sequence segment routing."""
        conv = {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {
                    "role": "assistant",
                    "content": "The capital of France is Paris. It is a beautiful city.",
                },
            ]
        }
        ids, _mask = tokenizer.render_conversation(conv)
        T = len(ids) - 1  # inputs are ids[:-1]
        input_ids = ids[:-1]
        total_chars = source_aligned_char_offsets(tokenizer, input_ids)[-1]

        # Synthetic features: 1 segment covering the full rendered input.
        syn_fids = [t.tensor([1, 2, 3], dtype=t.int64)]
        syn_mags = [t.tensor([1, 2, 1], dtype=t.int64)]
        syn_char_starts = [0]
        syn_char_ends = [total_chars]

        enc_fids, enc_mags, enc_mask, seg_map = _build_sft_encoder_inputs(
            ids=input_ids,
            tokenizer=tokenizer,
            seg_features=(syn_fids, syn_mags, syn_char_starts, syn_char_ends),
        )

        # Verify shapes
        assert enc_fids.ndim == 2  # (n_seg, F_max)
        assert enc_mask.ndim == 2
        assert seg_map.shape == (T,)

        assert (seg_map > 0).all(), "Every full-sequence token should be conditioned"

    def test_multi_segment_synthetic(self, tokenizer):
        """Test with multiple segments covering different parts of the full input."""
        assistant_text = "The capital of France is Paris. It has been the capital for centuries. Paris is known for the Eiffel Tower."
        conv = {
            "messages": [
                {"role": "user", "content": "Tell me about France's capital."},
                {"role": "assistant", "content": assistant_text},
            ]
        }
        ids, _mask = tokenizer.render_conversation(conv)
        input_ids = ids[:-1]
        total_chars = source_aligned_char_offsets(tokenizer, input_ids)[-1]

        # 3 segments tiling [0, total_chars)
        third = total_chars // 3
        syn_fids = [
            t.tensor([10, 20], dtype=t.int64),
            t.tensor([30, 40, 50], dtype=t.int64),
            t.tensor([60], dtype=t.int64),
        ]
        syn_mags = [
            t.tensor([1, 2], dtype=t.int64),
            t.tensor([3, 1, 2], dtype=t.int64),
            t.tensor([1], dtype=t.int64),
        ]
        syn_char_starts = [0, third, 2 * third]
        syn_char_ends = [third, 2 * third, total_chars]

        enc_fids, enc_mags, enc_mask, seg_map = _build_sft_encoder_inputs(
            ids=input_ids,
            tokenizer=tokenizer,
            seg_features=(syn_fids, syn_mags, syn_char_starts, syn_char_ends),
        )

        # Should have at least 2 segments (1 empty prefix + feature segments)
        assert enc_fids.shape[0] >= 2
        assert (seg_map > 0).all(), "All full-sequence positions should map to features"


class TestPerDatasetLossDecreases:
    """Verify training loss decreases on each SFT dataset."""

    @pytest.fixture(scope="class")
    def device(self):
        if not t.cuda.is_available():
            pytest.skip("CUDA required")
        return "cuda"

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return get_tokenizer_or_skip()

    def _build_tiny_model(self, device, tokenizer):
        vocab_size = tokenizer.get_vocab_size()
        dec_config = GPTConfig(
            sequence_len=256,
            vocab_size=vocab_size,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=128,
            window_pattern="L",
        )
        enc_config = EncoderConfig(n_embd=64, depth=1, head_dim=32)
        xattn_config = CrossAttentionConfig(frequency=1)
        with t.device("meta"):
            model = EncoderDecoderGPT(
                dec_config,
                enc_config,
                xattn_config,
                n_sae_features=1000,
                num_buckets=10,
            )
        model.to_empty(device=device)
        model.init_weights()
        return model

    def _train_steps(self, model, task, tokenizer, device, n_steps=20):
        _reset_dynamo_for_muon_tests()

        gen = _make_conditioned_sft_generator(
            task=task,
            tokenizer=tokenizer,
            B=2,
            T=256,
            device=device,
            sft_state=SFTState(),
        )
        optimizer = model.setup_optimizer()
        losses = []
        for _ in range(n_steps):
            batch = next(gen)
            inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, _fp_targets = batch
            with t.amp.autocast("cuda", dtype=t.bfloat16):
                loss = model(inputs, targets, enc_fids, enc_mags, enc_mask, seg_map)
            loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            losses.append(loss.item())
        return losses

    def test_safety_loss_decreases(self, device, tokenizer):
        task = PKUSafeRLHF(stop=100)
        model = self._build_tiny_model(device, tokenizer)
        losses = self._train_steps(model, task, tokenizer, device)
        first_5 = sum(losses[:5]) / 5
        last_5 = sum(losses[-5:]) / 5
        assert last_5 < first_5, (
            f"Safety loss did not decrease: first 5={first_5:.4f}, last 5={last_5:.4f}"
        )

    def test_general_loss_decreases(self, device, tokenizer):
        task = Alpaca(stop=100)
        model = self._build_tiny_model(device, tokenizer)
        losses = self._train_steps(model, task, tokenizer, device)
        first_5 = sum(losses[:5]) / 5
        last_5 = sum(losses[-5:]) / 5
        assert last_5 < first_5, (
            f"General loss did not decrease: first 5={first_5:.4f}, last 5={last_5:.4f}"
        )

    def test_code_loss_decreases(self, device, tokenizer):
        task = CodeAlpaca(stop=100)
        model = self._build_tiny_model(device, tokenizer)
        losses = self._train_steps(model, task, tokenizer, device)
        first_5 = sum(losses[:5]) / 5
        last_5 = sum(losses[-5:]) / 5
        assert last_5 < first_5, (
            f"Code loss did not decrease: first 5={first_5:.4f}, last 5={last_5:.4f}"
        )


class TestGSM8KSFT:
    """Test GSM8K renders correctly for SFT."""

    def test_gsm8k_renders_correctly(self):
        tokenizer = get_tokenizer_or_skip()
        task = GSM8K(subset="main", split="train", stop=10)
        for i in range(min(5, len(task))):
            conv = task[i]
            ids, mask = tokenizer.render_conversation(conv)
            assert sum(mask) > 0, f"GSM8K example {i} has no assistant tokens"
            decoded = tokenizer.decode(ids)
            assert len(decoded) > 0


class TestSFTMixtures:
    """Verify the shared SFT mixtures produce non-empty Tasks."""

    def test_all_mixture_names_are_available(self):
        assert {mixture.value for mixture in SFTMixture} == {
            "standard",
            "capabilities",
            "safety",
            "standard_steered",
            "standard_steered_balanced",
        }

    def test_all_mixtures_produce_tasks(self, monkeypatch: pytest.MonkeyPatch):
        def fake_build_mixture(_mixture: object, **_overrides: object):
            return (
                [_NonEmptyRegistryTask(), _NonEmptyRegistryTask()],
                [_NonEmptyRegistryTask()],
            )

        monkeypatch.setattr(
            "tasks.mixtures.build_mixture",
            fake_build_mixture,
        )

        for mixture in SFTMixture:
            task, val = build_task_mixtures(mixture)
            assert len(task) > 0, f"Mixture '{mixture.value}' produced no train data"
            assert isinstance(task, SteeredTaskMixture) is (mixture in STEERED_MIXTURES)
            assert val is not None
            assert len(val) > 0, f"Mixture '{mixture.value}' produced no val data"


class TestNoSAELabelsEquivalence:
    """Verify that cross-attention with empty encoder mask produces zero output."""

    def _make_xattn_layer(self):
        from steering.conditional.encoder_decoder.cross_attention import (
            CrossAttentionLayer,
            CrossAttentionLayerConfig,
        )

        config = CrossAttentionLayerConfig(
            dec_n_embd=128,
            enc_n_embd=64,
            n_head=2,
            n_kv_head=2,
            head_dim=32,
        )
        layer = CrossAttentionLayer(config)
        layer.init_weights()
        return layer

    def test_xattn_output_is_zero_with_empty_mask(self):
        """Forward through cross-attention with all-False mask → output is exactly zero."""
        layer = self._make_xattn_layer()
        B, T, n_seg, F = 2, 16, 1, 4
        decoder_hidden = t.randn(B, T, 128)
        encoder_kv = t.randn(B, n_seg, F, 64)
        encoder_mask = t.zeros(B, n_seg, F, dtype=t.bool)  # all False
        seg_map = t.zeros(B, T, dtype=t.int64)

        with t.no_grad():
            output = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert (output == 0).all(), (
            f"Expected all-zero output with empty mask, got max abs: {output.abs().max().item()}"
        )

    def test_no_gradients_to_xattn_params_with_empty_mask(self):
        """Forward+backward with all-False mask → xattn param gradients are None or zero."""
        layer = self._make_xattn_layer()
        B, T, n_seg, F = 2, 16, 1, 4
        decoder_hidden = t.randn(B, T, 128, requires_grad=True)
        encoder_kv = t.randn(B, n_seg, F, 64)
        encoder_mask = t.zeros(B, n_seg, F, dtype=t.bool)
        seg_map = t.zeros(B, T, dtype=t.int64)

        output = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        loss = output.sum()
        loss.backward()

        for name, param in layer.named_parameters():
            if param.grad is not None:
                assert (param.grad == 0).all(), (
                    f"Expected zero gradient for {name}, got max abs: {param.grad.abs().max().item()}"
                )

    def test_gradients_flow_with_real_features(self):
        """With some True mask positions, xattn params get nonzero gradients."""
        layer = self._make_xattn_layer()
        B, T, n_seg, F = 2, 16, 1, 4
        decoder_hidden = t.randn(B, T, 128, requires_grad=True)
        encoder_kv = t.randn(B, n_seg, F, 64)
        encoder_mask = t.ones(B, n_seg, F, dtype=t.bool)  # all True
        seg_map = t.zeros(B, T, dtype=t.int64)

        output = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        loss = output.sum()
        loss.backward()

        has_nonzero = any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters())
        assert has_nonzero, "Expected nonzero gradients with real features"


class TestConfigFromCheckpoint:
    """Test reading encoder/xattn config from checkpoint metadata."""

    def test_roundtrip_save_load(self):
        """Save a tiny model with pretraining-style metadata, load it back."""
        from dataclasses import asdict

        tokenizer = get_tokenizer_or_skip()
        vocab_size = tokenizer.get_vocab_size()

        # Build a tiny model (mimicking pretraining script)
        dec_config = GPTConfig(
            sequence_len=256,
            vocab_size=vocab_size,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=128,
            window_pattern="L",
        )
        enc_config = EncoderConfig(n_embd=64, depth=1, head_dim=32)
        xattn_config = CrossAttentionConfig(frequency=1)
        n_sae_features = 1000
        num_buckets = 10

        with t.device("meta"):
            model = EncoderDecoderGPT(
                dec_config,
                enc_config,
                xattn_config,
                n_sae_features=n_sae_features,
                num_buckets=num_buckets,
            )
        model.to_empty(device="cpu")
        model.init_weights()

        # Save with pretraining-style metadata
        meta = {
            "step": 100,
            "model_config": asdict(dec_config),
            "user_config": {
                "encoder": {
                    "n_embd": 64,
                    "depth": 1,
                    "n_head": None,
                    "n_kv_head": None,
                    "head_dim": 32,
                },
                "cross_attention": {
                    "frequency": 1,
                    "n_head": None,
                    "n_kv_head": None,
                },
                "num_buckets": num_buckets,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(tmpdir, 100, model.state_dict(), None, meta)

            # Load back: extract configs from metadata (same flow as main())
            model_data, _, loaded_meta = load_checkpoint(
                tmpdir,
                100,
                "cpu",
                load_optimizer=False,
            )

            dec_config2 = GPTConfig(**loaded_meta["model_config"])
            enc_config2 = EncoderConfig(**loaded_meta["user_config"]["encoder"])
            xattn_config2 = CrossAttentionConfig(**loaded_meta["user_config"]["cross_attention"])
            loaded_num_buckets = loaded_meta["user_config"]["num_buckets"]

            # Infer n_sae_features from weights
            loaded_n_sae = model_data["encoder.feature_embed.weight"].shape[0]

            with t.device("meta"):
                model2 = EncoderDecoderGPT(
                    dec_config2,
                    enc_config2,
                    xattn_config2,
                    n_sae_features=loaded_n_sae,
                    num_buckets=loaded_num_buckets,
                )
            model2.to_empty(device="cpu")
            model2.init_weights()
            # strict=True: all keys must match exactly
            model2.load_state_dict(model_data, strict=True, assign=True)

        # Verify configs match
        assert dec_config2.n_layer == dec_config.n_layer
        assert dec_config2.n_embd == dec_config.n_embd
        assert loaded_n_sae == n_sae_features
        assert loaded_num_buckets == num_buckets


class TestSFTFeatureLoader:
    """Test load_sft_features with synthetic feature shards and parquet index mapping."""

    def _write_synthetic_shard(self, tmpdir, shard_name, doc_entries):
        """Write a synthetic feature shard in safetensors format.

        doc_entries: list of (doc_id, [(seg_idx, position, fids, fmags, char_start, char_end), ...])
        """
        from steering.feature_io import save_features

        all_fids = []
        all_mags = []
        offsets = [0]
        doc_ids_list = []
        seg_indices_list = []
        seg_positions_list = []
        char_starts_list = []
        char_ends_list = []

        for doc_id, segments in doc_entries:
            for seg_idx, position, fids, fmags, cs, ce in segments:
                all_fids.append(fids)
                all_mags.append(fmags)
                offsets.append(offsets[-1] + len(fids))
                doc_ids_list.append(doc_id)
                seg_indices_list.append(seg_idx)
                seg_positions_list.append(position)
                char_starts_list.append(cs)
                char_ends_list.append(ce)

        n_segments = len(doc_ids_list)
        data = {
            "offsets": t.tensor(offsets, dtype=t.int64),
            "feature_ids": t.cat(all_fids).to(t.int32) if all_fids else t.tensor([], dtype=t.int32),
            "feature_mags": t.cat(all_mags).to(t.float32)
            if all_mags
            else t.tensor([], dtype=t.float32),
            "doc_ids": t.tensor(doc_ids_list, dtype=t.int64),
            "segment_indices": t.tensor(seg_indices_list, dtype=t.int64),
            "segment_positions": t.tensor(seg_positions_list, dtype=t.int64),
            "segment_char_starts": t.tensor(char_starts_list, dtype=t.int64),
            "segment_char_ends": t.tensor(char_ends_list, dtype=t.int64),
            "n_segments": n_segments,
            "n_docs": len(doc_entries),
            "segment_size": 128,
            "layer": 8,
            "context_window": 2048,
            "pooling": "mean",
            "segment_mode": "fixed_tokens",
        }
        save_features(data, f"{tmpdir}/{shard_name}_features")

    def _write_parquet_shard(self, parquet_dir, shard_name, original_indices):
        """Write a parquet shard with original_index column."""
        from pathlib import Path

        import pyarrow as pa
        import pyarrow.parquet as pq

        Path(parquet_dir).mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "text": ["dummy"] * len(original_indices),
                "original_index": original_indices,
            }
        )
        pq.write_table(table, f"{parquet_dir}/{shard_name}.parquet")

    def test_load_single_shard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = f"{tmpdir}/features"
            parquet_dir = f"{tmpdir}/parquet"
            import os

            os.makedirs(feature_dir)

            self._write_synthetic_shard(
                feature_dir,
                "shard_00000",
                [
                    (
                        0,
                        [(0, 0, t.tensor([1, 2, 3]), t.tensor([0.5, 1.0, 0.3]), 0, 50)],
                    ),
                    (
                        1,
                        [
                            (0, 0, t.tensor([10, 20]), t.tensor([0.8, 0.2]), 0, 30),
                            (1, 128, t.tensor([30]), t.tensor([1.5]), 30, 60),
                        ],
                    ),
                ],
            )
            # No skips: doc 0 -> mixture idx 0, doc 1 -> mixture idx 1
            self._write_parquet_shard(parquet_dir, "shard_00000", [0, 1])

            features = load_sft_features(feature_dir, parquet_dir)

            assert 0 in features
            assert 1 in features
            assert len(features[0]) == 1
            assert len(features[1]) == 2
            # Segments should be sorted by segment_idx
            assert features[1][0].segment_idx == 0
            assert features[1][1].segment_idx == 1
            # Check feature data
            assert features[0][0].feature_ids.tolist() == [1, 2, 3]

    def test_load_multiple_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = f"{tmpdir}/features"
            parquet_dir = f"{tmpdir}/parquet"
            import os

            os.makedirs(feature_dir)

            self._write_synthetic_shard(
                feature_dir,
                "shard_00000",
                [
                    (0, [(0, 0, t.tensor([1]), t.tensor([0.5]), 0, 50)]),
                ],
            )
            self._write_synthetic_shard(
                feature_dir,
                "shard_00001",
                [
                    (0, [(0, 0, t.tensor([10, 20]), t.tensor([0.8, 0.2]), 0, 100)]),
                ],
            )
            # Shard 0: 1 doc -> mixture idx 0; Shard 1: 1 doc -> mixture idx 5
            self._write_parquet_shard(parquet_dir, "shard_00000", [0])
            self._write_parquet_shard(parquet_dir, "shard_00001", [5])

            features = load_sft_features(feature_dir, parquet_dir)

            assert 0 in features
            assert 5 in features
            assert len(features) == 2

    def test_original_index_remapping_with_skips(self):
        """Verify that features are keyed by original mixture index, not sequential doc_id.

        Simulates a shard where conversations 2 and 5 were skipped during parquet
        conversion (no assistant tokens). The parquet has 8 rows from a range of 10
        conversations (0-9), with original_indices [0,1,3,4,6,7,8,9].
        Features should be keyed by these original indices, NOT by 0-7.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = f"{tmpdir}/features"
            parquet_dir = f"{tmpdir}/parquet"
            import os

            os.makedirs(feature_dir)

            # 8 docs in the feature shard (2 were skipped from original 10)
            doc_entries = []
            for local_id in range(8):
                doc_entries.append(
                    (
                        local_id,
                        [(0, 0, t.tensor([local_id * 10 + 1]), t.tensor([1.0]), 0, 50)],
                    )
                )
            self._write_synthetic_shard(feature_dir, "shard_00000", doc_entries)

            # Parquet records original mixture indices (skipped 2 and 5)
            original_indices = [0, 1, 3, 4, 6, 7, 8, 9]
            self._write_parquet_shard(parquet_dir, "shard_00000", original_indices)

            features = load_sft_features(feature_dir, parquet_dir)

            # Features should be keyed by original mixture index
            assert set(features.keys()) == {0, 1, 3, 4, 6, 7, 8, 9}
            # Skipped indices should NOT be present
            assert 2 not in features
            assert 5 not in features
            # Verify correct feature data is mapped to correct index
            # local_doc_id=0 (mixture idx 0) should have feature_id [1]
            assert features[0][0].feature_ids.tolist() == [1]
            # local_doc_id=2 (mixture idx 3) should have feature_id [21]
            assert features[3][0].feature_ids.tolist() == [21]
            # local_doc_id=4 (mixture idx 6) should have feature_id [41]
            assert features[6][0].feature_ids.tolist() == [41]
            # local_doc_id=7 (mixture idx 9) should have feature_id [71]
            assert features[9][0].feature_ids.tolist() == [71]

    def test_cross_shard_remapping_with_skips(self):
        """Verify correct remapping across multiple shards with different skip patterns.

        Shard 0: conversations 0-4, skipped 1 -> parquet rows for [0, 2, 3, 4]
        Shard 1: conversations 5-9, skipped 7 -> parquet rows for [5, 6, 8, 9]

        The key issue this tests: features must be keyed by original mixture index
        across shard boundaries, not by cumulative sequential doc_id.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = f"{tmpdir}/features"
            parquet_dir = f"{tmpdir}/parquet"
            import os

            os.makedirs(feature_dir)

            # Shard 0: 4 docs (skipped mixture idx 1)
            self._write_synthetic_shard(
                feature_dir,
                "shard_00000",
                [
                    (0, [(0, 0, t.tensor([100]), t.tensor([1.0]), 0, 10)]),  # mixture 0
                    (1, [(0, 0, t.tensor([200]), t.tensor([1.0]), 0, 20)]),  # mixture 2
                    (2, [(0, 0, t.tensor([300]), t.tensor([1.0]), 0, 30)]),  # mixture 3
                    (3, [(0, 0, t.tensor([400]), t.tensor([1.0]), 0, 40)]),  # mixture 4
                ],
            )
            self._write_parquet_shard(parquet_dir, "shard_00000", [0, 2, 3, 4])

            # Shard 1: 4 docs (skipped mixture idx 7)
            self._write_synthetic_shard(
                feature_dir,
                "shard_00001",
                [
                    (0, [(0, 0, t.tensor([500]), t.tensor([1.0]), 0, 50)]),  # mixture 5
                    (1, [(0, 0, t.tensor([600]), t.tensor([1.0]), 0, 60)]),  # mixture 6
                    (2, [(0, 0, t.tensor([700]), t.tensor([1.0]), 0, 70)]),  # mixture 8
                    (3, [(0, 0, t.tensor([800]), t.tensor([1.0]), 0, 80)]),  # mixture 9
                ],
            )
            self._write_parquet_shard(parquet_dir, "shard_00001", [5, 6, 8, 9])

            features = load_sft_features(feature_dir, parquet_dir)

            # Should have all 8 non-skipped indices
            assert set(features.keys()) == {0, 2, 3, 4, 5, 6, 8, 9}
            assert 1 not in features  # skipped in shard 0
            assert 7 not in features  # skipped in shard 1

            # Verify correct feature data per original index
            assert features[0][0].feature_ids.tolist() == [100]
            assert features[2][0].feature_ids.tolist() == [200]
            assert features[5][0].feature_ids.tolist() == [500]
            assert features[9][0].feature_ids.tolist() == [800]

    def test_missing_parquet_raises(self):
        """An empty parquet dir is diagnosed as a PATH problem, not divergence.

        The stem set-difference would call this "orphan feature shards …
        regenerate the matched artifact family" and send the operator off to
        redo a multi-hour conversion; the usual cause is a typo'd or unmounted
        ``sft_parquet_dir`` (review finding 9, PR #1311).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = f"{tmpdir}/features"
            parquet_dir = f"{tmpdir}/parquet"
            import os

            os.makedirs(feature_dir)
            os.makedirs(parquet_dir)

            self._write_synthetic_shard(
                feature_dir,
                "shard_00000",
                [
                    (0, [(0, 0, t.tensor([1]), t.tensor([0.5]), 0, 50)]),
                ],
            )
            # No parquet written

            with pytest.raises(
                DataLoadError,
                match=r"contains no \*\.parquet shards.*check sft_parquet_dir",
            ):
                load_sft_features(feature_dir, parquet_dir)

    def test_missing_original_index_column_raises(self):
        """Verify that parquet without original_index column raises a clear error."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = f"{tmpdir}/features"
            parquet_dir = f"{tmpdir}/parquet"
            import os

            os.makedirs(feature_dir)
            os.makedirs(parquet_dir)

            self._write_synthetic_shard(
                feature_dir,
                "shard_00000",
                [
                    (0, [(0, 0, t.tensor([1]), t.tensor([0.5]), 0, 50)]),
                ],
            )
            # Write parquet WITHOUT original_index
            table = pa.table({"text": ["dummy"]})
            pq.write_table(table, f"{parquet_dir}/shard_00000.parquet")

            with pytest.raises(DataLoadError, match="original_index"):
                load_sft_features(feature_dir, parquet_dir)

    def test_identity_mapping_no_skips(self):
        """When no conversations are skipped, original_index == sequential doc_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = f"{tmpdir}/features"
            parquet_dir = f"{tmpdir}/parquet"
            import os

            os.makedirs(feature_dir)

            n_docs = 50
            doc_entries = [
                (i, [(0, 0, t.tensor([i]), t.tensor([1.0]), 0, 10)]) for i in range(n_docs)
            ]
            self._write_synthetic_shard(feature_dir, "shard_00000", doc_entries)
            self._write_parquet_shard(parquet_dir, "shard_00000", list(range(n_docs)))

            features = load_sft_features(feature_dir, parquet_dir)

            assert len(features) == n_docs
            for i in range(n_docs):
                assert i in features
                assert features[i][0].feature_ids.tolist() == [i]

    def test_multi_segment_docs_with_remapping(self):
        """Verify multi-segment documents are correctly remapped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = f"{tmpdir}/features"
            parquet_dir = f"{tmpdir}/parquet"
            import os

            os.makedirs(feature_dir)

            # Doc 0 (mixture idx 10) has 3 segments, doc 1 (mixture idx 15) has 1
            self._write_synthetic_shard(
                feature_dir,
                "shard_00000",
                [
                    (
                        0,
                        [
                            (0, 0, t.tensor([1, 2]), t.tensor([1.0, 2.0]), 0, 100),
                            (1, 128, t.tensor([3, 4]), t.tensor([3.0, 4.0]), 100, 200),
                            (2, 256, t.tensor([5]), t.tensor([5.0]), 200, 300),
                        ],
                    ),
                    (
                        1,
                        [
                            (0, 0, t.tensor([10]), t.tensor([1.0]), 0, 50),
                        ],
                    ),
                ],
            )
            self._write_parquet_shard(parquet_dir, "shard_00000", [10, 15])

            features = load_sft_features(feature_dir, parquet_dir)

            assert set(features.keys()) == {10, 15}
            assert len(features[10]) == 3
            assert len(features[15]) == 1
            # Verify segment ordering
            assert features[10][0].segment_idx == 0
            assert features[10][1].segment_idx == 1
            assert features[10][2].segment_idx == 2
            assert features[10][0].char_start == 0
            assert features[10][0].char_end == 100


class TestConditionedSFTDataGenerator:
    """Test make_sft_data_generator with full feature conditioning."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return get_tokenizer_or_skip()

    def _write_synthetic_shard(self, tmpdir, shard_name, doc_entries):
        """Write a synthetic feature shard in safetensors format."""
        from steering.feature_io import save_features

        all_fids = []
        all_mags = []
        offsets = [0]
        doc_ids_list = []
        seg_indices_list = []
        seg_positions_list = []
        char_starts_list = []
        char_ends_list = []

        for doc_id, segments in doc_entries:
            for seg_idx, position, fids, fmags, cs, ce in segments:
                all_fids.append(fids)
                all_mags.append(fmags)
                offsets.append(offsets[-1] + len(fids))
                doc_ids_list.append(doc_id)
                seg_indices_list.append(seg_idx)
                seg_positions_list.append(position)
                char_starts_list.append(cs)
                char_ends_list.append(ce)

        n_segments = len(doc_ids_list)
        data = {
            "offsets": t.tensor(offsets, dtype=t.int64),
            "feature_ids": t.cat(all_fids).to(t.int32) if all_fids else t.tensor([], dtype=t.int32),
            "feature_mags": t.cat(all_mags).to(t.float32)
            if all_mags
            else t.tensor([], dtype=t.float32),
            "doc_ids": t.tensor(doc_ids_list, dtype=t.int64),
            "segment_indices": t.tensor(seg_indices_list, dtype=t.int64),
            "segment_positions": t.tensor(seg_positions_list, dtype=t.int64),
            "segment_char_starts": t.tensor(char_starts_list, dtype=t.int64),
            "segment_char_ends": t.tensor(char_ends_list, dtype=t.int64),
            "n_segments": n_segments,
            "n_docs": len(doc_entries),
            "segment_size": 128,
            "layer": 8,
            "context_window": 2048,
            "pooling": "mean",
            "segment_mode": "fixed_tokens",
        }
        save_features(data, f"{tmpdir}/{shard_name}_features")

    def _write_parquet_shard(self, parquet_dir, shard_name, original_indices):
        """Write a parquet shard with original_index column."""
        from pathlib import Path

        import pyarrow as pa
        import pyarrow.parquet as pq

        Path(parquet_dir).mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "text": ["dummy"] * len(original_indices),
                "original_index": original_indices,
            }
        )
        pq.write_table(table, f"{parquet_dir}/{shard_name}.parquet")

    def test_conditioned_shapes_and_segments(self, tokenizer):
        """Verify conditioned shapes and non-trivial seg_map."""
        from steering.conditional.thresholding import ThresholdConfig

        task = Alpaca(stop=20)

        # Write synthetic features for each example: one segment covering the
        # full rendered sample in feature-source coordinates.
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = f"{tmpdir}/features"
            parquet_dir = f"{tmpdir}/parquet"
            import os

            os.makedirs(feature_dir)

            doc_entries = []
            for i in range(20):
                conv = task[i]
                ids, _mask = tokenizer.render_conversation(conv)
                total_chars = source_aligned_char_offsets(tokenizer, ids)[-1]
                # Feature segment covers the full rendered sample.
                doc_entries.append(
                    (
                        i,
                        [
                            (
                                0,
                                0,
                                t.tensor([1, 2, 3]),
                                t.tensor([5.0, 3.0, 1.0]),
                                0,
                                total_chars,
                            )
                        ],
                    )
                )
            self._write_synthetic_shard(feature_dir, "shard_00000", doc_entries)
            # No skips: original_index == sequential doc_id
            self._write_parquet_shard(parquet_dir, "shard_00000", list(range(20)))

            # Use a simple threshold config that keeps all features
            from steering.conditional.thresholding import ThresholdType

            threshold_cfg = ThresholdConfig(
                threshold_type=ThresholdType.segment_topk,
                threshold=1000,
            )
            # Per-feature boundaries for the synthetic feature ids.
            boundaries = t.linspace(0.5, 9.5, 9).unsqueeze(0).expand(1000, -1)

            B, T = 2, 256
            gen = make_sft_data_generator(
                task=task,
                tokenizer=tokenizer,
                B=B,
                T=T,
                device="cpu",
                sft_state=SFTState(),
                feature_dir=feature_dir,
                parquet_dir=parquet_dir,
                threshold_cfg=threshold_cfg,
                boundaries=boundaries,
                alignment_probe_samples=0,
                expected_split=SFTSplit.TRAIN,
            )

            batch = next(gen)
            inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, _fp_targets = batch

            # Shape checks
            assert inputs.shape == (B, T), f"inputs shape: {inputs.shape}"
            assert targets.shape == (B, T), f"targets shape: {targets.shape}"
            assert enc_fids.ndim == 3 and enc_fids.shape[0] == B
            assert enc_mags.shape == enc_fids.shape
            assert enc_mask.shape == enc_fids.shape
            assert seg_map.shape == (B, T)

            # With features, some encoder mask positions should be True
            assert enc_mask.any(), "With features, some encoder masks should be True"

            # seg_map should have non-zero entries for conditioned positions.
            assert (seg_map > 0).any(), (
                "Expected non-zero seg_map entries for conditioned positions"
            )

            # Full-sequence loss is active for every non-padding target.
            assert (targets != -1).any(), "Expected some unmasked positions"
