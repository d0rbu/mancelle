"""Focused contracts for the real-Gemma offline integration oracle."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from omegaconf import OmegaConf

from steering.conditional.feature_mask import FeatureMaskMode
from steering.eval.axbench_base.stage0c_api_generation import Stage0cApiGenerator
from steering.eval.axbench_base.synth.batching import SynthRequest
from steering.eval.axbench_base.synth.continuation_guards import (
    ResponseEnvelope,
    prefix_suffix_echo_chars,
)
from steering.eval.axbench_base.synth.continuation_protocol import (
    build_stage1_continuation_request,
)
from steering.eval.axbench_base.synth.llm_response_cache import make_cache_key
from steering.eval.axbench_base.synth.qc import AvoidContinuationOutcome
from steering.eval.axbench_base.templates import load_template
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.cache_identity import resolve_feature_selection_hash
from tests.axbench_base.fixtures import real_gemma_offline as oracle_module
from tests.axbench_base.fixtures.offline_openrouter import (
    OfflineOpenRouterError,
    OfflineOpenRouterTransport,
    OpenRouterRequest,
)
from tests.axbench_base.integration import test_full_pipeline_real_gemma as integration


def _request() -> OpenRouterRequest:
    return OpenRouterRequest(
        ordinal=0,
        url="https://openrouter.invalid",
        model="offline",
        headers={},
        timeout=None,
        payload={},
        messages=[],
        request_class=None,
        client=None,
        cache_key=None,
        payload_hash="0" * 64,
        prompt_hash="0" * 64,
    )


def _request_with_class(request_class: str) -> OpenRouterRequest:
    request = _request()
    request.request_class = request_class
    return request


def test_local_oracle_rejects_unknown_request_class() -> None:
    oracle = oracle_module.RealGemmaOfflineOracle(OfflineOpenRouterTransport())

    with pytest.raises(OfflineOpenRouterError, match="unknown request class"):
        oracle.response(_request_with_class("unexpected_stage"))


def test_exact_request_class_guard_rejects_an_extra_class() -> None:
    transport = OfflineOpenRouterTransport()
    transport.requests.extend(
        _request_with_class(request_class)
        for request_class in (
            *sorted(integration._EXPECTED_OPENROUTER_REQUEST_CLASSES),
            "unexpected_stage",
        )
    )

    with pytest.raises(AssertionError, match="request classes changed"):
        integration._assert_exact_openrouter_request_classes(transport)


def test_exact_stage4_method_guard_rejects_a_dropped_method() -> None:
    expected = integration._EXPECTED_REAL_GEMMA_STAGE4_METHODS
    assert len(expected) == 25
    integration._assert_exact_stage4_methods(set(expected))

    with pytest.raises(AssertionError, match="method set changed"):
        integration._assert_exact_stage4_methods(
            set(expected) - {"ours_replace_all_encdec"}
        )


@pytest.mark.parametrize(
    ("request_class", "expected_roles"),
    (
        ("stage0c_definition", ("user",)),
        ("stage0c_continuation", ("system", "user")),
        ("stage1_synth_rich", ("user",)),
        ("stage1_synth_avoid", ("user",)),
        ("stage5_judge", ("user",)),
    ),
)
def test_openrouter_message_roles_are_pinned_per_request_class(
    request_class: str,
    expected_roles: tuple[str, ...],
) -> None:
    request = _request_with_class(request_class)
    request.messages = [
        {"role": role, "content": f"{role} content"}
        for role in expected_roles
    ]
    integration._assert_openrouter_message_roles(request)

    wrong_roles = (
        ("user",)
        if expected_roles == ("system", "user")
        else ("system", "user")
    )
    request.messages = [
        {"role": role, "content": f"{role} content"}
        for role in wrong_roles
    ]
    with pytest.raises(AssertionError, match="message roles changed"):
        integration._assert_openrouter_message_roles(request)


def test_real_stage4_identity_resolves_then_provider_reaches_missing_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "missing-context-cache"
    conditioning = integration._real_stage4_conditioning_config(
        cache_root,
        stats_source_sha256="c" * 64,
    )
    feature_selection_hash = resolve_feature_selection_hash(
        conditioning["identity"],
        conditioning,
    )
    assert len(feature_selection_hash) == 64
    assert set(feature_selection_hash) <= set("0123456789abcdef")
    cfg = SimpleNamespace(
        output_root=str(tmp_path),
        run_tag="real-provider-missing-cache",
        conditioning=conditioning,
    )
    monkeypatch.setattr(
        integration._cli,
        "load_stage4_contexts_manifest",
        lambda _path: {
            "encdec_sequence_len": 1024,
            "contexts_fingerprint": "b" * 64,
            "feature_contract_fingerprint": None,
        },
    )

    with pytest.raises(
        AxBenchBaseError,
        match="Failed to build the AxBench-Base Stage 4 encoder-decoder",
    ) as exc_info:
        integration._cli._build_stage4_encdec_context_provider(
            cast(Any, cfg),
            encdec_model=SimpleNamespace(
                dec_config=SimpleNamespace(sequence_len=1024)
            ),
            encdec_meta=cast(
                Any,
                SimpleNamespace(
                    n_sae_features=16384,
                    num_buckets=2,
                ),
            ),
        )

    assert exc_info.value.__cause__ is not None
    assert cache_root.as_posix() in str(exc_info.value.__cause__)


def test_local_oracle_removes_only_a_leading_prefix_suffix_echo() -> None:
    source = "Earlier unrelated material. The exact local source suffix"

    assert oracle_module._without_leading_prefix_echo(
        source,
        "  The exact local source suffix then continues naturally.",
    ) == " then continues naturally."
    assert oracle_module._without_leading_prefix_echo(
        source,
        "The exact local source suffixThe exact local source suffix new text.",
    ) == " new text."
    assert oracle_module._without_leading_prefix_echo(
        source,
        "A genuinely new continuation.",
    ) == "A genuinely new continuation."


def test_local_oracle_places_exact_member_peak_near_prompt_start() -> None:
    tokenizer = SimpleNamespace(
        encode=lambda text, **_kwargs: text.split(),
        decode=lambda tokens: " ".join(tokens),
    )
    evidence = [
        {
            "context_before_peak": "The archive offers ",
            "peak_token_text": "downloads",
            "context_after_peak": " for every release.",
        },
        {
            "context_before_peak": "ignored ",
            "peak_token_text": "second",
            "context_after_peak": " row",
        },
    ]

    assert oracle_module._activation_evidence_prefixes(
        evidence,
        tokenizer=tokenizer,
        max_scored_tokens=8,
    ) == (
        "The archive offers downloads for every release.",
        "ignored second row",
    )


def test_local_oracle_never_begins_evidence_inside_a_token() -> None:
    tokenizer = SimpleNamespace(
        encode=lambda text, **_kwargs: text.split(),
        decode=lambda tokens: " ".join(tokens),
    )
    evidence = [
        {
            "context_before_peak": (
                "unrelated prose then how to create an economy marked by higher"
            ),
            "peak_token_text": " skills",
            "context_after_peak": ", higher productivity and higher wages.",
        }
    ]

    assert oracle_module._activation_evidence_prefixes(
        evidence,
        tokenizer=tokenizer,
        max_scored_tokens=10,
    ) == (
        "to create an economy marked by higher skills, higher productivity and "
        "higher wages.",
    )
    assert oracle_module._activation_evidence_prefixes(
        evidence,
        tokenizer=tokenizer,
        max_scored_tokens=1,
    ) == ("skills, higher productivity and higher wages.",)


def test_local_oracle_generation_budget_excludes_irrelevant_tail() -> None:
    tokenizer = SimpleNamespace(
        encode=lambda text, **_kwargs: text.split(),
    )

    assert oracle_module._local_generation_budget(
        tokenizer,
        " ".join(f"token-{index}" for index in range(80)),
        scored_tokens=64,
    ) == 2
    assert oracle_module._local_generation_budget(
        tokenizer,
        " ".join(f"token-{index}" for index in range(60)),
        scored_tokens=64,
    ) == 6
    assert oracle_module._local_generation_budget(
        tokenizer,
        " ".join(f"token-{index}" for index in range(10)),
        scored_tokens=64,
    ) == 56


@pytest.mark.parametrize(
    "template_name",
    (
        "synth_continue_with_concept.txt",
        "synth_continue_avoiding_concept.txt",
    ),
)
def test_local_oracle_extracts_prefix_from_live_synth_templates(
    template_name: str,
) -> None:
    oracle = oracle_module.RealGemmaOfflineOracle(OfflineOpenRouterTransport())
    prefix = "corpus text\n\nEND SOURCE.\nthat marker belongs to the corpus"
    continuation_request = build_stage1_continuation_request(
        load_template(template_name),
        label="real_gemma_oracle_test",
        kind="STAGE1_ORACLE_TEST",
        identity={"template_name": template_name},
        prefix_text=prefix,
        description="test concept",
    )

    assert oracle._stage1_prefix(continuation_request.prompt, _request()) == prefix
    assert oracle._stage1_response(
        continuation_request.prompt,
        prefix,
        " new text",
    ) == (
        continuation_request.envelope.begin
        + "\n"
        + prefix[-96:]
        + " new text "
        + continuation_request.envelope.end
    )


def test_local_oracle_round_robins_hashed_stage0c_member_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = oracle_module.RealGemmaOfflineOracle(OfflineOpenRouterTransport())
    evidence_by_member = {
        8: (
            {
                "context_before_peak": "representative",
                "peak_token_text": " one",
                "context_after_peak": " first",
            },
            {
                "context_before_peak": "representative",
                "peak_token_text": " one",
                "context_after_peak": " second",
            },
        ),
        4775: (
            {
                "context_before_peak": "sibling",
                "peak_token_text": " evidence",
                "context_after_peak": " first",
            },
            {
                "context_before_peak": "sibling",
                "peak_token_text": " evidence",
                "context_after_peak": " second",
            },
        ),
    }
    tokenizer = SimpleNamespace(
        encode=lambda text, **_kwargs: text.split(),
        decode=lambda tokens: " ".join(tokens),
    )
    context = SimpleNamespace(
        cfg=SimpleNamespace(output_length=8),
        cluster_contract=SimpleNamespace(
            target_for_representative=lambda _feature_id: SimpleNamespace(
                allowed_member_ids=(8, 4775)
            )
        ),
    )
    envelope = ResponseEnvelope(
        begin="[[AXBENCH_DEFINITION_BEGIN_A1]]",
        end="[[AXBENCH_DEFINITION_END_A1]]",
    )
    monkeypatch.setattr(oracle, "_ctx", lambda _request: context)
    monkeypatch.setattr(oracle, "_user_prompt", lambda _request: "definition prompt")
    monkeypatch.setattr(oracle, "_envelope", lambda _prompt: envelope)
    monkeypatch.setattr(
        oracle,
        "_evidence_rows",
        lambda _prompt, request: list(
            evidence_by_member[oracle._feature_id(request)]
        ),
    )

    for member_id in (8, 4775):
        request = _request()
        request.cache_key = make_cache_key(
            model="offline",
            prompt=f"definition prompt for {member_id}",
            tag=f"stage0c_definition_f{member_id}_mt1024_a0",
        )
        assert len(request.cache_key) == 64
        assert f"f{member_id}" not in request.cache_key
        with oracle.stage0c_member_context(member_id):
            oracle._definition_response(request)

    assert set(oracle._activation_evidence) == {8, 4775}

    assert oracle._stage1_evidence_prefixes(
        cast(Any, context),
        representative_id=8,
        tokenizer=tokenizer,
    ) == (
        "representative one first",
        "sibling evidence first",
        "representative one second",
        "sibling evidence second",
    )


def test_live_oracle_uses_stage0c_member_attrs_with_opaque_cache_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = oracle_module.RealGemmaOfflineOracle(OfflineOpenRouterTransport())
    request = _request()
    request.cache_key = make_cache_key(
        model="offline",
        prompt="clustered continuation prompt",
        tag="stage0c_continuation_target_f8_member_f4775_c0_s0",
    )
    assert len(request.cache_key) == 64
    assert "4775" not in request.cache_key

    def definition_for_feature(_generator: Any, _feature_id: int) -> int:
        return oracle._feature_id(request)

    def run_one(
        _llm_client: Any,
        _synth_request: Any,
        *,
        profile_region_name: str,
        submitted_at: float,
    ) -> tuple[int, int | None, str, float]:
        return (
            oracle._feature_id(request),
            oracle_module._feature_context.get(),
            profile_region_name,
            submitted_at,
        )

    monkeypatch.setattr(
        Stage0cApiGenerator,
        "_definition_for_feature",
        definition_for_feature,
    )
    monkeypatch.setattr(integration._synth_batching, "_run_one", run_one)
    integration._bind_live_oracle_contexts(monkeypatch, oracle)

    generator = object.__new__(Stage0cApiGenerator)
    with oracle.feature_context(8):
        assert generator._definition_for_feature(4775) == 4775
        result = integration._synth_batching._run_one(
            object(),
            SynthRequest(
                index=0,
                prompt="clustered continuation prompt",
                cache_key=request.cache_key,
                attrs={"description_feature_id": 4775},
            ),
            profile_region_name="stage0c.api.complete",
            submitted_at=12.5,
        )

    assert result == (4775, 8, "stage0c.api.complete", 12.5)


def test_peak_injection_cannot_reconstruct_a_prefix_suffix_echo() -> None:
    echoed_body = "a long source suffix whose repeated body exceeds the minimum"
    source = f"Earlier material / {echoed_body}"

    generated = oracle_module._with_required_peak(source, echoed_body, "/")

    assert generated.startswith(". / ")
    assert echoed_body not in generated
    assert prefix_suffix_echo_chars(source, generated) == 0


def _oracle_that_fails_first_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[oracle_module.RealGemmaOfflineOracle, Any, list[int], list[int]]:
    oracle = oracle_module.RealGemmaOfflineOracle(OfflineOpenRouterTransport())
    oracle._max_seeds = 2
    cfg = SimpleNamespace(
        verification_length=2,
        verification_percentile_threshold=0.5,
        extraction_tokenizer_id="test-tokenizer",
        output_length=2,
        accept_threshold=1.0,
        extraction_threshold=0.0,
        generation=SimpleNamespace(master_seed=11),
    )
    concept = SimpleNamespace(
        concept_id=3,
        feature_index=7,
        description="test concept",
    )
    context = SimpleNamespace(
        cfg=cfg,
        concepts={3: concept},
        base_model=object(),
        base_sae=object(),
        percentile_table=object(),
        target_model_meta=SimpleNamespace(sae_layer=4),
        cluster_contract=None,
    )
    tokenizer = SimpleNamespace(
        encode=lambda text, **_kwargs: text.split(),
        decode=lambda _ids: "decoded continuation",
    )
    monkeypatch.setattr(oracle, "_ctx", lambda _request: context)
    monkeypatch.setattr(oracle, "_user_prompt", lambda _request: "Variation key: 7")
    monkeypatch.setattr(
        oracle_module,
        "get_extraction_tokenizer",
        lambda _tokenizer_id: tokenizer,
    )

    seeds: list[int] = []
    token_budgets: list[int] = []

    def generate(
        _context: Any,
        _prompt: str,
        *,
        seed: int,
        max_new_tokens: int,
    ) -> str:
        seeds.append(seed)
        token_budgets.append(max_new_tokens)
        if len(seeds) == 1:
            raise ValueError("synthetic immediate EOS")
        return "generated continuation"

    monkeypatch.setattr(oracle, "_generate_text", generate)
    monkeypatch.setattr(
        oracle,
        "_stage1_response",
        lambda _prompt, _prefix, continuation: continuation.lstrip(),
    )
    monkeypatch.setattr(
        oracle,
        "_envelope",
        lambda _prompt: SimpleNamespace(begin="[[BEGIN]]", end="[[END]]"),
    )
    return oracle, context, seeds, token_budgets


def test_stage0c_oracle_retries_immediate_eos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle, context, seeds, token_budgets = _oracle_that_fails_first_generation(
        monkeypatch
    )
    oracle._definitions[7] = ("definition", "required-peak")
    cluster_target = object()
    context.cluster_contract = SimpleNamespace(
        target_for_representative=lambda _feature_id: cluster_target
    )
    original_generate = oracle._generate_text

    def echo_prompt_after_retry(*args: Any, **kwargs: Any) -> str:
        generated = original_generate(*args, **kwargs)
        return f"{args[1]} {generated}"

    monkeypatch.setattr(oracle, "_generate_text", echo_prompt_after_retry)
    monkeypatch.setattr(oracle, "_feature_id", lambda _request: 7)
    monkeypatch.setattr(oracle, "_source_context", lambda _prompt, _request: "source")
    monkeypatch.setattr(
        oracle,
        "_source_parts",
        lambda _source, _request: ("prefix", " bridge"),
    )
    monkeypatch.setattr(
        oracle,
        "_envelope",
        lambda _prompt: SimpleNamespace(begin="[[BEGIN]]", end="[[END]]"),
    )
    # main moved this parser to tests.axbench_base.stage0c_api_helpers and the
    # fixture now calls the module-level import, so patch where it is looked up.
    monkeypatch.setattr(
        oracle_module,
        "parse_continuation_response",
        lambda *_args, **_kwargs: [1, 2],
    )
    scored_cluster_targets: list[Any] = []

    def compute(*_args: Any, **kwargs: Any) -> tuple[float, float]:
        scored_cluster_targets.append(kwargs.get("cluster_target"))
        return 0.0, 100.0

    monkeypatch.setattr(oracle_module, "compute_fc_percentiles", compute)

    response = oracle._stage0c_continuation_response(_request())

    assert "generated continuation" in response
    assert "required-peak" in response
    assert response.count(
        "In the following years, the project continued to grow, and"
    ) == 1
    assert seeds == [7, 1_000_010]
    assert token_budgets == [2, 2]
    assert scored_cluster_targets == [cluster_target]


def test_stage1_rich_oracle_retries_immediate_eos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle, context, seeds, _token_budgets = _oracle_that_fails_first_generation(
        monkeypatch
    )
    cluster_target = object()
    context.cluster_contract = SimpleNamespace(
        target_for_representative=lambda _feature_id: cluster_target
    )
    monkeypatch.setattr(oracle, "_concept_id", lambda _request: 3)
    monkeypatch.setattr(oracle, "_stage1_prefix", lambda _prompt, _request: "prefix")
    monkeypatch.setattr(
        oracle,
        "_stage1_evidence_prefixes",
        lambda *_args, **_kwargs: ("evidence one", "evidence two"),
    )
    monkeypatch.setattr(
        oracle_module,
        "parse_stage1_continuation_response",
        lambda **_kwargs: "decoded continuation",
    )
    scored_cluster_targets: list[Any] = []

    def compute(*_args: Any, **kwargs: Any) -> tuple[float, float]:
        scored_cluster_targets.append(kwargs.get("cluster_target"))
        return 0.0, 100.0

    monkeypatch.setattr(oracle_module, "compute_fc_percentiles", compute)

    assert oracle._stage1_rich_response(_request()) == (
        "evidence two generated continuation"
    )
    assert seeds == [3038, 3039]
    assert scored_cluster_targets == [cluster_target]


def test_stage1_rich_oracle_rejects_half_point_then_accepts_thirty_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle, context, _seeds, _token_budgets = _oracle_that_fails_first_generation(
        monkeypatch
    )
    context.cfg.accept_threshold = 0.3
    monkeypatch.setattr(
        oracle,
        "_generate_text",
        lambda *_args, **_kwargs: "generated continuation",
    )
    monkeypatch.setattr(oracle, "_concept_id", lambda _request: 3)
    monkeypatch.setattr(oracle, "_stage1_prefix", lambda _prompt, _request: "prefix")
    monkeypatch.setattr(
        oracle,
        "_stage1_evidence_prefixes",
        lambda *_args, **_kwargs: ("evidence one", "evidence two"),
    )
    monkeypatch.setattr(
        oracle_module,
        "parse_stage1_continuation_response",
        lambda **kwargs: kwargs["response"],
    )
    percentile_pairs = iter(((10.0, 10.5), (10.0, 40.0)))
    scored = 0

    def compute(*_args: Any, **_kwargs: Any) -> tuple[float, float]:
        nonlocal scored
        scored += 1
        return next(percentile_pairs)

    monkeypatch.setattr(oracle_module, "compute_fc_percentiles", compute)

    response = oracle._stage1_rich_response(_request())

    assert response == "evidence two generated continuation"
    assert scored == 2


def test_stage1_avoid_oracle_retries_immediate_eos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle, context, seeds, token_budgets = _oracle_that_fails_first_generation(
        monkeypatch
    )
    cluster_target = object()
    context.cluster_contract = SimpleNamespace(
        target_for_representative=lambda _feature_id: cluster_target
    )
    monkeypatch.setattr(oracle, "_concept_id", lambda _request: 3)
    monkeypatch.setattr(oracle, "_stage1_prefix", lambda _prompt, _request: "prefix")
    monkeypatch.setattr(
        oracle_module,
        "parse_stage1_continuation_response",
        lambda **_kwargs: "decoded continuation",
    )
    classified_cluster_targets: list[Any] = []

    def classify(**kwargs: Any) -> AvoidContinuationOutcome:
        classified_cluster_targets.append(kwargs.get("cluster_target"))
        return AvoidContinuationOutcome.INACTIVE

    monkeypatch.setattr(oracle_module, "classify_avoid_continuation", classify)

    assert oracle._stage1_avoid_response(_request()) == "generated continuation"
    assert seeds == [6092, 6093]
    assert token_budgets == [2, 2]
    assert classified_cluster_targets == [cluster_target]


def test_live_oracle_wrappers_forward_cluster_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = object()
    context_contracts: list[Any] = []

    class _Oracle:
        @contextmanager
        def stage_context(self, *_args: Any, **kwargs: Any) -> Any:
            context_contracts.append(kwargs.get("cluster_contract"))
            yield

    def prepare(*_args: Any, **kwargs: Any) -> Any:
        return kwargs.get("cluster_contract")

    def positive(*_args: Any, **kwargs: Any) -> Any:
        return kwargs.get("cluster_contract")

    def negative(*_args: Any, **kwargs: Any) -> Any:
        return kwargs.get("cluster_contract")

    monkeypatch.setattr(integration._cli, "prepare_negative_prefixes", prepare)
    monkeypatch.setattr(integration._cli, "synthesize_positive_mode", positive)
    monkeypatch.setattr(integration._cli, "synthesize_negative_mode", negative)
    integration._bind_live_oracle_contexts(
        monkeypatch,
        cast(oracle_module.RealGemmaOfflineOracle, _Oracle()),
    )
    bound_prepare = cast(Any, integration._cli.prepare_negative_prefixes)
    bound_positive = cast(Any, integration._cli.synthesize_positive_mode)
    bound_negative = cast(Any, integration._cli.synthesize_negative_mode)

    assert (
        bound_prepare(
            object(),
            object(),
            object(),
            object(),
            object(),
            cluster_contract=contract,
        )
        is contract
    )
    assert (
        bound_positive(
            object(),
            [],
            [],
            object(),
            object(),
            object(),
            cluster_contract=contract,
        )
        is contract
    )
    assert (
        bound_negative(
            object(),
            [],
            object(),
            object(),
            object(),
            cluster_contract=contract,
        )
        is contract
    )
    assert context_contracts == [contract, contract, contract]


def test_real_oracle_remembers_live_cluster_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = object()
    oracle = oracle_module.RealGemmaOfflineOracle(OfflineOpenRouterTransport())
    monkeypatch.setattr(
        oracle_module,
        "read_n_sae_features_from_stats",
        lambda _path: 16_384,
    )
    monkeypatch.setattr(
        oracle_module,
        "build_percentile_table_from_stats",
        lambda *_args, **_kwargs: object(),
    )

    with oracle.stage_context(
        cast(
            Any,
            SimpleNamespace(
                feature_statistics_path=str(tmp_path / "statistics.safetensors")
            ),
        ),
        base_model=object(),
        base_sae=object(),
        target_model_meta=cast(Any, object()),
        concepts=[],
        cluster_contract=cast(Any, contract),
    ):
        pass

    assert oracle.cluster_contract is contract


def test_real_stage4_cache_builder_forwards_cluster_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cluster_source = tmp_path / "feature-clusters.yaml"
    cluster_source.write_text(
        '{"schema_version": 1, "clusters": [[8, 4775]]}\n',
        encoding="utf-8",
    )
    mask_source = tmp_path / "feature-mask.yaml"
    mask_source.write_text(
        "[8, 4775]\n",
        encoding="utf-8",
    )
    checkpoint_dir = tmp_path / "clustered-checkpoint"
    mask_config = {"mode": "whitelist", "path": str(mask_source)}
    contract = SimpleNamespace(
        feature_mask=SimpleNamespace(mode=FeatureMaskMode.whitelist),
        feature_clusters=SimpleNamespace(source_path=cluster_source),
        checkpoint=SimpleNamespace(
            checkpoint_dir=checkpoint_dir,
            step=54_600,
            resolved_tag="d12-clustered",
        ),
        feature_contract_fingerprint="f" * 64,
        provenance_dict=lambda: {"feature_mask_path": str(mask_source)},
    )
    cfg = OmegaConf.create(
        {
            "conditioning": {
                "identity": {
                    "model_id": "gemma-2-2b",
                    "sae_id": "gemmascope-residual-l8-16k",
                    "layer": 8,
                    "pooling": "max_pool",
                    "feature_selection_hash": None,
                },
                "feature_mask": None,
                "feature_clusters": None,
            }
        }
    )
    cfg_obj = SimpleNamespace(
        run_tag="clustered-real-gemma",
        output_root=str(tmp_path),
        seed_extraction_gpu_batch_size=2,
        seed_extraction_cpu_batch_size=4,
        extraction_tokenizer_id="test-tokenizer",
    )
    encdec_meta = SimpleNamespace(
        n_sae_features=16_384,
        num_buckets=2,
        feature_selection={
            "distribution": "uniform",
            "delta_weights": "percentile_delta",
        },
        negative_selection=None,
        feature_mask={"mode": "blacklist", "ids": [99]},
    )
    runtime_state_kwargs: list[dict[str, Any]] = []
    runtime_state_closed: list[bool] = []

    class _RuntimeState:
        def __init__(self, **kwargs: Any) -> None:
            runtime_state_kwargs.append(kwargs)

        @contextmanager
        def encdec_model(self) -> Any:
            yield SimpleNamespace(
                _loaded_meta_data={"user_config": {"seed": 42}}
            )

        def encdec_meta(self) -> Any:
            return encdec_meta

        @contextmanager
        def base_model(self) -> Any:
            yield SimpleNamespace(nnterp_model=None, native_model=None)

        @contextmanager
        def base_sae(self) -> Any:
            yield object()

        def close(self) -> None:
            runtime_state_closed.append(True)

    monkeypatch.setattr(integration._cli, "cfg_to_dataclass", lambda _cfg: cfg_obj)
    monkeypatch.setattr(integration._cli, "RuntimeState", _RuntimeState)
    monkeypatch.setattr(
        integration._cli,
        "_encdec_model_sequence_len",
        lambda _model: 1024,
    )
    monkeypatch.setattr(
        integration,
        "parse_feature_dir",
        lambda _path: SimpleNamespace(
            layer=8,
            pooling="max_pool",
            segment_size=128,
            activation_type="res",
            dataset_phase=SimpleNamespace(value="base"),
            context_independent_features=False,
            ignore_bos_features=False,
        ),
    )
    monkeypatch.setattr(
        integration._seed_pools,
        "_sae_hook_name",
        lambda _sae: "blocks.8.hook_resid_post",
    )
    monkeypatch.setattr(
        integration._seed_pools,
        "_run_l0_tier",
        lambda _cfg: "dev",
    )

    def process_shard(*_args: Any, **kwargs: Any) -> Any:
        output_path = Path(kwargs["output_dir"]) / "features.safetensors"
        output_path.write_bytes(b"real-stage3e-extraction")
        return SimpleNamespace(output_path=str(output_path))

    monkeypatch.setattr(integration, "process_shard", process_shard)
    monkeypatch.setattr(integration, "sha256_file", lambda _path: "s" * 64)
    built_configs: list[dict[str, Any]] = []
    validated_configs: list[dict[str, Any]] = []
    resolved_hashes: list[str] = []
    manifest_path = tmp_path / "cache" / "c1024" / ("h" * 16) / "manifest.json"

    def build_cache(cache_cfg: Any) -> Path:
        config = OmegaConf.to_container(cache_cfg, resolve=True)
        assert isinstance(config, dict)
        built_configs.append(cast(dict[str, Any], config))
        resolved_hashes.append(
            resolve_feature_selection_hash(config["identity"], config)
        )
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text("{}\n", encoding="utf-8")
        return manifest_path

    def validate_cache(cache_cfg: Any) -> None:
        config = OmegaConf.to_container(cache_cfg, resolve=True)
        assert isinstance(config, dict)
        validated_configs.append(cast(dict[str, Any], config))

    monkeypatch.setattr(
        integration._context_cache_cli,
        "build_cache_from_config",
        build_cache,
    )
    monkeypatch.setattr(
        integration._context_cache_cli,
        "validate_cache_from_config",
        validate_cache,
    )
    monkeypatch.setattr(
        integration,
        "load_context_cache_manifest",
        lambda _cache_dir: {
            "identity": {"feature_selection_hash": "h" * 64}
        },
    )

    result = integration._build_real_stage4_context_cache(
        cfg,
        run_tag="clustered-real-gemma",
        out_root=tmp_path / "out",
        oracle=cast(
            oracle_module.RealGemmaOfflineOracle,
            SimpleNamespace(cluster_contract=contract),
        ),
    )

    assert result == manifest_path
    assert runtime_state_kwargs == [
        {"cfg": cfg_obj, "cluster_contract": contract}
    ]
    assert runtime_state_closed == [True]
    assert validated_configs == built_configs
    assert len(built_configs) == 1
    assert len(resolved_hashes) == 1
    assert len(resolved_hashes[0]) == 64
    assert set(resolved_hashes[0]) <= set("0123456789abcdef")
    built = built_configs[0]
    assert built["feature_selection"] == {"distribution": "uniform"}
    assert built["feature_mask"] == mask_config
    assert built["feature_clusters"] == {"path": str(cluster_source)}
    assert built["checkpoint_dir"] == str(checkpoint_dir)
    assert built["checkpoint_step"] == 54_600
    assert built["checkpoint_tag"] == "d12-clustered"
    assert built["metadata"] == {
        "source": "axbench-base-real-gemma-integration",
        "feature_contract_fingerprint": "f" * 64,
    }
    resumed_conditioning = OmegaConf.to_container(cfg.conditioning, resolve=True)
    assert isinstance(resumed_conditioning, dict)
    assert resumed_conditioning["feature_mask"] == mask_config
    assert resumed_conditioning["feature_clusters"] == {
        "path": str(cluster_source)
    }
    assert resumed_conditioning["feature_selection"] == {
        "distribution": "uniform"
    }
    assert resumed_conditioning["identity"]["feature_selection_hash"] == "h" * 64
