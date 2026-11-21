"""Launch the continuous self-conditioned chat TUI."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import hydra
import torch as t
from omegaconf import DictConfig

from scripts.chat_tui.app import ChatApp, ModelThread
from scripts.chat_tui.convo import ConvoCatalog
from scripts.chat_tui.descriptions import OfflineDescriptionService
from scripts.chat_tui.extraction import GemmaExtractionService, TurnGemmaExtractionService
from scripts.chat_tui.session import (
    ChatSession,
    ChatSessionConfig,
    build_session_self_cond_config,
)
from scripts.chat_tui.setup import load_chat_setup, load_chat_target_model
from scripts.chat_tui.signals import AxbenchSignalsSource
from scripts.chat_tui.steering import (
    SteeringMethod,
    SteeringSelection,
    build_score_policy,
)
from scripts.chat_tui.turn_session import TurnChatSession
from scripts.completion_tui.description_service import DescriptionService
from scripts.completion_tui.transparent import enable_transparent_background
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfConditionedFeatureProvider,
)
from steering.segmentation import SegmentMode


def _no_steer_provider_probe(setup, model, config: ChatSessionConfig, device: str) -> None:
    """Construct and discard the same no-steer provider shape used by the session."""

    head_width = (
        int(setup.feature_vocab.size)
        if setup.feature_vocab is not None
        else int(setup.n_sae_features)
    )
    policy = build_score_policy(
        SteeringSelection(SteeringMethod.OFF, ()),
        positive_selection=setup.output_selection,
        negative_selection=setup.output_negative_selection,
        head_width=head_width,
        feature_mask=setup.feature_mask,
        head_row_original_ids=(
            None if setup.feature_vocab is None else setup.feature_vocab.new_to_old_tensor()
        ),
    )
    selection_rng = t.Generator(device="cpu")
    selection_rng.manual_seed(config.seed + 1)
    provider_config = build_session_self_cond_config(
        setup,
        config,
        selection_rng,
        policy,
        segment_mode=setup.segment_mode,
    )
    SelfConditionedFeatureProvider(model, provider_config, t.device(device))


def _load_eager_extraction(
    model_thread: ModelThread,
    extraction: GemmaExtractionService,
) -> None:
    """Synchronously load the extraction stack on its owning model thread."""

    complete = threading.Event()
    errors: list[Exception] = []

    def job() -> None:
        try:
            extraction.extraction_stack()
        except Exception as exc:
            errors.append(exc)
        finally:
            complete.set()

    model_thread.submit(job)
    complete.wait()
    if errors:
        raise errors[0]


def build_app(cfg: DictConfig) -> tuple[ChatApp, ModelThread]:
    """Build the app and its single started model worker."""

    setup = load_chat_setup(
        Path(cfg.target_meta_path),
        Path(cfg.feature_statistics_path),
        pretrain_meta_path=None if cfg.pretrain_meta_path is None else Path(cfg.pretrain_meta_path),
    )
    turn_mode = setup.segment_mode is SegmentMode.turn_delimited_fixed_tokens
    cadence = (
        setup.turn_self_cond_segment_size_rustbpe if turn_mode else int(cfg.self_cond_segment_size)
    )
    if type(cadence) is not int or cadence <= 0:
        raise ValueError("chat TUI requires a positive integer self-conditioning cadence")
    device = str(cfg.device)
    model, tokenizer = load_chat_target_model(setup, device, dtype=str(cfg.dtype))
    session_config = ChatSessionConfig(
        self_cond_segment_size=cadence,
        temperature=float(cfg.temperature),
        top_k=None if cfg.top_k is None else int(cfg.top_k),
        seed=int(cfg.seed),
        max_new_tokens_per_turn=int(cfg.max_new_tokens_per_turn),
        deterministic_rewind=bool(cfg.deterministic_rewind),
    )
    _no_steer_provider_probe(setup, model, session_config, device)
    extraction = (
        TurnGemmaExtractionService(setup, tokenizer, device, Path(cfg.feature_statistics_path))
        if turn_mode
        else GemmaExtractionService(setup, tokenizer, device)
    )
    session_type = TurnChatSession if turn_mode else ChatSession
    session = session_type(
        setup=setup,
        model=model,
        tokenizer=tokenizer,
        device=device,
        config=session_config,
        extraction=extraction,
    )

    model_thread = ModelThread()
    model_thread.start()
    description_service = None
    try:
        if bool(cfg.eager_extraction):
            _load_eager_extraction(model_thread, extraction)

        def catalog_loader() -> ConvoCatalog:
            return ConvoCatalog.load(tokenizer)

        if cfg.descriptions_json_path:
            description_service = OfflineDescriptionService(Path(cfg.descriptions_json_path))
        elif cfg.neuronpedia_model_id and cfg.neuronpedia_sae_source:
            description_service = DescriptionService(
                str(cfg.neuronpedia_model_id),
                str(cfg.neuronpedia_sae_source),
                Path(cfg.description_cache_path),
            )
        app_holder: list[ChatApp] = []

        def marshal(callback: Callable[[], None]) -> None:
            if not app_holder:
                raise RuntimeError("chat app is not ready to marshal callbacks")
            app_holder[0].call_from_thread(callback)

        signals_source = AxbenchSignalsSource(
            setup=setup,
            prepared_dir=Path(cfg.prepared_dir),
            cache_dir=Path(cfg.signal_cache_dir),
            model_thread=model_thread,
            extraction_service=extraction,
            marshal=marshal,
        )
        app = ChatApp(
            session=session,
            tokenizer=tokenizer,
            checkpoint_label=Path(setup.target_meta_path).name,
            seed=session_config.seed,
            catalog_loader=catalog_loader,
            sequence_len=int(model.dec_config.sequence_len),
            max_new_tokens_per_turn=session_config.max_new_tokens_per_turn,
            signals_source=signals_source,
            description_service=description_service,
            deterministic_rewind=session_config.deterministic_rewind,
            model_thread=model_thread,
        )
        app_holder.append(app)
        return app, model_thread
    except Exception:
        if description_service is not None:
            description_service.shutdown()
        model_thread.shutdown()
        raise


@hydra.main(config_path="../../configs/serving", config_name="chat_tui", version_base=None)
def main(cfg: DictConfig) -> None:
    app, model_thread = build_app(cfg)
    description_service = getattr(app, "_description_service", None)
    try:
        enable_transparent_background(app)
        app.run()
    finally:
        if description_service is not None and hasattr(description_service, "shutdown"):
            description_service.shutdown()
        model_thread.shutdown()


if __name__ == "__main__":
    main()
