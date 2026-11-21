"""Campaign 2 mixtures, local parent identity, and measured epoch contracts."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hydra import compose, initialize_config_dir

from scripts.utils import build_hf_sft_data_manifest as manifest
from steering.sft.hf_decoder import validate_decoder_sft_config
from steering.sft.loss_mask import LossMaskMode
from tasks.common import TaskMixture
from tasks.customjson import CustomJSON
from tasks.mixture_names import SFTMixture
from tests.scripts.campaign2_fixtures import export_fixture

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mixture", [SFTMixture.SAFETY, SFTMixture.CAPABILITIES])
def test_chained_config_uses_local_parent_and_matching_mixture_sizes(
    mixture: SFTMixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_PARENT_EXPORT", "/data/step_000700")
    monkeypatch.setenv("HF_PARENT_REVISION", "a" * 40)
    monkeypatch.setenv("NUM_ITERATIONS", "800")
    with initialize_config_dir(config_dir=str(ROOT / "configs/training"), version_base=None):
        cfg = compose(config_name=f"chat_sft_pythia_2_8b_{mixture.value}")
    validate_decoder_sft_config(cfg, world_size=4)
    assert cfg.mixture == mixture.value and cfg.steered_jsonl is None
    assert cfg.huggingface.model_id == "/data/step_000700"
    assert cfg.huggingface.revision == "a" * 40
    assert cfg.sft_loss_mask == "completion_only"
    assert cfg.huggingface.restart_save_every == cfg.save_every == 100
    assert cfg.simple_spelling_size == 200000 and cfg.spelling_bee_size == 80000
    assert (cfg.max_seq_len, cfg.total_batch_size, cfg.device_batch_size) == (2048, 524288, 1)


@pytest.mark.parametrize("mixture", [SFTMixture.SAFETY, SFTMixture.CAPABILITIES])
def test_manifest_binds_actual_stage_and_rejects_content_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mixture: SFTMixture,
) -> None:
    identity = tmp_path / "identity_conversations.jsonl"
    identity.write_text(json.dumps([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]) + "\n")
    words = tmp_path / "words_alpha.txt"
    words.write_text("word\n")
    monkeypatch.setattr(manifest, "ensure_word_list_available", lambda **kwargs: words)
    calls = []

    def build(selected: SFTMixture, **kwargs: object) -> tuple[TaskMixture, TaskMixture]:
        calls.append((selected, kwargs))
        task = TaskMixture([CustomJSON(str(identity))] * 4)
        return task, task

    monkeypatch.setattr(manifest, "build_task_mixtures", build)
    monkeypatch.setattr(manifest, "load_hf_config_and_tokenizer", lambda **kwargs: (
        None, None, SimpleNamespace(chat_template_sha256="a" * 64),
    ))
    epoch = {"world_size": 4, "device_batch_size": 1, "max_seq_len": 2048,
             "total_batch_size": 524288, "sft_loss_mask": "completion_only",
             "gradient_accumulation_steps": 64, "per_rank_steps": [1, 1, 1, 2],
             "num_iterations": 2, "scheduled_tokens": 1048576}
    kwargs: dict[str, Any] = {"base_dir": tmp_path, "mixture": mixture, "loss_mask": LossMaskMode.COMPLETION_ONLY,
              "measured_epoch": epoch}
    first = manifest.build_manifest(**kwargs)
    assert calls == [(mixture, {})]
    assert first["mixture"] == mixture.value and "steered" not in first["inputs"]
    assert ("spelling_words" in first["inputs"]) == (mixture is SFTMixture.CAPABILITIES)
    output = tmp_path / "frozen.json"
    manifest.write_manifest(first, output)
    identity.write_text(identity.read_text().replace('"A"', '"B"'))
    with pytest.raises(ValueError, match="live SFT data differs"):
        manifest.verify_manifest(manifest.build_manifest(**kwargs), output)
    with pytest.raises(ValueError, match="does not consume"):
        manifest.build_manifest(**kwargs, steered_jsonl=identity)


def test_parent_hashes_weights_and_pins_source_revision(tmp_path: Path) -> None:
    root = export_fixture(tmp_path)
    before = manifest._parent_identity(root, SFTMixture.STANDARD_STEERED)
    assert before["source_revision"] == "a" * 40
    (root / "model.safetensors").write_bytes(b"changed weights")
    assert before != manifest._parent_identity(root, SFTMixture.STANDARD_STEERED)
    with pytest.raises(ValueError, match="wrong parent mixture"):
        manifest._parent_identity(root, SFTMixture.SAFETY)


def test_parent_build_reuses_frozen_epoch_and_authenticates_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = tmp_path / "identity_conversations.jsonl"
    identity.write_text(
        json.dumps(
            [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
        )
        + "\n"
    )
    task = TaskMixture([CustomJSON(str(identity))] * 4)
    monkeypatch.setattr(manifest, "build_task_mixtures", lambda *args, **kwargs: (task, task))
    monkeypatch.setattr(
        manifest,
        "load_hf_config_and_tokenizer",
        lambda **kwargs: (None, None, SimpleNamespace(chat_template_sha256="b" * 64)),
    )
    epoch = {
        "world_size": 4,
        "device_batch_size": 1,
        "max_seq_len": 2048,
        "total_batch_size": 524288,
        "sft_loss_mask": "completion_only",
        "gradient_accumulation_steps": 64,
        "per_rank_steps": [1, 1, 1, 2],
        "num_iterations": 2,
        "scheduled_tokens": 1048576,
    }
    data = manifest.build_manifest(
        base_dir=tmp_path,
        mixture=SFTMixture.SAFETY,
        loss_mask=LossMaskMode.COMPLETION_ONLY,
        measured_epoch=epoch,
    )
    data_manifest = tmp_path / "data.json"
    manifest.write_manifest(data, data_manifest)
    parent_export = export_fixture(tmp_path)

    def no_replay(*args: Any, **kwargs: Any) -> None:
        pytest.fail("parent-bound build must reuse the frozen CPU horizon")

    monkeypatch.setattr(manifest, "measure_epoch", no_replay)
    output = tmp_path / "bound.json"
    cfg = SimpleNamespace(
        mode="build",
        output_path=str(output),
        parent_export=str(parent_export),
        data_manifest=str(data_manifest),
        expected_parent_mixture="standard_steered",
        base_dir=str(tmp_path),
        steered_jsonl=None,
        mixture="safety",
        device_batch_size=1,
        sft_loss_mask="completion_only",
        sft_config_name="chat_sft_pythia_2_8b",
    )
    manifest.main.__wrapped__(cfg)
    bound = json.loads(output.read_bytes())
    assert bound["epoch"] == epoch
    assert bound["campaign2_parent"]["source_revision"] == "a" * 40
    assert bound["campaign2_data_sha256"] == hashlib.sha256(
        data_manifest.read_bytes()
    ).hexdigest()
    monkeypatch.setattr(
        manifest,
        "load_hf_config_and_tokenizer",
        lambda **kwargs: (
            None,
            None,
            SimpleNamespace(
                chat_template_sha256=(
                    "changed" if kwargs["model_id"] == str(parent_export) else "b" * 64
                )
            ),
        ),
    )
    with pytest.raises(ValueError, match="parent tokenizer"):
        manifest.main.__wrapped__(SimpleNamespace(**{**vars(cfg), "output_path": str(tmp_path / "other.json")}))


@pytest.mark.parametrize("field,value,message", [
    ("sft_loss_mask", "full_sequence", "completion_only"),
    ("completed_optimizer_step", 8, "step mismatch"),
    ("mixture", "capabilities", "wrong parent mixture"),
])
def test_parent_metadata_fails_closed(tmp_path: Path, field: str, value: object, message: str) -> None:
    root = export_fixture(tmp_path)
    path = root / "sft_metadata.json"
    meta = json.loads(path.read_text())
    meta[field] = value
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=message):
        manifest._parent_identity(root, SFTMixture.STANDARD_STEERED)
