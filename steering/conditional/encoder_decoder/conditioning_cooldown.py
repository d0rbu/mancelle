"""Step-indexed conditioning dropout, applied after training-batch prefetch.

The schedule uses zero-based optimizer-update indices. Randomness is private
and keyed by the consumed update/microbatch, so lookahead and evaluation cannot
advance it. Existing constant dataloader dropout is deliberately unchanged.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch as t

_CONFIG_KEYS = frozenset(
    {
        "enabled",
        "start_fraction",
        "end_fraction",
        "start_step",
        "end_step",
        "cooldown_tokens",
        "granularity",
    }
)


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        bound = "" if minimum is None else f" >= {minimum}"
        raise ValueError(
            f"conditioning_cooldown.{name} must be a strict integer{bound}, got {value!r}"
        )
    return value


def _fraction(value: object, name: str) -> float:
    if (type(value) is not int and type(value) is not float) or not 0 <= value <= 1:
        raise ValueError(
            f"conditioning_cooldown.{name} must be finite and in [0, 1], got {value!r}"
        )
    return float(value)


def _granularity(value: object) -> str:
    if type(value) is not str or value not in ("segment", "document"):
        raise ValueError(
            f"conditioning_cooldown.granularity must be segment or document, got {value!r}"
        )
    return str(value)


@dataclass(frozen=True)
class ConditioningCooldownConfig:
    start_fraction: float
    end_fraction: float
    start_step: int | None
    end_step: int | None
    cooldown_tokens: int
    granularity: str


def parse_conditioning_cooldown(raw: object) -> ConditioningCooldownConfig | None:
    """Validate the small Hydra block; absent/disabled preserves legacy behavior."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("conditioning_cooldown must be a mapping or null")
    raw = cast("Mapping[str, Any]", raw)
    unknown = set(raw) - _CONFIG_KEYS
    if unknown:
        raise ValueError(
            f"conditioning_cooldown contains unknown keys: {sorted(unknown, key=str)!r}"
        )
    enabled = raw.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("conditioning_cooldown.enabled must be a bool")
    config = ConditioningCooldownConfig(
        start_fraction=_fraction(raw.get("start_fraction", 1.0), "start_fraction"),
        end_fraction=_fraction(raw.get("end_fraction", 0.5), "end_fraction"),
        start_step=(
            None
            if raw.get("start_step") is None
            else _integer(raw["start_step"], "start_step", minimum=0)
        ),
        end_step=(
            None
            if raw.get("end_step") is None
            else _integer(raw["end_step"], "end_step", minimum=0)
        ),
        cooldown_tokens=_integer(
            raw.get("cooldown_tokens", 20_000_000_000), "cooldown_tokens", minimum=1
        ),
        granularity=_granularity(raw.get("granularity", "segment")),
    )
    if (
        config.start_step is not None
        and config.end_step is not None
        and config.start_step >= config.end_step
    ):
        raise ValueError("conditioning_cooldown requires start_step < end_step")
    return config if enabled else None


@dataclass(frozen=True)
class ConditioningCooldown:
    start_fraction: float
    end_fraction: float
    start_step: int
    end_step: int
    granularity: str
    seed: int
    world_size: int
    device_batch_size: int
    max_seq_len: int
    total_batch_size: int
    num_iterations: int
    device_type: str
    splice_policy: str = "junctions_last"

    def fraction_at(self, step: int) -> float:
        """Return endpoint-clamped retained probability for an optimizer update."""
        _integer(step, "step", minimum=0)
        if step <= self.start_step:
            return self.start_fraction
        if step >= self.end_step:
            return self.end_fraction
        progress = (step - self.start_step) / (self.end_step - self.start_step)
        return self.start_fraction + progress * (self.end_fraction - self.start_fraction)

    def metadata_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, **asdict(self)}

    def resolve_config(
        self, config: ConditioningCooldownConfig | None
    ) -> ConditioningCooldown | None:
        """Resolve a config against this checkpoint's validated training geometry."""
        return resolve_conditioning_cooldown(
            config,
            num_iterations=self.num_iterations,
            total_batch_size=self.total_batch_size,
            seed=self.seed,
            world_size=self.world_size,
            device_batch_size=self.device_batch_size,
            max_seq_len=self.max_seq_len,
            device_type=self.device_type,
        )

    @classmethod
    def from_metadata(cls, raw: object) -> ConditioningCooldown:
        """Read exact resolved metadata without accepting partial or coerced state."""
        expected = {"schema_version", *cls.__dataclass_fields__}
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise ValueError(
                "conditioning_cooldown checkpoint metadata has missing or unknown fields"
            )
        raw = cast("Mapping[str, Any]", raw)
        if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
            raise ValueError("conditioning_cooldown checkpoint schema_version must equal 1")
        if raw["splice_policy"] != "junctions_last":
            raise ValueError(
                "conditioning_cooldown checkpoint splice_policy must be junctions_last"
            )
        config = parse_conditioning_cooldown(
            {
                "enabled": True,
                "start_fraction": raw["start_fraction"],
                "end_fraction": raw["end_fraction"],
                "start_step": _integer(raw["start_step"], "start_step", minimum=0),
                "end_step": _integer(raw["end_step"], "end_step", minimum=0),
                "granularity": raw["granularity"],
            }
        )
        resolved = resolve_conditioning_cooldown(
            config,
            num_iterations=raw["num_iterations"],
            total_batch_size=raw["total_batch_size"],
            seed=raw["seed"],
            world_size=raw["world_size"],
            device_batch_size=raw["device_batch_size"],
            max_seq_len=raw["max_seq_len"],
            device_type=raw["device_type"],
        )
        assert resolved is not None
        return resolved


def resolve_conditioning_cooldown(
    config: ConditioningCooldownConfig | None,
    *,
    num_iterations: int,
    total_batch_size: int,
    seed: int,
    world_size: int,
    device_batch_size: int,
    max_seq_len: int,
    device_type: str,
) -> ConditioningCooldown | None:
    """Resolve the inclusive update window using nominal global training tokens."""
    if config is None:
        return None
    for name, value in (
        ("num_iterations", num_iterations),
        ("total_batch_size", total_batch_size),
        ("world_size", world_size),
        ("device_batch_size", device_batch_size),
        ("max_seq_len", max_seq_len),
    ):
        _integer(value, name, minimum=1)
    _integer(seed, "seed")
    if device_type not in ("cpu", "cuda", "mps"):
        raise ValueError("conditioning_cooldown.device_type must be cpu, cuda, or mps")
    if total_batch_size % (world_size * device_batch_size * max_seq_len):
        raise ValueError(
            "conditioning_cooldown total_batch_size must be divisible by the world microbatch token count"
        )
    end_step = num_iterations - 1 if config.end_step is None else config.end_step
    if config.start_step is None:
        # Integer ceil is exact even for token budgets larger than float precision.
        updates = (config.cooldown_tokens + total_batch_size - 1) // total_batch_size
        if updates < 2:
            raise ValueError(
                "conditioning_cooldown needs at least two updates for distinct endpoints"
            )
        start_step = end_step - updates + 1
    else:
        start_step = config.start_step
    if not 0 <= start_step < end_step < num_iterations:
        raise ValueError(
            "conditioning_cooldown requires 0 <= start_step < end_step < num_iterations; "
            f"resolved [{start_step}, {end_step}] for {num_iterations} updates. "
            "The token window must fit the horizon; set explicit steps or a smaller cooldown_tokens."
        )
    return ConditioningCooldown(
        start_fraction=config.start_fraction,
        end_fraction=config.end_fraction,
        start_step=start_step,
        end_step=end_step,
        granularity=config.granularity,
        seed=seed,
        world_size=world_size,
        device_batch_size=device_batch_size,
        max_seq_len=max_seq_len,
        total_batch_size=total_batch_size,
        num_iterations=num_iterations,
        device_type=device_type,
    )


def validate_conditioning_cooldown_resume(
    cooldown: ConditioningCooldown | None,
    meta: Mapping[str, Any],
) -> None:
    """Require unchanged resolved semantics on resume, or a pre-cooldown legacy base."""
    saved_raw = meta.get("conditioning_cooldown")
    user_config = meta.get("user_config", {})
    if not isinstance(user_config, Mapping):
        raise ValueError("conditioning_cooldown resume requires mapping user_config")
    saved_config = parse_conditioning_cooldown(user_config.get("conditioning_cooldown"))
    if saved_raw is None and saved_config is not None:
        raise ValueError("active conditioning_cooldown checkpoint is missing resolved metadata")
    if saved_raw is not None and saved_config is None:
        raise ValueError(
            "conditioning_cooldown checkpoint metadata requires an active saved config"
        )
    if saved_raw is None and cooldown is None:
        return
    step = _integer(meta.get("step"), "checkpoint step", minimum=0)
    if saved_raw is not None:
        saved = ConditioningCooldown.from_metadata(saved_raw)
        for source in (meta, user_config):
            for name in (
                "seed",
                "device_batch_size",
                "max_seq_len",
                "total_batch_size",
                "num_iterations",
            ):
                if name not in source:
                    continue
                recorded_value = _integer(source[name], f"saved {name}")
                if (
                    source is user_config
                    and name in ("total_batch_size", "num_iterations")
                    and recorded_value == -1
                ):
                    continue
                if recorded_value != getattr(saved, name):
                    raise ValueError(
                        f"conditioning_cooldown saved {name} disagrees with resolved metadata"
                    )
        saved_device_type = user_config.get("device_type", "")
        if type(saved_device_type) is not str or saved_device_type not in ("", saved.device_type):
            raise ValueError(
                "conditioning_cooldown saved device_type disagrees with resolved metadata"
            )
        dataloader_state = meta.get("dataloader_state_dict")
        if isinstance(dataloader_state, Mapping) and "ddp_world_size" in dataloader_state:
            if (
                _integer(dataloader_state["ddp_world_size"], "saved ddp_world_size", minimum=1)
                != saved.world_size
            ):
                raise ValueError(
                    "conditioning_cooldown saved world_size disagrees with resolved metadata"
                )
        recorded = saved.resolve_config(saved_config)
        if recorded != saved:
            raise ValueError("conditioning_cooldown saved config disagrees with resolved metadata")
        if cooldown != saved:
            raise ValueError(
                "conditioning_cooldown resume must preserve the resolved schedule, seed, and execution geometry"
            )
    elif cooldown is not None and step > cooldown.start_step:
        raise ValueError(
            "cannot start conditioning_cooldown from a legacy checkpoint after start_step"
        )
    if cooldown is not None and step >= cooldown.num_iterations:
        raise ValueError("conditioning_cooldown checkpoint has no remaining training updates")


def apply_conditioning_cooldown(
    enc_mask: t.Tensor,
    seg_doc_id: t.Tensor,
    *,
    fraction: float,
    granularity: str,
    seed: int,
    step: int,
    rank: int,
    micro_step: int,
    splice_mask: t.Tensor | None = None,
) -> t.Tensor:
    """Mask units independently, spending the keep budget on splice junctions first.

    Priority counts real segments (or unique row-local documents), including ones
    whose selected conditioning is naturally empty. It changes only retention
    probabilities: existing empty/padded mask entries can never become active.
    """
    fraction = _fraction(fraction, "fraction")
    granularity = _granularity(granularity)
    _integer(seed, "seed")
    _integer(step, "step", minimum=0)
    _integer(rank, "rank", minimum=0)
    _integer(micro_step, "micro_step", minimum=0)
    if enc_mask.dtype != t.bool or enc_mask.ndim != 3 or min(enc_mask.shape) < 1:
        raise ValueError("conditioning_cooldown enc_mask must be a nonempty bool (B, S, F) tensor")
    if (
        seg_doc_id.dtype != t.int64
        or seg_doc_id.shape != enc_mask.shape[:2]
        or seg_doc_id.device != enc_mask.device
    ):
        raise ValueError("conditioning_cooldown seg_doc_id must be int64 (B, S) on the mask device")
    if splice_mask is not None and (
        splice_mask.dtype != t.bool
        or splice_mask.shape != enc_mask.shape[:2]
        or splice_mask.device != enc_mask.device
    ):
        raise ValueError("conditioning_cooldown splice_mask must be bool (B, S) on the mask device")
    if fraction == 1.0:
        return enc_mask
    if fraction == 0.0:
        return t.zeros_like(enc_mask)
    key = f"conditioning_cooldown_v1:{seed}:{step}:{rank}:{micro_step}".encode("ascii")
    random_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    # MPS has no per-device Generator implementation; the small draw is CPU-owned.
    random_device = enc_mask.device if enc_mask.device.type != "mps" else t.device("cpu")
    generator = t.Generator(device=random_device).manual_seed(random_seed)
    random = t.rand(enc_mask.shape[:2], dtype=t.float32, device=random_device, generator=generator)
    if random.device != enc_mask.device:
        random = random.to(enc_mask.device)
    if granularity == "document":
        if bool(((seg_doc_id < -1) | (seg_doc_id >= enc_mask.shape[1])).any()):
            raise ValueError(
                "conditioning_cooldown document IDs must be -1 padding or row-local IDs in [0, S)"
            )
    valid = seg_doc_id >= 0
    protected = None if splice_mask is None else splice_mask & valid
    if protected is None or not bool(protected.any()):
        # Preserve the exact original scalar-threshold calculation and draw
        # shape when the microbatch has no real splice junctions.
        drop = random < (1.0 - fraction)
    else:
        if granularity == "document":
            counts = t.zeros_like(seg_doc_id)
            counts.scatter_add_(1, seg_doc_id.clamp_min(0), valid.long())
            unit_valid = counts > 0
            protected_counts = t.zeros_like(seg_doc_id)
            protected_counts.scatter_add_(1, seg_doc_id.clamp_min(0), protected.long())
            unit_protected = protected_counts > 0
        else:
            unit_valid, unit_protected = valid, protected
        protected_count = unit_protected.sum(dtype=t.float32)
        unprotected_count = (unit_valid & ~unit_protected).sum(dtype=t.float32)
        retained = fraction * (protected_count + unprotected_count)
        keep_protected = (retained / protected_count.clamp_min(1)).clamp(max=1)
        keep_unprotected = ((retained - protected_count) / unprotected_count.clamp_min(1)).clamp(
            0, 1
        )
        keep = t.where(unit_protected, keep_protected, keep_unprotected)
        # With only one real tier, use the identical scalar threshold as the
        # original algorithm; multiply/divide/subtract can move it by one ULP.
        drop_probability = t.where(unprotected_count == 0, 1.0 - fraction, 1.0 - keep)
        drop = random < drop_probability
    if granularity == "document":
        drop = drop.gather(1, seg_doc_id.clamp_min(0))
    return enc_mask & ~drop.unsqueeze(-1)
