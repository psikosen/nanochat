"""Configuration objects shared between PyTorch and MLX LoRA adapters."""
from __future__ import annotations

"""Shared configuration for LoRA adapters across backends."""

from dataclasses import dataclass, field
from typing import Iterable, Tuple


def _default_targets() -> Tuple[str, ...]:
    """Default module suffixes where LoRA adapters are injected."""
    return (
        "attn.c_q",
        "attn.c_k",
        "attn.c_v",
        "attn.c_proj",
        "mlp.c_fc",
        "mlp.c_proj",
    )


def _default_track() -> Tuple[str, ...]:
    return ("transformer.wte", "lm_head")


@dataclass
class LoRAConfig:
    """Hyperparameters controlling LoRA adapter injection.

    Attributes
    ----------
    rank:
        Rank of the low-rank adapter matrices. Set to 0 to disable adapters.
    alpha:
        Scaling factor applied to the adapter output. Typical values mirror the
        rank (e.g. 16/32).
    dropout:
        Dropout probability applied before the adapter projection. Defaults to 0
        because most adapter fine-tunes omit dropout to reduce variance.
    target_modules:
        Collection of module suffixes that should receive adapters. The matcher
        performs an ``endswith`` comparison against the fully qualified module
        names inside the model.
    trainable_modules:
        Additional module prefixes that remain trainable during adapter
        fine-tuning (e.g. to unfreeze embeddings or the LM head). Prefix matches
        are used.
    lora_lr:
        Optimiser learning rate for LoRA parameters.
    lora_weight_decay:
        Weight decay applied to LoRA parameters.
    betas:
        Adam betas used for the LoRA optimiser.
    eps:
        Adam epsilon used for numerical stability.
    merge_weights:
        Whether adapters should be merged into the base weights when saving.
    adapter_name:
        Identifier used when persisting metadata and registry entries.
    seed:
        Optional RNG seed for deterministic adapter initialisation.
    """

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: Tuple[str, ...] = field(default_factory=_default_targets)
    trainable_modules: Tuple[str, ...] = field(default_factory=_default_track)
    lora_lr: float = 5e-4
    lora_weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    merge_weights: bool = False
    adapter_name: str = "default"
    seed: int | None = None

    def is_enabled(self) -> bool:
        return self.rank > 0

    def matches(self, module_name: str) -> bool:
        return any(module_name.endswith(target) for target in self.target_modules)

    def keep_trainable(self, param_name: str) -> bool:
        return any(param_name.startswith(prefix) for prefix in self.trainable_modules)

    def to_metadata(self) -> dict:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "trainable_modules": list(self.trainable_modules),
            "lora_lr": self.lora_lr,
            "lora_weight_decay": self.lora_weight_decay,
            "betas": list(self.betas),
            "eps": self.eps,
            "merge_weights": self.merge_weights,
            "adapter_name": self.adapter_name,
            "seed": self.seed,
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "LoRAConfig":
        metadata = dict(metadata)
        metadata.setdefault("target_modules", _default_targets())
        metadata.setdefault("trainable_modules", _default_track())
        metadata.setdefault("betas", (0.9, 0.999))
        return cls(
            rank=metadata["rank"],
            alpha=metadata["alpha"],
            dropout=metadata.get("dropout", 0.0),
            target_modules=tuple(metadata.get("target_modules", _default_targets())),
            trainable_modules=tuple(metadata.get("trainable_modules", _default_track())),
            lora_lr=metadata.get("lora_lr", 5e-4),
            lora_weight_decay=metadata.get("lora_weight_decay", 0.0),
            betas=tuple(metadata.get("betas", (0.9, 0.999))),
            eps=metadata.get("eps", 1e-8),
            merge_weights=metadata.get("merge_weights", False),
            adapter_name=metadata.get("adapter_name", "default"),
            seed=metadata.get("seed"),
        )


def ensure_tuple(value: Iterable[str] | str) -> Tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(value)
