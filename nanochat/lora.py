"""PyTorch utilities for applying and managing LoRA adapters."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.lora_config import LoRAConfig


class LoRALinear(nn.Module):
    """Wrap a ``nn.Linear`` module with trainable LoRA adapters."""

    def __init__(self, base: nn.Linear, config: LoRAConfig):
        super().__init__()
        if config.rank <= 0:
            raise ValueError("LoRA rank must be positive when enabling adapters")
        self.base = base
        self.rank = config.rank
        self.scaling = config.alpha / config.rank
        self.dropout_prob = config.dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.merged = False
        in_features = base.in_features
        out_features = base.out_features
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        if config.seed is not None:
            torch.manual_seed(config.seed)
        self.lora_a = nn.Parameter(torch.empty(self.rank, in_features, **factory_kwargs))
        self.lora_b = nn.Parameter(torch.empty(out_features, self.rank, **factory_kwargs))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        base.weight.requires_grad = False
        if base.bias is not None:
            base.bias.requires_grad = False
        self.lora_a._nanochat_lora_param = True  # type: ignore[attr-defined]
        self.lora_b._nanochat_lora_param = True  # type: ignore[attr-defined]

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scaling={self.scaling}, merged={self.merged}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        if not self.merged:
            adapted = F.linear(self.dropout(x), self.lora_a)
            adapted = F.linear(adapted, self.lora_b) * self.scaling
            result = result + adapted
        return result

    def merge(self) -> None:
        if self.merged:
            return
        delta = torch.matmul(self.lora_b, self.lora_a) * self.scaling
        self.base.weight.data.add_(delta)
        self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            return
        delta = torch.matmul(self.lora_b, self.lora_a) * self.scaling
        self.base.weight.data.sub_(delta)
        self.merged = False

    def lora_state_dict(self) -> Dict[str, torch.Tensor]:
        return {"lora_a": self.lora_a, "lora_b": self.lora_b}

    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.lora_a.data.copy_(state_dict["lora_a"])
        self.lora_b.data.copy_(state_dict["lora_b"])


def _iter_named_linears(
    module: nn.Module, prefix: str = ""
) -> Iterator[Tuple[str, nn.Module, str, nn.Linear]]:
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            yield full_name, module, name, child
        else:
            yield from _iter_named_linears(child, full_name)


def inject_lora_adapters(model: nn.Module, config: LoRAConfig) -> List[str]:
    """Replace matching linear layers with ``LoRALinear`` wrappers."""

    if not config.is_enabled():
        return []
    replaced: List[str] = []
    for name, parent, attr_name, linear in _iter_named_linears(model):
        if config.matches(name):
            setattr(parent, attr_name, LoRALinear(linear, config))
            replaced.append(name)
    return replaced


def iter_lora_modules(module: nn.Module) -> Iterator[Tuple[str, LoRALinear]]:
    for name, child in module.named_modules():
        if isinstance(child, LoRALinear):
            yield name, child


def gather_lora_parameters(module: nn.Module) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for _, lora_module in iter_lora_modules(module):
        params.extend([lora_module.lora_a, lora_module.lora_b])
    return params


def freeze_non_lora_parameters(module: nn.Module, config: LoRAConfig) -> None:
    for name, param in module.named_parameters():
        if getattr(param, "_nanochat_lora_param", False):
            continue
        if config.keep_trainable(name):
            continue
        param.requires_grad = False


def merge_lora_weights(module: nn.Module) -> None:
    for _, lora_module in iter_lora_modules(module):
        lora_module.merge()


def unmerge_lora_weights(module: nn.Module) -> None:
    for _, lora_module in iter_lora_modules(module):
        lora_module.unmerge()


def lora_state_dict(module: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    state: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, lora_module in iter_lora_modules(module):
        state[name] = lora_module.lora_state_dict()
    return state


def load_lora_state_dict(module: nn.Module, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
    for name, lora_module in iter_lora_modules(module):
        if name in state:
            lora_module.load_lora_state_dict(state[name])


def ensure_lora_seed(config: LoRAConfig) -> None:
    if config.seed is not None:
        torch.manual_seed(config.seed)
