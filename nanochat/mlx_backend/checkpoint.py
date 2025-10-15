"""Utilities to bridge nanochat checkpoints with MLX weights."""
from __future__ import annotations

import os
from typing import Dict

import numpy as np

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from .gpt import MLXGPT


def _map_key(torch_key: str) -> str:
    if torch_key.startswith("transformer.wte."):
        return torch_key.replace("transformer.wte", "wte", 1)
    if torch_key.startswith("transformer.h."):
        parts = torch_key.split(".")
        layer = parts[2]
        remainder = ".".join(parts[3:])
        return f"blocks.layer_{layer}.{remainder}"
    return torch_key


def convert_torch_state_dict(state_dict: Dict[str, "torch.Tensor"]) -> Dict[str, np.ndarray]:
    if torch is None:  # pragma: no cover
        raise ImportError("Torch is required to convert checkpoints to MLX format.")
    mlx_state: Dict[str, np.ndarray] = {}
    for key, tensor in state_dict.items():
        new_key = _map_key(key)
        mlx_state[new_key] = tensor.detach().cpu().numpy()
    return mlx_state


def save_mlx_weights(mlx_state: Dict[str, np.ndarray], path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savez(path, **mlx_state)


def load_mlx_weights(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def load_weights_into_model(model: MLXGPT, weights_path: str) -> None:
    state = load_mlx_weights(weights_path)
    model.load_state_dict(state)
