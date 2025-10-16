"""Registry for managing LoRA adapters on disk."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

from nanochat.gpt import GPT
from nanochat.lora_config import LoRAConfig


@dataclass
class AdapterMetadata:
    name: str
    config: LoRAConfig
    metrics: Dict[str, float]
    created_at: str
    description: str = ""
    path: Optional[Path] = None

    @classmethod
    def from_json(cls, payload: Dict[str, object], path: Path) -> "AdapterMetadata":
        return cls(
            name=str(payload["name"]),
            config=LoRAConfig.from_metadata(payload["config"]),
            metrics={k: float(v) for k, v in payload.get("metrics", {}).items()},
            created_at=str(payload.get("created_at", datetime.utcnow().isoformat())),
            description=str(payload.get("description", "")),
            path=path,
        )

    def to_json(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "config": self.config.to_metadata(),
            "metrics": self.metrics,
            "created_at": self.created_at,
            "description": self.description,
        }


class AdapterRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _adapter_dir(self, name: str) -> Path:
        return self.root / name

    def list_adapters(self) -> List[AdapterMetadata]:
        adapters: List[AdapterMetadata] = []
        for entry in sorted(self.root.glob("*/metadata.json")):
            with entry.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            adapters.append(AdapterMetadata.from_json(payload, entry.parent))
        return adapters

    def load(self, model: GPT, name: str, *, capture_base: bool = True) -> tuple[AdapterMetadata, Dict[str, Dict[str, torch.Tensor]]]:
        directory = self._adapter_dir(name)
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Adapter '{name}' metadata not found at {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metadata = AdapterMetadata.from_json(payload, directory)
        if not model.has_lora():
            model.enable_lora(metadata.config)
        base_state = model.get_lora_state_dict() if capture_base else {}
        state_path = directory / "adapter.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Adapter '{name}' state not found at {state_path}")
        state = torch.load(state_path, map_location=model.get_device())
        model.load_lora_state_dict(state)
        return metadata, base_state

    def save(
        self,
        model: GPT,
        metadata: AdapterMetadata,
        *,
        merge_weights: bool | None = None,
    ) -> AdapterMetadata:
        directory = self._adapter_dir(metadata.name)
        directory.mkdir(parents=True, exist_ok=True)
        merge = metadata.config.merge_weights if merge_weights is None else merge_weights
        if merge:
            model.merge_lora()
        try:
            state = model.get_lora_state_dict()
            state_path = directory / "adapter.pt"
            torch.save(state, state_path)
            payload = metadata.to_json()
            if not payload.get("created_at"):
                payload["created_at"] = datetime.utcnow().isoformat()
            metadata_path = directory / "metadata.json"
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        finally:
            if merge:
                model.unmerge_lora()
        metadata.path = directory
        return metadata

    def delete(self, name: str) -> None:
        directory = self._adapter_dir(name)
        if not directory.exists():
            return
        for path in directory.glob("*"):
            path.unlink()
        directory.rmdir()


__all__ = ["AdapterRegistry", "AdapterMetadata"]
