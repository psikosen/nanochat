"""Lightweight LoRA adapter training utilities."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch


@dataclass
class LoRATrainingConfig:
    epochs: int = 1
    batch_size: int = 4
    max_length: int = 512
    grad_accumulation: int = 1
    clip_grad_norm: float | None = 1.0
    shuffle: bool = True


def _batch(iterable: Sequence, size: int) -> Iterable[Sequence]:
    for idx in range(0, len(iterable), size):
        yield iterable[idx : idx + size]


def _tokenise_example(tokenizer, prompt: str, completion: str, max_length: int) -> Dict[str, List[int]]:
    bos = tokenizer.get_bos_token_id()
    prompt_ids = tokenizer.encode(prompt, prepend=bos)
    completion_ids = tokenizer.encode(completion)
    input_ids = prompt_ids + completion_ids
    target_ids = [-1] * len(prompt_ids) + completion_ids
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        target_ids = target_ids[:max_length]
    return {"input_ids": input_ids, "target_ids": target_ids}


def train_lora_adapters(model, tokenizer, samples, config: LoRATrainingConfig) -> Dict[str, float]:
    if not samples:
        raise ValueError("No samples provided for adapter training")
    device = model.get_device()
    model.train()
    optimizers = model.setup_optimizers()
    for optimizer in optimizers:
        optimizer.zero_grad()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("Model has no trainable parameters for LoRA training")
    total_loss = 0.0
    total_steps = 0
    examples = list(samples)
    for epoch in range(config.epochs):
        if config.shuffle:
            random.shuffle(examples)
        for batch in _batch(examples, config.batch_size):
            losses = []
            for sample in batch:
                tokens = _tokenise_example(tokenizer, sample.prompt, sample.response, config.max_length)
                input_tensor = torch.tensor(tokens["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
                target_tensor = torch.tensor(tokens["target_ids"], dtype=torch.long, device=device).unsqueeze(0)
                loss = model.forward(input_tensor, targets=target_tensor) / config.grad_accumulation
                losses.append(loss)
                loss.backward()
            if (total_steps + 1) % config.grad_accumulation == 0:
                if config.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(trainable_params, config.clip_grad_norm)
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
            batch_loss = torch.stack(losses).sum().item()
            total_loss += batch_loss
            total_steps += 1
    mean_loss = total_loss / max(total_steps, 1)
    return {"mean_loss": mean_loss, "updates": total_steps, "epochs": config.epochs}


__all__ = ["LoRATrainingConfig", "train_lora_adapters"]
