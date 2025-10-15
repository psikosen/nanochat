"""GPT model implemented with Apple's MLX for Apple Silicon inference/training."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - import guarded for non-mac environments
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # pragma: no cover
    mx = None  # type: ignore
    nn = None  # type: ignore

MLX_AVAILABLE = mx is not None and nn is not None


@dataclass
class GPTConfig:
    """Configuration shared with the PyTorch implementation."""

    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768


def _require_mlx():
    if not MLX_AVAILABLE:  # pragma: no cover - runtime guard
        raise ImportError(
            "The MLX backend requires the 'mlx' package. Install nanochat with the "
            "'[mlx]' extra on Apple Silicon macOS to enable it."
        )


if not MLX_AVAILABLE:

    def rms_norm(*args, **kwargs):  # pragma: no cover - stub
        _require_mlx()

    def apply_rotary_emb(*args, **kwargs):  # pragma: no cover - stub
        _require_mlx()

    def repeat_kv(*args, **kwargs):  # pragma: no cover - stub
        _require_mlx()

    class KVCache:  # pragma: no cover - stub
        def __init__(self, *args, **kwargs):
            _require_mlx()

    class MLXGPT:  # pragma: no cover - stub
        def __init__(self, *args, **kwargs):
            _require_mlx()

        def generate(self, *args, **kwargs):
            _require_mlx()

        def load_state_dict(self, *args, **kwargs):
            _require_mlx()

        def estimate_flops(self) -> float:
            _require_mlx()

else:

    def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
        denom = mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)
        return x * denom

    def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return mx.concatenate([y1, y2], axis=-1)

    def repeat_kv(x: mx.array, repeats: int) -> mx.array:
        if repeats == 1:
            return x
        b, h, t, d = x.shape
        x = mx.reshape(x, (b, h, 1, t, d))
        x = mx.broadcast_to(x, (b, h, repeats, t, d))
        return mx.reshape(x, (b, h * repeats, t, d))

    class CausalSelfAttention(nn.Module):  # type: ignore[misc]
        def __init__(self, config: GPTConfig, layer_idx: int):
            super().__init__()
            self.layer_idx = layer_idx
            self.n_head = config.n_head
            self.n_kv_head = config.n_kv_head
            self.n_embd = config.n_embd
            self.head_dim = self.n_embd // self.n_head
            assert self.n_embd % self.n_head == 0
            assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
            self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
            self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        def __call__(
            self, x: mx.array, cos_sin: Tuple[mx.array, mx.array], kv_cache: Optional["KVCache"]
        ) -> mx.array:
            b, t, _ = x.shape
            q = self.c_q(x)
            k = self.c_k(x)
            v = self.c_v(x)
            q = mx.reshape(q, (b, t, self.n_head, self.head_dim))
            k = mx.reshape(k, (b, t, self.n_kv_head, self.head_dim))
            v = mx.reshape(v, (b, t, self.n_kv_head, self.head_dim))
            cos, sin = cos_sin
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
            q = rms_norm(q)
            k = rms_norm(k)
            q = mx.transpose(q, (0, 2, 1, 3))
            k = mx.transpose(k, (0, 2, 1, 3))
            v = mx.transpose(v, (0, 2, 1, 3))
            if kv_cache is not None:
                k, v = kv_cache.insert_kv(self.layer_idx, k, v)
            n_rep = self.n_head // self.n_kv_head
            k = repeat_kv(k, n_rep)
            v = repeat_kv(v, n_rep)
            tq = q.shape[2]
            tk = k.shape[2]
            attn = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) / math.sqrt(self.head_dim)
            if kv_cache is None or tq == tk:
                mask = mx.tril(mx.ones((tq, tk), dtype=attn.dtype))
                attn = attn + (mask - 1) * 1e6
            elif tq == 1:
                pass
            else:
                prefix = tk - tq
                prefix_mask = mx.concatenate(
                    [
                        mx.ones((tq, prefix), dtype=attn.dtype),
                        mx.tril(mx.ones((tq, tq), dtype=attn.dtype)),
                    ],
                    axis=1,
                )
                attn = attn + (prefix_mask - 1) * 1e6
            attn = mx.softmax(attn, axis=-1)
            y = mx.matmul(attn, v)
            y = mx.transpose(y, (0, 2, 1, 3))
            y = mx.reshape(y, (b, t, self.n_embd))
            return self.c_proj(y)

    class MLP(nn.Module):  # type: ignore[misc]
        def __init__(self, config: GPTConfig):
            super().__init__()
            hidden = 4 * config.n_embd
            self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
            self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

        def __call__(self, x: mx.array) -> mx.array:
            x = self.c_fc(x)
            x = mx.square(mx.maximum(x, 0))
            return self.c_proj(x)

    class Block(nn.Module):  # type: ignore[misc]
        def __init__(self, config: GPTConfig, layer_idx: int):
            super().__init__()
            self.attn = CausalSelfAttention(config, layer_idx)
            self.mlp = MLP(config)

        def __call__(self, x: mx.array, cos_sin: Tuple[mx.array, mx.array], kv_cache: Optional["KVCache"]) -> mx.array:
            x = x + self.attn(rms_norm(x), cos_sin, kv_cache)
            x = x + self.mlp(rms_norm(x))
            return x

    class KVCache:
        def __init__(self, num_layers: int):
            self.num_layers = num_layers
            self.cache: List[Tuple[Optional[mx.array], Optional[mx.array]]] = [
                (None, None) for _ in range(num_layers)
            ]
            self.pos = 0

        def reset(self) -> None:
            self.cache = [(None, None) for _ in range(self.num_layers)]
            self.pos = 0

        def get_pos(self) -> int:
            return self.pos

        def clone_for_batch(self, batch_size: int) -> "KVCache":
            new_cache = KVCache(self.num_layers)
            new_cache.pos = self.pos
            new_cache.cache = []
            for k_layer, v_layer in self.cache:
                if k_layer is None:
                    new_cache.cache.append((None, None))
                    continue
                shape = (batch_size,) + k_layer.shape[1:]
                new_k = mx.broadcast_to(k_layer, shape)
                new_v = mx.broadcast_to(v_layer, shape)
                new_cache.cache.append((new_k, new_v))
            return new_cache

        def insert_kv(self, layer_idx: int, k: mx.array, v: mx.array) -> Tuple[mx.array, mx.array]:
            k_layer, v_layer = self.cache[layer_idx]
            if k_layer is None:
                new_k, new_v = k, v
            else:
                new_k = mx.concatenate([k_layer, k], axis=2)
                new_v = mx.concatenate([v_layer, v], axis=2)
            self.cache[layer_idx] = (new_k, new_v)
            if layer_idx == self.num_layers - 1:
                self.pos += k.shape[2]
            return new_k, new_v

    class MLXGPT(nn.Module):  # type: ignore[misc]
        def __init__(self, config: GPTConfig):
            _require_mlx()
            super().__init__()
            self.config = config
            self.wte = nn.Embedding(config.vocab_size, config.n_embd)
            self.h: List[Block] = [Block(config, i) for i in range(config.n_layer)]
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.rotary_seq_len = config.sequence_len * 10
            head_dim = config.n_embd // config.n_head
            cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)

        def register_buffer(self, name: str, value: mx.array) -> None:
            setattr(self, name, value)

        def _precompute_rotary_embeddings(
            self, seq_len: int, head_dim: int, base: int = 10000, device: Optional[mx.Device] = None
        ) -> Tuple[mx.array, mx.array]:
            if device is None:
                device = mx.default_device()
            channel_range = mx.arange(0, head_dim, 2, dtype=mx.float32)
            inv_freq = 1.0 / (base ** (channel_range / head_dim))
            positions = mx.arange(seq_len, dtype=mx.float32)
            freqs = mx.outer(positions, inv_freq)
            cos = mx.cos(freqs)
            sin = mx.sin(freqs)
            cos = mx.astype(cos, mx.bfloat16)
            sin = mx.astype(sin, mx.bfloat16)
            cos = mx.reshape(cos, (1, seq_len, 1, head_dim // 2))
            sin = mx.reshape(sin, (1, seq_len, 1, head_dim // 2))
            return cos, sin

        def get_device(self) -> str:
            return str(mx.default_device())

        def parameters(self) -> Iterable[mx.array]:  # type: ignore[override]
            for param in self.wte.parameters():
                yield param
            for block in self.h:
                yield from block.parameters()
            yield from self.lm_head.parameters()

        def _cos_sin(self, t: int, offset: int) -> Tuple[mx.array, mx.array]:
            cos = self.cos[:, offset : offset + t]
            sin = self.sin[:, offset : offset + t]
            return cos, sin

        def __call__(
            self,
            idx: mx.array,
            targets: Optional[mx.array] = None,
            kv_cache: Optional[KVCache] = None,
            loss_reduction: str = "mean",
        ) -> mx.array:
            b, t = idx.shape
            offset = 0 if kv_cache is None else kv_cache.get_pos()
            cos_sin = self._cos_sin(t, offset)
            x = self.wte(idx)
            x = rms_norm(x)
            for block in self.h:
                x = block(x, cos_sin, kv_cache)
            x = rms_norm(x)
            logits = self.lm_head(x)
            softcap = 15.0
            logits = softcap * mx.tanh(logits / softcap)
            if targets is None:
                return logits
            logits = mx.astype(logits, mx.float32)
            flat_logits = mx.reshape(logits, (-1, logits.shape[-1]))
            flat_targets = mx.reshape(targets, (-1,))
            mask = flat_targets != -1
            log_probs = mx.log_softmax(flat_logits, axis=-1)
            safe_targets = mx.where(mask, flat_targets, mx.zeros_like(flat_targets))
            gather_idx = mx.expand_dims(safe_targets, axis=-1)
            selected = mx.take_along_axis(log_probs, gather_idx, axis=-1)
            selected = mx.squeeze(selected, axis=-1)
            selected = mx.where(mask, selected, mx.zeros_like(selected))
            loss = -selected
            if loss_reduction == "sum":
                return mx.sum(loss)
            denom = mx.sum(mask)
            denom = mx.maximum(denom, mx.array(1, dtype=denom.dtype))
            return mx.sum(loss) / denom

        def estimate_flops(self) -> float:
            nparams = sum(mx.size(p) for p in self.parameters())
            embedding_params = mx.size(self.wte.weight)
            l = self.config.n_layer
            h = self.config.n_head
            q = self.config.n_embd // self.config.n_head
            t = self.config.sequence_len
            return float(6 * (nparams - embedding_params) + 12 * l * h * q * t)

        def load_state_dict(self, weights: dict[str, mx.array]) -> None:
            for name, value in weights.items():
                target = self
                parts = name.split(".")
                for part in parts[:-1]:
                    if part == "blocks":
                        continue
                    if part.startswith("layer_"):
                        idx = int(part.split("_")[-1])
                        target = self.h[idx]
                    else:
                        target = getattr(target, part)
                param_name = parts[-1]
                setattr(target, param_name, mx.array(value))

        def generate(
            self,
            tokens: List[int],
            max_tokens: int,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            seed: int = 42,
        ) -> Iterable[int]:
            rng = np.random.default_rng(seed)
            ids = mx.array(tokens, dtype=mx.int32)[None, :]
            kv_cache = KVCache(len(self.h))
            logits = self(ids, kv_cache=kv_cache)
            logits = logits[:, -1, :]
            token = self._sample_token(logits, rng, temperature, top_k)
            ids = mx.concatenate([ids, mx.array([[token]], dtype=mx.int32)], axis=1)
            yield token
            for _ in range(max_tokens - 1):
                next_input = mx.array([[token]], dtype=mx.int32)
                logits = self(next_input, kv_cache=kv_cache)
                logits = logits[:, -1, :]
                token = self._sample_token(logits, rng, temperature, top_k)
                ids = mx.concatenate([ids, mx.array([[token]], dtype=mx.int32)], axis=1)
                yield token

        def _sample_token(
            self,
            logits: mx.array,
            rng: np.random.Generator,
            temperature: float,
            top_k: Optional[int],
        ) -> int:
            logits_np = mx.to_numpy(logits)[0]
            if top_k is not None:
                k = min(top_k, logits_np.shape[-1])
                top_indices = np.argpartition(logits_np, -k)[-k:]
                mask = np.full_like(logits_np, -np.inf)
                mask[top_indices] = logits_np[top_indices]
                logits_np = mask
            if temperature > 0:
                logits_np = logits_np / temperature
                probs = np.exp(logits_np - np.max(logits_np))
                probs = probs / np.sum(probs)
                token = int(rng.choice(len(probs), p=probs))
            else:
                token = int(np.argmax(logits_np))
            return token
