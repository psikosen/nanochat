"""Inference engine for MLX models."""
from __future__ import annotations

import signal
import warnings
from collections import deque
from contextlib import contextmanager
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None  # type: ignore

from typing import Any

from .gpt import KVCache, MLXGPT


def _require_mlx():
    if mx is None:  # pragma: no cover
        raise ImportError(
            "The MLX backend requires the 'mlx' package. Install nanochat with the '[mlx]' extra."
        )


@contextmanager
def timeout(duration: int, formula: str):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)


def eval_with_timeout(formula: str, max_time: int = 3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula)
    except Exception:
        signal.alarm(0)
        return None


def use_calculator(expr: str) -> Optional[str]:
    expr = expr.replace(",", "")
    if any(ch not in "0123456789*+-/.() " for ch in expr):
        return None
    if "**" in expr:
        return None
    result = eval_with_timeout(expr)
    if result is None:
        return None
    return str(result)


def sample_next_token(
    logits: mx.array,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> List[int]:
    logits_np = mx.to_numpy(logits)
    if temperature == 0:
        return np.argmax(logits_np, axis=-1).astype(int).tolist()
    if top_k is not None:
        k = min(top_k, logits_np.shape[-1])
        indices = np.argpartition(logits_np, -k, axis=-1)[..., -k:]
        masked = np.full_like(logits_np, -np.inf)
        np.put_along_axis(masked, indices, np.take_along_axis(logits_np, indices, axis=-1), axis=-1)
        logits_np = masked
    logits_np = logits_np / max(temperature, 1e-8)
    logits_np = logits_np - logits_np.max(axis=-1, keepdims=True)
    probs = np.exp(logits_np)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    next_ids = [int(rng.choice(probs.shape[-1], p=row)) for row in probs]
    return next_ids


class RowState:
    def __init__(self, current_tokens: Optional[List[int]] = None):
        self.current_tokens = current_tokens or []
        self.forced_tokens: deque[int] = deque()
        self.in_python_block = False
        self.python_expr_tokens: List[int] = []
        self.completed = False


class MLXEngine:
    def __init__(self, model: MLXGPT, tokenizer: Any):
        _require_mlx()
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        tokens: Sequence[int],
        num_samples: int = 1,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: int = 42,
    ) -> Iterable[Tuple[List[int], List[int]]]:
        assert isinstance(tokens, Sequence) and isinstance(tokens[0], int)
        rng = np.random.default_rng(seed)
        model_cfg = self.model.config
        kv_prefill = KVCache(model_cfg.n_layer)
        ids = mx.array([list(tokens)], dtype=mx.int32)
        logits = self.model(ids, kv_cache=kv_prefill)
        logits = logits[:, -1, :]
        sampled = sample_next_token(logits, rng, temperature, top_k)
        kv_decode = kv_prefill.clone_for_batch(num_samples)
        row_states = [RowState(list(tokens)) for _ in range(num_samples)]
        num_generated = 0
        first_iteration = True
        get_special = self.tokenizer.encode_special
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break
            if first_iteration:
                sampled_tokens = [sampled[0]] * num_samples
                first_iteration = False
            else:
                ids = mx.array(sampled_tokens, dtype=mx.int32)[:, None]
                logits = self.model(ids, kv_cache=kv_decode)
                logits = logits[:, -1, :]
                sampled_tokens = sample_next_token(logits, rng, temperature, top_k)
            token_column: List[int] = []
            token_masks: List[int] = []
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                state.current_tokens.append(next_token)
                if next_token in (assistant_end, bos):
                    state.completed = True
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(result)
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)
            yield token_column, token_masks
            num_generated += 1
            sampled_tokens = token_column

    def generate_batch(self, tokens: Sequence[int], num_samples: int = 1, **kwargs):
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [list(tokens) for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token in (assistant_end, bos):
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break
        return results, masks
