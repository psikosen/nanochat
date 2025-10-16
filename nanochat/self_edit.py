"""Self-edit orchestration utilities with structured logging."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from nanochat.fewshot import ArcFewShotPromptBuilder, FewShotPrompt


@dataclass
class SelfEditConfig:
    """Configuration for the self-edit loop."""

    num_demonstrations: int = 3
    samples_per_prompt: int = 3
    temperature: float = 0.7
    top_k: int | None = 50
    max_tokens: int = 256
    score_threshold: float = 0.5
    log_path: Path = Path("logs/self_edit.log")
    seed: int = 2025


@dataclass
class SelfEditSample:
    prompt: str
    response: str
    score: float
    metadata: Dict[str, object]


@dataclass
class SelfEditSummary:
    samples: List[SelfEditSample]
    accepted: List[SelfEditSample]
    rejected: List[SelfEditSample]
    prompt_metadata: Dict[str, object]

    @property
    def acceptance_rate(self) -> float:
        if not self.samples:
            return 0.0
        return len(self.accepted) / len(self.samples)


class SelfEditLogger:
    """Persistent structured logger following the canonical schema."""

    schema_fields = (
        "filename",
        "timestamp",
        "classname",
        "function",
        "system_section",
        "line_num",
        "error",
        "db_phase",
        "method",
        "message",
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")

    def log(
        self,
        classname: str,
        function: str,
        system_section: str,
        message: str,
        *,
        line_num: int = 0,
        error: str | None = None,
        db_phase: str = "none",
        method: str = "NONE",
        extra: Dict[str, object] | None = None,
    ) -> None:
        timestamp = datetime.utcnow().isoformat()
        entry = {
            "filename": str(self.path),
            "timestamp": timestamp,
            "classname": classname,
            "function": function,
            "system_section": system_section,
            "line_num": line_num,
            "error": error,
            "db_phase": db_phase,
            "method": method,
            "message": message,
        }
        if extra:
            entry.update(extra)
        self.handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        skepticism_line = (
            "[Continuous skepticism (Sherlock Protocol)] "
            "Could this change affect unexpected files/systems? {unexpected}; "
            "Any hidden dependencies or cascades? {dependencies}; "
            "What edge cases and failure modes are unhandled? {edges}; "
            "If stuck, work backward from the desired outcome."
        ).format(
            unexpected=extra.get("unexpected", "unknown") if extra else "unknown",
            dependencies=extra.get("dependencies", "unknown") if extra else "unknown",
            edges=extra.get("edges", "unknown") if extra else "unknown",
        )
        self.handle.write(skepticism_line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "SelfEditLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[override]
        self.close()


class SelfEditOrchestrator:
    """Run the self-edit loop for ARC-style prompts."""

    def __init__(
        self,
        prompt_builder: ArcFewShotPromptBuilder,
        generator: Callable[[str, SelfEditConfig], str],
        scorer: Callable[[Dict[str, object], str], float],
        logger: SelfEditLogger,
        config: SelfEditConfig,
    ) -> None:
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.scorer = scorer
        self.logger = logger
        self.config = config
        self.rng = random.Random(config.seed)

    def run(self, indices: Sequence[int]) -> List[SelfEditSummary]:
        summaries: List[SelfEditSummary] = []
        for idx in indices:
            prompt = self.prompt_builder.build_prompt(
                query_index=idx,
                num_demonstrations=self.config.num_demonstrations,
                seed=self.rng.randint(0, 2**31 - 1),
            )
            summary = self._process_prompt(idx, prompt)
            summaries.append(summary)
        return summaries

    def _process_prompt(self, idx: int, prompt: FewShotPrompt) -> SelfEditSummary:
        samples: List[SelfEditSample] = []
        accepted: List[SelfEditSample] = []
        rejected: List[SelfEditSample] = []
        for sample_id in range(self.config.samples_per_prompt):
            response = self.generator(prompt.formatted, self.config)
            score = self.scorer(prompt.query["prompt"], response)
            sample = SelfEditSample(
                prompt=prompt.formatted,
                response=response,
                score=score,
                metadata={
                    "sample_id": sample_id,
                    "query_index": idx,
                    "demo_indices": prompt.query["metadata"]["demo_indices"],
                },
            )
            samples.append(sample)
            if score >= self.config.score_threshold:
                accepted.append(sample)
                decision = "accepted"
            else:
                rejected.append(sample)
                decision = "rejected"
            self.logger.log(
                classname=self.__class__.__name__,
                function="_process_prompt",
                system_section="self_edit",
                message=f"Sample {sample_id} for query {idx} scored {score:.3f} ({decision})",
                extra={
                    "score": score,
                    "decision": decision,
                    "unexpected": score < self.config.score_threshold,
                    "dependencies": len(prompt.demos),
                    "edges": "low-score" if score < self.config.score_threshold else "",
                },
            )
        summary = SelfEditSummary(samples=samples, accepted=accepted, rejected=rejected, prompt_metadata=prompt.query["metadata"])
        self.logger.log(
            classname=self.__class__.__name__,
            function="_process_prompt",
            system_section="summary",
            message=f"Query {idx} acceptance rate {summary.acceptance_rate:.3f}",
            extra={
                "unexpected": summary.acceptance_rate < 0.2,
                "dependencies": len(prompt.demos),
                "edges": "insufficient_accepts" if not summary.accepted else "",
            },
        )
        return summary


def default_scorer(example: Dict[str, object], response: str) -> float:
    """Default scoring that checks if the response starts with the gold letter."""

    gold = str(example.get("answer", "")).strip()
    response = response.strip()
    if not gold:
        return 0.0
    return 1.0 if response.upper().startswith(gold.upper()) else 0.0


def greedy_generator(model) -> Callable[[str, SelfEditConfig], str]:
    """Wrap a model with a minimal text generation callable."""

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Model must expose a tokenizer attribute for greedy generation")

    def _generate(prompt: str, config: SelfEditConfig) -> str:
        tokens = tokenizer.encode(prompt)
        bos = tokenizer.get_bos_token_id()
        tokens = [bos] + tokens
        outputs = []
        for token in model.generate(tokens, max_tokens=config.max_tokens, temperature=config.temperature, top_k=config.top_k):
            outputs.append(token)
        return tokenizer.decode(outputs)

    return _generate


__all__ = [
    "SelfEditConfig",
    "SelfEditSample",
    "SelfEditSummary",
    "SelfEditLogger",
    "SelfEditOrchestrator",
    "default_scorer",
    "greedy_generator",
]
