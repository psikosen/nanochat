"""Few-shot prompt assembly and augmentation utilities."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from nanochat.lora_config import LoRAConfig  # re-export convenience


@dataclass
class FewShotDemo:
    prompt: str
    completion: str
    metadata: Dict[str, object]


@dataclass
class FewShotPrompt:
    instructions: str
    demos: List[FewShotDemo]
    query: Dict[str, object]
    formatted: str


class ArcAugmenter:
    """Apply lightweight augmentations to ARC-style multiple choice examples."""

    def __init__(self, shuffle_choices: bool = True, question_templates: Sequence[str] | None = None):
        self.shuffle_choices = shuffle_choices
        self.question_templates = list(question_templates or [
            "Consider the following science question and respond with the best option.",
            "Answer this exam-style question by selecting the single correct letter.",
            "Choose the option that best completes the statement.",
        ])

    def augment(self, example: Dict[str, object], rng: random.Random) -> Tuple[Dict[str, object], Dict[str, object]]:
        mutated = {
            "question": str(example["question"]).strip(),
            "choices": list(example["choices"]),
            "letters": list(example["letters"]),
            "answer": example.get("answer"),
        }
        metadata: Dict[str, object] = {}
        if self.shuffle_choices and len(mutated["choices"]) > 1:
            zipped = list(zip(mutated["letters"], mutated["choices"]))
            rng.shuffle(zipped)
            base_letters = [chr(ord("A") + i) for i in range(len(zipped))]
            correct_choice = None
            for i, (orig_letter, choice) in enumerate(zipped):
                if orig_letter == mutated["answer"]:
                    correct_choice = base_letters[i]
            mutated["choices"] = [choice for _, choice in zipped]
            mutated["letters"] = base_letters
            if mutated["answer"] is not None and correct_choice is not None:
                mutated["answer"] = correct_choice
            metadata["choice_permutation"] = {
                "original": [letter for letter, _ in zipped],
                "new": mutated["letters"],
            }
        if self.question_templates:
            template = rng.choice(self.question_templates)
            metadata["question_template"] = template
            mutated["question"] = f"{template} {mutated['question']}"
        return mutated, metadata


class ArcFewShotPromptBuilder:
    """Construct prompts for ARC-style datasets with augmentations."""

    def __init__(
        self,
        conversations: Sequence[Dict[str, object]],
        instructions: str,
        max_demonstrations: int = 4,
        augmentation: ArcAugmenter | None = None,
    ) -> None:
        if not conversations:
            raise ValueError("Conversations collection must be non-empty")
        self.conversations = conversations
        self.instructions = instructions.strip()
        self.max_demonstrations = max_demonstrations
        self.augmentation = augmentation or ArcAugmenter()

    def _normalise(self, conversation: Dict[str, object]) -> Dict[str, object]:
        for key in ("question", "choices", "letters"):
            if key not in conversation:
                raise KeyError(f"Conversation missing '{key}' field required for few-shot prompts")
        return {
            "question": conversation["question"],
            "choices": list(conversation["choices"]),
            "letters": list(conversation["letters"]),
            "answer": conversation.get("answer"),
            "messages": conversation.get("messages", []),
        }

    def _format_choices(self, letters: Sequence[str], choices: Sequence[str]) -> str:
        lines = [f"  {letter}. {choice}" for letter, choice in zip(letters, choices)]
        return "\n".join(lines)

    def _render_demo(self, example: Dict[str, object], metadata: Dict[str, object]) -> FewShotDemo:
        formatted = ["Question:", str(example["question"]).strip(), "Choices:"]
        formatted.append(self._format_choices(example["letters"], example["choices"]))
        completion = str(example.get("answer", "")).strip()
        formatted.append("Answer: " + completion)
        prompt = "\n".join(formatted[:-1])
        return FewShotDemo(prompt=prompt, completion=completion, metadata=metadata)

    def _render_query(self, example: Dict[str, object]) -> str:
        sections = ["Question:", str(example["question"]).strip(), "Choices:"]
        sections.append(self._format_choices(example["letters"], example["choices"]))
        sections.append("Answer:")
        return "\n".join(sections)

    def build_prompt(
        self,
        query_index: int,
        num_demonstrations: int,
        seed: int | None = None,
    ) -> FewShotPrompt:
        if not (0 <= query_index < len(self.conversations)):
            raise IndexError("query_index out of range")
        rng = random.Random(seed if seed is not None else 1729 + query_index)
        query_raw = self._normalise(self.conversations[query_index])
        available = [i for i in range(len(self.conversations)) if i != query_index]
        requested = min(num_demonstrations, self.max_demonstrations, len(available))
        demo_indices = rng.sample(available, requested)
        demos: List[FewShotDemo] = []
        for idx in demo_indices:
            base_example = self._normalise(self.conversations[idx])
            augmented, metadata = self.augmentation.augment(base_example, rng)
            metadata.update({"source_index": idx})
            demos.append(self._render_demo(augmented, metadata))
        formatted_sections = [self.instructions, ""] if self.instructions else []
        for i, demo in enumerate(demos, start=1):
            formatted_sections.append(f"Example {i}:")
            formatted_sections.append(demo.prompt)
            formatted_sections.append("Answer: " + demo.completion)
            formatted_sections.append("")
        formatted_sections.append("Problem:")
        formatted_sections.append(self._render_query(query_raw))
        prompt_text = "\n".join(formatted_sections)
        metadata = {
            "demo_indices": demo_indices,
            "query_index": query_index,
            "letters": query_raw["letters"],
        }
        return FewShotPrompt(
            instructions=self.instructions,
            demos=demos,
            query={"prompt": query_raw, "metadata": metadata},
            formatted=prompt_text,
        )


__all__ = [
    "FewShotDemo",
    "FewShotPrompt",
    "ArcAugmenter",
    "ArcFewShotPromptBuilder",
    "LoRAConfig",
]
