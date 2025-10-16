import json

import pytest

torch = pytest.importorskip("torch")

from nanochat.gpt import GPT, GPTConfig
from nanochat.lora_config import LoRAConfig
from nanochat.adapter_registry import AdapterMetadata, AdapterRegistry
from nanochat.fewshot import ArcFewShotPromptBuilder
from nanochat.self_edit import SelfEditConfig, SelfEditLogger, SelfEditOrchestrator


def make_conversations():
    base_example = {
        "messages": [
            {"role": "user", "content": "Multiple Choice question: Sample?"},
            {"role": "assistant", "content": "A"},
        ],
        "question": "Sample question?",
        "choices": ["First option", "Second option", "Third option"],
        "letters": ["A", "B", "C"],
        "answer": "A",
    }
    alt_example = {
        "messages": [
            {"role": "user", "content": "Multiple Choice question: Another?"},
            {"role": "assistant", "content": "B"},
        ],
        "question": "Another question?",
        "choices": ["Alt one", "Alt two", "Alt three"],
        "letters": ["A", "B", "C"],
        "answer": "B",
    }
    return [base_example, alt_example]


def test_lora_adapter_roundtrip(tmp_path):
    config = GPTConfig(
        sequence_len=32,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        lora=LoRAConfig(rank=4, alpha=8, target_modules=("attn.c_q", "attn.c_v")),
    )
    model = GPT(config)
    optimizers = model.setup_optimizers()
    assert len(optimizers) == 1
    assert model.has_lora()
    state = model.get_lora_state_dict()
    assert state, "LoRA state dict should not be empty"

    registry = AdapterRegistry(tmp_path)
    metadata = AdapterMetadata(
        name="demo",
        config=config.lora,
        metrics={"accuracy": 0.42},
        created_at="2025-01-01T00:00:00Z",
        description="Test adapter",
    )
    registry.save(model, metadata, merge_weights=False)
    # Reset the lora weights to zero before load to ensure they change
    for layer_state in state.values():
        for tensor in layer_state.values():
            tensor.zero_()
    loaded, base = registry.load(model, "demo")
    assert loaded.name == "demo"
    assert "accuracy" in loaded.metrics
    assert base == {} or isinstance(base, dict)
    reloaded_state = model.get_lora_state_dict()
    for name, tensors in reloaded_state.items():
        assert all(torch.any(param != 0) for param in tensors.values()), f"Expected non-zero weights for {name}"


def test_arc_prompt_builder_output():
    conversations = make_conversations()
    builder = ArcFewShotPromptBuilder(conversations, instructions="Answer with the correct letter.", max_demonstrations=1)
    prompt = builder.build_prompt(query_index=0, num_demonstrations=1, seed=123)
    assert "Example 1" in prompt.formatted
    assert "Problem:" in prompt.formatted
    assert "Answer:" in prompt.formatted.splitlines()[-1]
    assert prompt.query["metadata"]["letters"] == ["A", "B", "C"]


def test_self_edit_logging(tmp_path):
    conversations = make_conversations()
    builder = ArcFewShotPromptBuilder(conversations, instructions="Choose the best option.", max_demonstrations=1)

    def generator(prompt_text, config):
        return "A"

    def scorer(example, response):
        return 1.0 if response.strip().startswith(example["answer"]) else 0.0

    log_path = tmp_path / "self_edit.log"
    config = SelfEditConfig(num_demonstrations=1, samples_per_prompt=2, score_threshold=0.5)
    with SelfEditLogger(log_path) as logger:
        orchestrator = SelfEditOrchestrator(builder, generator, scorer, logger, config)
        summaries = orchestrator.run([0])
    assert summaries[0].acceptance_rate == 1.0
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) >= 4
    record = json.loads(lines[0])
    for field in SelfEditLogger.schema_fields:
        assert field in record
    assert lines[1].startswith("[Continuous skepticism (Sherlock Protocol)]")
