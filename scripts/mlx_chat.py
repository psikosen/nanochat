"""Interact with a nanochat checkpoint converted for MLX."""
from __future__ import annotations

import argparse
import json

from nanochat.tokenizer import get_tokenizer
from nanochat.mlx_backend import GPTConfig, MLXEngine, MLXGPT, load_mlx_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", help="Path to the MLX weights .npz file.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON file containing the model configuration.",
    )
    parser.add_argument("--prompt", default="", help="Optional one-shot prompt.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(weights_path: str, config_path: str) -> MLXGPT:
    with open(config_path, "r", encoding="utf-8") as fh:
        config_dict = json.load(fh)
    config = GPTConfig(**config_dict)
    model = MLXGPT(config)
    weights = load_mlx_weights(weights_path)
    model.load_state_dict(weights)
    return model


def main() -> None:
    args = parse_args()
    tokenizer = get_tokenizer()
    model = build_model(args.weights, args.config)
    engine = MLXEngine(model, tokenizer)
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    conversation_tokens = [bos]
    print("\nNanoChat MLX Interactive Mode")
    print("-" * 50)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to reset the dialogue")
    print("-" * 50)
    while True:
        if args.prompt:
            user_input = args.prompt
        else:
            try:
                user_input = input("\nUser: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            conversation_tokens = [bos]
            print("Conversation cleared.")
            continue
        if not user_input:
            continue
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(user_input))
        conversation_tokens.append(user_end)
        conversation_tokens.append(assistant_start)
        print("\nAssistant: ", end="", flush=True)
        response_tokens = []
        generate_kwargs = dict(
            num_samples=1,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )
        for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
            token = token_column[0]
            response_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
        print()
        if not response_tokens or response_tokens[-1] != assistant_end:
            response_tokens.append(assistant_end)
        conversation_tokens.extend(response_tokens)
        if args.prompt:
            break


if __name__ == "__main__":
    main()
