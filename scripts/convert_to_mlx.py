"""Convert a nanochat PyTorch checkpoint to MLX weight format."""
from __future__ import annotations

import argparse
import json
import os

from nanochat.checkpoint_manager import find_last_step, load_checkpoint
from nanochat.mlx_backend.checkpoint import convert_torch_state_dict, save_mlx_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint_dir",
        help="Path to the checkpoint directory containing model_<step>.pt files.",
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Specific training step to convert. Defaults to the last available step.",
    )
    parser.add_argument("--output", required=True, help="Output .npz path for the MLX weights.")
    parser.add_argument(
        "--config-output",
        help="Optional path for the JSON model config. Defaults to <output>.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step = args.step
    if step is None:
        step = find_last_step(args.checkpoint_dir)
    model_state, _, meta = load_checkpoint(args.checkpoint_dir, step, device="cpu", load_optimizer=False)
    mlx_state = convert_torch_state_dict(model_state)
    save_mlx_weights(mlx_state, args.output)
    config_path = args.config_output or f"{args.output}.json"
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(meta["model_config"], fh, indent=2)
    print(f"Saved MLX weights to {os.path.abspath(args.output)}")
    print(f"Saved model config to {os.path.abspath(config_path)}")


if __name__ == "__main__":
    main()
