# nanochat MLX Port

This directory documents how to run nanochat end-to-end on Apple Silicon Macs (M1–M4) using [MLX](https://github.com/ml-explore/mlx).

## Prerequisites

1. Install Python 3.10 or newer via [uv](https://github.com/astral-sh/uv) or `pyenv`.
2. Install nanochat with the MLX extra (see [Installation](#installation)).
3. Convert an existing nanochat checkpoint to MLX format.

## Installation

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .[mlx]
```

`mlx` currently requires macOS 14+ on Apple Silicon. The `[mlx]` extra keeps PyTorch optional so the port can run without CUDA wheels.

## Converting Checkpoints

Use `scripts/convert_to_mlx.py` to convert a PyTorch checkpoint directory to MLX weights and JSON config:

```bash
python -m scripts.convert_to_mlx path/to/checkpoint_dir --output ~/nanochat_mlx/base_weights.npz
```

This command emits:

- `base_weights.npz`: MLX arrays for the model parameters.
- `base_weights.npz.json`: GPT configuration required to instantiate the MLX model.

## Running the Chat CLI

Once weights are converted, launch the MLX chat CLI:

```bash
python -m scripts.mlx_chat ~/nanochat_mlx/base_weights.npz --config ~/nanochat_mlx/base_weights.npz.json
```

The CLI mirrors `scripts/chat_cli.py` with support for the Python tool use protocol. Generation happens on the Apple Neural Engine/GPU via MLX.

Use `--prompt` for single response mode or interactively chat after the tool starts.

## Limitations

- Training is not yet ported; only inference is supported today.
- KV cache cloning uses broadcasting; extremely large batches may require further optimization.
- PyTorch is still needed once to convert checkpoints.

## Testing

Unit tests are designed to skip automatically when MLX is not installed so CI on Linux remains green.
