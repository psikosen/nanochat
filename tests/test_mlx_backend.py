import pytest

np = pytest.importorskip("numpy")

from nanochat.mlx_backend import GPTConfig
from nanochat.mlx_backend.checkpoint import convert_torch_state_dict, save_mlx_weights, load_mlx_weights


def test_mlx_backend_exposes_config_defaults():
    config = GPTConfig()
    assert config.n_layer == 12
    assert config.vocab_size == 50304


def test_convert_torch_state_dict_maps_keys(tmp_path):
    torch = pytest.importorskip("torch")
    state = {
        "transformer.wte.weight": torch.ones((10, 4)),
        "transformer.h.0.attn.c_q.weight": torch.ones((16, 4)),
        "transformer.h.1.mlp.c_fc.weight": torch.ones((32, 8)),
        "lm_head.weight": torch.ones((10, 4)),
    }
    converted = convert_torch_state_dict(state)
    assert "wte.weight" in converted
    assert "blocks.layer_0.attn.c_q.weight" in converted
    assert "blocks.layer_1.mlp.c_fc.weight" in converted
    assert "lm_head.weight" in converted
    out_path = tmp_path / "weights.npz"
    save_mlx_weights(converted, str(out_path))
    loaded = load_mlx_weights(str(out_path))
    assert set(loaded) == set(converted)
    for key in loaded:
        np.testing.assert_allclose(loaded[key], converted[key])
