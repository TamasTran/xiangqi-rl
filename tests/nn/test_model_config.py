from __future__ import annotations

import pytest

pytest.importorskip("torch")

from xiangqi_rl.engine import NUM_COLS, NUM_ROWS  # noqa: E402
from xiangqi_rl.env import ACTION_DIM, NUM_CHANNELS, OBSERVATION_SHAPE, action_to_move  # noqa: E402
from xiangqi_rl.nn import ModelConfig  # noqa: E402


def test_defaults_come_from_phase2_interfaces() -> None:
    cfg = ModelConfig()
    assert cfg.observation_shape == OBSERVATION_SHAPE == (NUM_CHANNELS, NUM_ROWS, NUM_COLS)
    assert cfg.action_dim == ACTION_DIM == 8100


def test_action_encoding_roundtrip_agrees_with_action_dim() -> None:
    cfg = ModelConfig()
    for action in (1, 89, 90, 4000, cfg.action_dim - 2):
        move = action_to_move(action)
        assert move.to_action() == action == 90 * move.from_sq + move.to_sq
    with pytest.raises(Exception):
        action_to_move(cfg.action_dim)


def test_dict_roundtrip() -> None:
    cfg = ModelConfig(hidden_channels=32, num_res_blocks=2, norm="batch")
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(TypeError):
        ModelConfig.from_dict({**cfg.to_dict(), "bogus": 1})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_channels": 0},
        {"in_channels": -1},
        {"action_dim": 0},
        {"board_height": 0},
        {"num_res_blocks": -1},
        {"hidden_channels": 30},  # not divisible by 8 groups
        {"norm": "layer"},
        {"value_activation": "relu"},
        {"hidden_channels": 2.5},
        {"num_res_blocks": True},
    ],
)
def test_invalid_config_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_zero_blocks_allowed() -> None:
    assert ModelConfig(num_res_blocks=0).num_res_blocks == 0
