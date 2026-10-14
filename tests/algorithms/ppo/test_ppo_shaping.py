"""Potential-based material shaping: F = gamma * Phi(s') - Phi(s), Phi(terminal) = 0."""

from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xiangqi_rl.agents import RandomAgent  # noqa: E402
from xiangqi_rl.algorithms.ppo import (  # noqa: E402
    SMOKE_OVERRIDES,
    OpponentEnv,
    PPOConfig,
    PPOTrainer,
    material_balance,
)
from xiangqi_rl.engine import Board, Game, GameConfig, Move, Side  # noqa: E402


class Scripted:
    def __init__(self, moves: list[str]) -> None:
        self.name = "scripted"
        self._moves = iter(moves)

    def select_action(self, game) -> int:
        return Move.from_uci(next(self._moves)).to_action()


def test_material_balance_values() -> None:
    board = Board.initial()
    assert material_balance(board, Side.RED) == 0.0 == material_balance(board, Side.BLACK)
    game = Game.new_game()
    game.make_move("b2b9")  # red cannon captures the black horse (value 4)
    assert material_balance(game.board, Side.RED) == pytest.approx(4 / 48)
    assert material_balance(game.board, Side.BLACK) == pytest.approx(-4 / 48)
    only_generals = Board.from_pieces({"e0": "K", "d9": "k", "a0": "R"})
    assert material_balance(only_generals, Side.RED) == pytest.approx(9 / 48)


def test_zero_coefficient_leaves_rewards_unchanged() -> None:
    env = OpponentEnv([RandomAgent(0)], game_config=GameConfig(max_plies=30), seed=1)
    _, info = env.reset()
    done = False
    while not done:
        action = int(np.flatnonzero(info["action_mask"])[0])
        _, reward, terminated, truncated, info = env.step(action)
        assert info["shaping_reward"] == 0.0 and reward == info["game_reward"]
        done = terminated or truncated


def test_capture_is_rewarded_immediately() -> None:
    env = OpponentEnv([Scripted(["c6c5"])], color_schedule="red",
                      material_shaping_coef=1.0, shaping_gamma=0.9)
    env.reset()
    _, reward, terminated, _, info = env.step(Move.from_uci("b2b9").to_action())
    assert not terminated and info["game_reward"] == 0.0
    assert info["shaping_reward"] == pytest.approx(0.9 * 4 / 48)  # gamma*Phi(s') - Phi(s0)=0
    assert reward == pytest.approx(info["shaping_reward"])


def _play_episode(env: OpponentEnv, seed: int) -> tuple[list[dict], float]:
    """Random agent moves; returns step infos and Phi(s0) recomputed independently."""
    rng = random.Random(seed)
    _, info = env.reset()
    phi0 = env._shaping_coef * material_balance(env._env.state.board, env.agent_side)
    infos, done = [], False
    while not done:
        action = rng.choice(list(np.flatnonzero(info["action_mask"])))
        _, reward, terminated, truncated, info = env.step(int(action))
        infos.append({**info, "reward": reward})
        done = terminated or truncated
    return infos, phi0


@pytest.mark.parametrize("gamma", [1.0, 0.9])
@pytest.mark.parametrize("schedule", ["red", "black"])
def test_shaping_telescopes_to_minus_phi0_on_terminated_episodes(gamma, schedule) -> None:
    env = OpponentEnv([RandomAgent(3)], game_config=GameConfig(max_plies=80),
                      color_schedule=schedule, material_shaping_coef=0.5, shaping_gamma=gamma)
    infos, phi0 = _play_episode(env, seed=11)
    assert infos[-1]["terminated"]
    assert any(i["shaping_reward"] != 0 for i in infos[:-1])  # material changed mid-game
    discounted = sum(gamma ** t * i["shaping_reward"] for t, i in enumerate(infos))
    # Ng et al.: sum_t gamma^t F_t = gamma^T Phi(s_T) - Phi(s_0), with Phi(terminal) = 0.
    assert discounted == pytest.approx(-phi0, abs=1e-9)
    assert infos[-1]["game_reward"] in (-1.0, 0.0, 1.0)


def test_truncated_episode_keeps_final_potential() -> None:
    env = OpponentEnv([RandomAgent(3)], game_config=GameConfig(max_plies=None),
                      max_episode_plies=40, color_schedule="red",
                      material_shaping_coef=0.5, shaping_gamma=1.0)
    infos, phi0 = _play_episode(env, seed=5)
    assert infos[-1]["truncated"] and phi0 == 0.0
    phi_final = 0.5 * material_balance(env._env.state.board, Side.RED)
    assert sum(i["shaping_reward"] for i in infos) == pytest.approx(phi_final - phi0, abs=1e-9)
    assert infos[-1]["game_reward"] == 0.0


def test_invalid_shaping_arguments() -> None:
    with pytest.raises(ValueError):
        OpponentEnv([RandomAgent(0)], material_shaping_coef=-0.1)
    with pytest.raises(ValueError):
        OpponentEnv([RandomAgent(0)], shaping_gamma=0.0)


def test_config_requires_unbounded_value_head_for_shaping() -> None:
    with pytest.raises(ValueError, match="value_activation"):
        PPOConfig(material_shaping_coef=0.5)
    with pytest.raises(ValueError):
        PPOConfig().with_overrides({"material_shaping_coef": -1.0})
    cfg = PPOConfig().with_overrides({"material_shaping_coef": 0.5,
                                      "model.value_activation": "none"})
    assert cfg.material_shaping_coef == 0.5


def test_trainer_outcomes_use_game_reward_not_shaped_reward(tmp_path) -> None:
    cfg = PPOConfig().with_overrides({
        **SMOKE_OVERRIDES, "device": "cpu", "output_dir": str(tmp_path / "run"),
        "eval_interval": 0, "rollout_steps": 128, "minibatch_size": 64,
        "material_shaping_coef": 1.0, "model.value_activation": "none", "opponents": ["random"],
    })
    t = PPOTrainer(cfg)
    episodes, _ = t.collect_rollout()
    assert episodes
    for e in episodes:
        assert e.reward in (-1.0, 0.0, 1.0)
        expected = {1.0: "win", -1.0: "loss", 0.0: "draw"}[e.reward]
        assert e.outcome in (expected, "truncated")
    buf = t.buffer
    assert np.any(buf.rewards[~buf.terminated] != 0)   # shaping reaches the training data
    metrics = t.update()
    assert np.isfinite(metrics["policy_loss"]) and np.isfinite(metrics["value_loss"])
