from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from xiangqi_rl.engine import Game  # noqa: E402
from xiangqi_rl.env import action_to_move  # noqa: E402
from xiangqi_rl.evaluation.ppo_evaluator import (  # noqa: E402
    PolicyAgent,
    evaluate_against_baselines,
    make_baseline,
)
from xiangqi_rl.nn import ModelConfig, create_model  # noqa: E402

CFG = ModelConfig(hidden_channels=16, num_res_blocks=1, norm_groups=4, policy_channels=4,
                  value_channels=4, value_hidden=16)


def test_policy_agent_only_plays_legal_moves_greedy_and_sampled() -> None:
    model = create_model(CFG, seed=0)
    for greedy in (True, False):
        agent = PolicyAgent(model, greedy=greedy, seed=1)
        game = Game.new_game()
        for _ in range(30):
            if game.is_terminal():
                break
            move = action_to_move(agent.select_action(game))
            assert game.is_legal(move)
            game.make_move(move)


def test_evaluation_counts_colours_and_is_repeatable() -> None:
    model = create_model(CFG, seed=0)
    kw = dict(games=6, seed=3, game_max_plies=30)
    r1 = evaluate_against_baselines(model, ("random", "heuristic"), **kw)
    r2 = evaluate_against_baselines(model, ("random", "heuristic"), **kw)
    assert r1 == r2                                         # same seed -> same numbers
    for name, r in r1.items():
        assert r.games == 6 and r.wins + r.draws + r.losses + r.truncated == 6
        assert r.by_color["red"].games == r.by_color["black"].games == 3
        assert r.invalid_actions == 0
        assert r.win_rate + r.draw_rate <= 1.0 and 0.0 <= r.score <= 1.0
        assert r.average_plies <= 30


def test_evaluation_restores_training_mode_and_leaves_weights() -> None:
    model = create_model(CFG, seed=0).train()
    before = [p.detach().clone() for p in model.parameters()]
    evaluate_against_baselines(model, ("random",), games=2, game_max_plies=10)
    assert model.training
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert all(p.grad is None for p in model.parameters())


def test_unknown_baseline_rejected() -> None:
    with pytest.raises(ValueError):
        make_baseline("stockfish", 0)
