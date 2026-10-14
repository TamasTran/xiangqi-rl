from __future__ import annotations

import pytest

from conftest import RED_MATE_IN_ONE
from xiangqi_rl.agents import NoLegalActionError, RandomAgent
from xiangqi_rl.engine import Game
from xiangqi_rl.env import action_to_move


def play_with(agent: RandomAgent, plies: int) -> list[int]:
    game = Game.new_game()
    actions = []
    for _ in range(plies):
        if game.is_terminal():
            break
        action = agent.select_action(game)
        actions.append(action)
        game.make_move(action_to_move(action))
    return actions


def test_only_legal_actions_over_many_positions() -> None:
    agent = RandomAgent(seed=0)
    game = Game.new_game()
    while not game.is_terminal() and game.ply < 150:
        action = agent.select_action(game)
        assert game.is_legal(action_to_move(action))
        game.make_move(action_to_move(action))


def test_same_seed_same_choices_different_seed_differs() -> None:
    assert play_with(RandomAgent(seed=5), 40) == play_with(RandomAgent(seed=5), 40)
    assert play_with(RandomAgent(seed=5), 40) != play_with(RandomAgent(seed=6), 40)


def test_does_not_use_global_random_state() -> None:
    import random

    random.seed(1)
    first = play_with(RandomAgent(seed=9), 20)
    random.seed(2)
    random.random()
    assert play_with(RandomAgent(seed=9), 20) == first


def test_does_not_mutate_game() -> None:
    game = Game.new_game()
    fen, keys = game.to_fen(), game.position_keys
    RandomAgent(seed=1).select_action(game)
    assert game.to_fen() == fen and game.position_keys == keys


def test_terminal_position_raises(make_fen) -> None:
    game = Game.from_fen(make_fen(RED_MATE_IN_ONE))
    game.make_move("i5i9")
    with pytest.raises(NoLegalActionError):
        RandomAgent(seed=0).select_action(game)
