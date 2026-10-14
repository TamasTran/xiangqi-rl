from __future__ import annotations

import pytest

from conftest import BLACK_MATE_IN_ONE, RED_MATE_IN_ONE
from xiangqi_rl.engine import GameResult, Move, Side
from xiangqi_rl.env import XiangqiEnv, result_for_side, step_reward


def act(uci: str) -> int:
    return Move.from_uci(uci).to_action()


@pytest.mark.parametrize(
    ("result", "side", "expected"),
    [
        (GameResult.RED_WINS, Side.RED, 1.0),
        (GameResult.RED_WINS, Side.BLACK, -1.0),
        (GameResult.BLACK_WINS, Side.BLACK, 1.0),
        (GameResult.BLACK_WINS, Side.RED, -1.0),
        (GameResult.DRAW, Side.RED, 0.0),
        (GameResult.DRAW, Side.BLACK, 0.0),
        (GameResult.IN_PROGRESS, Side.RED, 0.0),
    ],
)
def test_result_for_side(result, side, expected) -> None:
    assert result_for_side(result, side) == expected


def test_step_reward_perspectives() -> None:
    assert step_reward(GameResult.RED_WINS, Side.RED) == 1.0  # mover wins
    assert step_reward(GameResult.BLACK_WINS, Side.RED) == -1.0  # mover loses
    assert step_reward(GameResult.RED_WINS, Side.BLACK, perspective=Side.RED) == 1.0
    assert step_reward(GameResult.RED_WINS, Side.RED, perspective=Side.BLACK) == -1.0


def test_red_win_mover_perspective(make_fen) -> None:
    env = XiangqiEnv(initial_fen=make_fen(RED_MATE_IN_ONE))
    env.reset()
    _, reward, _, _, info = env.step(act("i5i9"))
    assert reward == 1.0 and info["mover"] == Side.RED


def test_black_win_mover_perspective_is_positive_for_black(make_fen) -> None:
    env = XiangqiEnv(initial_fen=make_fen(BLACK_MATE_IN_ONE, Side.BLACK))
    env.reset()
    _, reward, terminated, _, info = env.step(act("i4i0"))
    assert terminated and reward == 1.0
    assert info["mover"] == Side.BLACK and info["result"] == "black_wins"


def test_same_game_fixed_perspectives_differ(make_fen) -> None:
    fen = make_fen(BLACK_MATE_IN_ONE, Side.BLACK)
    rewards = {}
    for name, perspective in (("mover", None), ("red", Side.RED), ("black", Side.BLACK)):
        env = XiangqiEnv(initial_fen=fen, reward_perspective=perspective)
        env.reset()
        rewards[name] = env.step(act("i4i0"))[1]
    assert rewards == {"mover": 1.0, "red": -1.0, "black": 1.0}


def test_draw_and_nonterminal_rewards_are_zero() -> None:
    from xiangqi_rl.engine import GameConfig

    env = XiangqiEnv(game_config=GameConfig(max_plies=3))
    env.reset()
    rewards = [env.step(act(m))[1] for m in ["b0c2", "b9c7", "h0g2"]]
    assert rewards == [0.0, 0.0, 0.0]
    assert env.state.result is GameResult.DRAW


def test_truncation_has_no_win_or_loss() -> None:
    env = XiangqiEnv(max_episode_steps=1)
    env.reset()
    _, reward, terminated, truncated, info = env.step(act("h2e2"))
    assert (reward, terminated, truncated) == (0.0, False, True)
    assert info["result"] == "in_progress"


def test_mover_loses_via_perpetual_check(make_fen) -> None:
    # Black to move and in check from Ra8. Black steps e8-e9, Red re-checks
    # with Ra9, and so on. Red checks on every one of its moves, so when the
    # start position occurs a third time (after Red's move) the engine's
    # repetition policy makes the perpetual checker lose: the *mover* gets -1.
    pieces = {"d0": "K", "a8": "R", "e8": "k"}
    env = XiangqiEnv(initial_fen=make_fen(pieces, Side.BLACK))
    env.reset()
    cycle = ["e8e9", "a8a9", "e9e8", "a9a8"]
    out = None
    for uci in cycle * 2:
        out = env.step(act(uci))
        if out[2]:
            break
    _, reward, terminated, truncated, info = out
    assert terminated and not truncated
    assert info["termination_reason"] == "perpetual_check"
    assert info["mover"] == Side.RED and info["result"] == "black_wins"
    assert reward == -1.0
    assert env.steps == 8
