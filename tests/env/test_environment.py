from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_reset_seed_determinism
from gymnasium.utils.passive_env_checker import env_reset_passive_checker, env_step_passive_checker

from conftest import BLACK_MATE_IN_ONE, RED_MATE_IN_ONE
from xiangqi_rl.engine import GameConfig, GameResult, Move, Side
from xiangqi_rl.env import (
    ACTION_DIM,
    EpisodeOverError,
    IllegalActionError,
    XiangqiEnv,
    observation_space,
)


def act(uci: str) -> int:
    return Move.from_uci(uci).to_action()


def test_reset_returns_obs_and_info() -> None:
    env = XiangqiEnv()
    obs, info = env.reset(seed=42)
    assert observation_space.contains(obs)
    assert info["current_player"] == Side.RED and info["ply"] == 0
    assert info["action_mask"].sum() == 44 and info["mover"] is None
    assert env.action_space.n == ACTION_DIM
    assert env.side_to_move is Side.RED and env.steps == 0


def test_step_returns_five_values_and_alternates_sides() -> None:
    env = XiangqiEnv()
    env.reset(seed=0)
    out = env.step(act("h2e2"))
    obs, reward, terminated, truncated, info = out
    assert observation_space.contains(obs)
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert info["mover"] == Side.RED and info["current_player"] == Side.BLACK
    assert env.side_to_move is Side.BLACK
    env.step(act("h9g7"))
    assert env.side_to_move is Side.RED and env.steps == 2 and env.state.ply == 2


def test_legal_move_changes_state_as_expected() -> None:
    env = XiangqiEnv()
    env.reset()
    before = env.state
    env.step(act("b2b9"))  # cannon captures the black horse
    after = env.state
    assert before.board != after.board and after.board.count(Side.BLACK) == 15
    assert before.board.count(Side.BLACK) == 16  # old snapshot untouched


@pytest.mark.parametrize("bad", [-1, ACTION_DIM, 0, 3.0, True, "5", None])
def test_illegal_actions_raise_without_mutation(bad) -> None:
    env = XiangqiEnv()
    env.reset()
    env.step(act("h2e2"))
    snapshot = (env.state, env.steps, env.game_copy().history, env.action_masks())
    with pytest.raises(IllegalActionError):
        env.step(bad)
    after = (env.state, env.steps, env.game_copy().history, env.action_masks())
    assert snapshot[0] == after[0] and snapshot[1] == after[1] and snapshot[2] == after[2]
    assert np.array_equal(snapshot[3], after[3])
    # Env is still usable afterwards.
    env.step(act("h9g7"))


def test_well_formed_but_illegal_move_rejected() -> None:
    env = XiangqiEnv()
    env.reset()
    with pytest.raises(IllegalActionError, match="not legal"):
        env.step(act("a0a5"))  # chariot blocked by its own soldier
    with pytest.raises(IllegalActionError):
        env.step(act("a9a8"))  # Black's piece on Red's turn
    assert env.state.ply == 0


def test_step_before_reset_raises() -> None:
    env = XiangqiEnv()
    with pytest.raises(EpisodeOverError):
        env.step(0)


def test_natural_termination_checkmate(make_fen) -> None:
    env = XiangqiEnv(initial_fen=make_fen(RED_MATE_IN_ONE))
    env.reset()
    obs, reward, terminated, truncated, info = env.step(act("i5i9"))
    assert terminated and not truncated and reward == 1.0
    assert info["result"] == GameResult.RED_WINS.value
    assert info["termination_reason"] == "checkmate"
    assert not info["action_mask"].any() and not env.action_masks().any()
    with pytest.raises(EpisodeOverError):
        env.step(act("e9e8"))
    # Reset gives a clean episode.
    env.reset()
    assert env.steps == 0 and not env.state.is_terminal
    env.step(act("i5i9"))


def test_engine_ply_limit_is_natural_draw_not_truncation() -> None:
    env = XiangqiEnv(game_config=GameConfig(max_plies=2))
    env.reset()
    env.step(act("b0c2"))
    _, reward, terminated, truncated, info = env.step(act("b9c7"))
    assert terminated and not truncated and reward == 0.0
    assert info["termination_reason"] == "max_plies" and info["result"] == "draw"
    assert info["engine_terminal"] and not info["externally_truncated"]


def test_external_step_limit_truncates_without_result() -> None:
    env = XiangqiEnv(max_episode_steps=2)
    env.reset()
    _, r1, t1, tr1, _ = env.step(act("b0c2"))
    _, r2, t2, tr2, info = env.step(act("b9c7"))
    assert (t1, tr1) == (False, False)
    assert (t2, tr2) == (False, True) and r2 == 0.0
    assert info["result"] == GameResult.IN_PROGRESS.value and info["externally_truncated"]
    assert not env.state.is_terminal  # engine result untouched
    with pytest.raises(EpisodeOverError):
        env.step(act("c2b0"))
    env.reset()
    assert env.steps == 0
    env.step(act("b0c2"))  # usable again


def test_natural_terminal_on_limit_step_is_terminated_not_truncated(make_fen) -> None:
    env = XiangqiEnv(initial_fen=make_fen(RED_MATE_IN_ONE), max_episode_steps=1)
    env.reset()
    _, _, terminated, truncated, _ = env.step(act("i5i9"))
    assert terminated and not truncated


def test_reset_with_fen_option_and_bad_fen(make_fen) -> None:
    env = XiangqiEnv()
    env.reset(options={"fen": make_fen(BLACK_MATE_IN_ONE, Side.BLACK)})
    assert env.side_to_move is Side.BLACK
    with pytest.raises(ValueError):
        env.reset(options={"fen": "garbage"})


def test_reset_from_terminal_fen_is_over(make_fen) -> None:
    # Black to move with no legal moves: the episode is over immediately.
    fen = make_fen({"e0": "K", "i8": "R", "d9": "k"}, Side.BLACK)
    env = XiangqiEnv(initial_fen=fen)
    _, info = env.reset()
    assert info["engine_terminal"] and not info["action_mask"].any()
    with pytest.raises(EpisodeOverError):
        env.step(0)


def test_render_ansi_and_close() -> None:
    env = XiangqiEnv(render_mode="ansi")
    env.reset()
    assert "river" in env.render()
    assert XiangqiEnv().render() is None
    env.close()
    with pytest.raises(EpisodeOverError):
        env.step(0)


def test_gymnasium_passive_checks() -> None:
    """Gymnasium's passive checkers, driven with legal actions.

    ``check_env`` itself samples ``action_space.sample()``, which is almost
    never legal here (illegal actions raise by design), so the individual
    checks are run with actions taken from the legal-action mask instead.
    """
    env = XiangqiEnv()
    env_reset_passive_checker(env)
    check_reset_seed_determinism(env)
    obs, info = env.reset(seed=3)
    assert env.observation_space.contains(obs)
    env_step_passive_checker(env, int(np.flatnonzero(info["action_mask"])[0]))


def test_same_actions_same_trajectory() -> None:
    def run() -> list[bytes]:
        env = XiangqiEnv()
        env.reset(seed=1)
        out = []
        for uci in ["h2e2", "h9g7", "h0g2", "b9c7"]:
            obs, *_ = env.step(act(uci))
            out.append(obs.tobytes())
        return out

    assert run() == run()
