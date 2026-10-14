from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xiangqi_rl.agents import HeuristicAgent, RandomAgent  # noqa: E402
from xiangqi_rl.algorithms.ppo import OpponentEnv  # noqa: E402
from xiangqi_rl.engine import GameConfig, Move, Side  # noqa: E402
from xiangqi_rl.env import EpisodeOverError, IllegalActionError  # noqa: E402

# Red to move; its only legal move is e0f0 (d0 would face the black general,
# e1 is covered by the a1 chariot). Then Black mates with i4i0.
BLACK_MATES_AFTER_RED_MOVE = {"e0": "K", "a1": "r", "i4": "r", "d9": "k"}


class Scripted:
    """Opponent that plays a fixed list of UCCI moves and logs what it saw."""

    def __init__(self, moves: list[str], name: str = "scripted") -> None:
        self.name = name
        self._moves = iter(moves)
        self.seen_sides: list[Side] = []

    def select_action(self, game) -> int:
        self.seen_sides.append(game.side_to_move)
        return Move.from_uci(next(self._moves)).to_action()


def first_legal(info) -> int:
    return int(np.flatnonzero(info["action_mask"])[0])


def test_colour_alternation_and_black_opening() -> None:
    env = OpponentEnv([RandomAgent(0)], color_schedule="alternate")
    obs, info = env.reset()
    assert info["agent_side"] == Side.RED and info["ply"] == 0
    obs, info = env.reset()
    # Agent is BLACK: RED (opponent) has already played one move.
    assert info["agent_side"] == Side.BLACK and info["ply"] == 1
    assert info["side_to_move"] == Side.BLACK and info["opponent_moves"] == 1
    obs, info = env.reset()
    assert info["agent_side"] == Side.RED


@pytest.mark.parametrize("schedule", ["red", "black"])
def test_fixed_colour_schedules(schedule) -> None:
    env = OpponentEnv([RandomAgent(0)], color_schedule=schedule)
    for _ in range(3):
        _, info = env.reset()
        assert info["agent_side"] == (Side.RED if schedule == "red" else Side.BLACK)


def test_random_schedule_uses_both_colours_reproducibly() -> None:
    def sides(seed: int) -> list[int]:
        env = OpponentEnv([RandomAgent(0)], color_schedule="random", seed=seed)
        return [env.reset()[1]["agent_side"] for _ in range(20)]

    assert set(sides(1)) == {0, 1} and sides(1) == sides(1)


@pytest.mark.parametrize("opponent_cls", [RandomAgent, HeuristicAgent])
def test_agent_only_decides_on_its_own_turn_and_opponent_moves_are_legal(opponent_cls) -> None:
    opponent = opponent_cls(seed=0)
    env = OpponentEnv([opponent], game_config=GameConfig(max_plies=40))
    for _ in range(2):  # one episode per colour
        obs, info = env.reset()
        agent = info["agent_side"]
        done = False
        steps = 0
        while not done:
            assert info["side_to_move"] == agent            # a decision point of the agent
            assert info["action_mask"].any()
            obs, reward, terminated, truncated, info = env.step(first_legal(info))
            steps += 1
            done = terminated or truncated
        # Plies = agent moves + opponent moves (+1 if the opponent opened).
        assert info["ply"] == steps + info["opponent_moves"]
        assert not info["action_mask"].any()                # all False once over


def test_terminal_reward_uses_agent_perspective() -> None:
    # Agent BLACK; RED opponent plays its only legal move, then BLACK mates.
    from xiangqi_rl.engine import Board

    fen = Board.from_pieces(BLACK_MATES_AFTER_RED_MOVE).to_fen_placement() + " w - - 0 1"
    env = OpponentEnv([Scripted(["e0f0"])], color_schedule="black", initial_fen=fen)
    _, info = env.reset()
    assert info["agent_side"] == Side.BLACK and info["side_to_move"] == Side.BLACK
    obs, reward, terminated, truncated, info = env.step(Move.from_uci("i4i0").to_action())
    assert terminated and not truncated and reward == 1.0
    assert info["result"] == "black_wins" and info["termination_reason"] == "checkmate"
    with pytest.raises(EpisodeOverError):
        env.step(0)


def test_opponent_move_can_end_the_episode_with_a_loss_for_the_agent() -> None:
    from xiangqi_rl.engine import Board

    fen = Board.from_pieces(BLACK_MATES_AFTER_RED_MOVE).to_fen_placement() + " w - - 0 1"
    opponent = Scripted(["i4i0"])
    env = OpponentEnv([opponent], color_schedule="red", initial_fen=fen)
    _, info = env.reset()
    assert info["agent_side"] == Side.RED
    obs, reward, terminated, truncated, info = env.step(Move.from_uci("e0f0").to_action())
    assert terminated and not truncated and reward == -1.0
    assert info["result"] == "black_wins" and info["opponent_moves"] == 1
    assert opponent.seen_sides == [Side.BLACK]               # opponent only moved for BLACK


def test_draw_by_engine_limit_is_terminated_with_zero_reward() -> None:
    env = OpponentEnv([RandomAgent(0)], game_config=GameConfig(max_plies=6), color_schedule="red")
    _, info = env.reset()
    done = False
    while not done:
        _, reward, terminated, truncated, info = env.step(first_legal(info))
        done = terminated or truncated
    assert terminated and not truncated and reward == 0.0
    assert info["termination_reason"] in ("max_plies", "repetition", "checkmate", "stalemate")


def test_external_limit_truncates_with_zero_reward_and_agent_may_not_be_to_move() -> None:
    env = OpponentEnv([RandomAgent(0)], max_episode_plies=3, color_schedule="red")
    _, info = env.reset()
    _, r1, t1, tr1, info = env.step(first_legal(info))      # plies 1 (agent) + 2 (opponent)
    assert (t1, tr1) == (False, False) and info["ply"] == 2
    _, r2, t2, tr2, info = env.step(first_legal(info))      # ply 3 (agent) -> limit reached
    assert (t2, tr2) == (False, True) and r2 == 0.0
    assert info["result"] == "in_progress" and info["truncated"]
    assert info["side_to_move"] != info["agent_side"]       # opponent never got to reply
    with pytest.raises(EpisodeOverError):
        env.step(0)


def test_illegal_agent_action_is_rejected_without_side_effects() -> None:
    env = OpponentEnv([RandomAgent(0)], color_schedule="red")
    _, info = env.reset()
    with pytest.raises(IllegalActionError):
        env.step(0)
    obs, reward, *_ = env.step(first_legal(info))           # still usable
    assert reward == 0.0


def test_opponent_selection_per_episode() -> None:
    a, b = RandomAgent(0, name="a"), RandomAgent(0, name="b")
    env = OpponentEnv([a, b], seed=3)
    names = {env.reset()[1]["opponent"] for _ in range(20)}
    assert names == {"a", "b"}
    with pytest.raises(ValueError):
        OpponentEnv([])
    with pytest.raises(ValueError):
        OpponentEnv([a], color_schedule="blue")
