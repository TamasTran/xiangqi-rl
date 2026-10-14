from __future__ import annotations

import pytest

from conftest import RED_MATE_IN_ONE
from xiangqi_rl.agents import HeuristicAgent, RandomAgent
from xiangqi_rl.engine import Game, GameConfig, GameResult, Move, Side
from xiangqi_rl.env import XiangqiEnv
from xiangqi_rl.evaluation import play_game, run_match


class Scripted:
    """Plays a fixed list of UCCI moves and records every position it saw."""

    def __init__(self, name: str, moves: list[str]) -> None:
        self.name = name
        self._moves = iter(moves)
        self.seen_sides: list[Side] = []

    def select_action(self, game: Game) -> int:
        self.seen_sides.append(game.side_to_move)
        return Move.from_uci(next(self._moves)).to_action()


def test_turn_order_alternates_and_colours_respected() -> None:
    red = Scripted("r", ["b0c2", "c2b0"])
    black = Scripted("b", ["b9c7", "c7b9"])
    env = XiangqiEnv(max_episode_steps=4)
    rec = play_game(red, black, env, seed=0)
    assert red.seen_sides == [Side.RED, Side.RED]
    assert black.seen_sides == [Side.BLACK, Side.BLACK]
    assert rec.plies == 4 and rec.truncated and rec.result is GameResult.IN_PROGRESS
    assert rec.termination == "external_step_limit"


def test_checkmate_game_record(make_fen) -> None:
    env = XiangqiEnv(initial_fen=make_fen(RED_MATE_IN_ONE))
    rec = play_game(Scripted("r", ["i5i9"]), Scripted("b", []), env)
    assert rec.result is GameResult.RED_WINS and rec.plies == 1
    assert rec.termination == "checkmate" and not rec.truncated and rec.invalid_actions == 0


def test_illegal_action_is_counted_and_forfeits() -> None:
    env = XiangqiEnv()
    rec = play_game(Scripted("r", ["a0a5"]), Scripted("b", ["b9c7"]), env)
    assert rec.invalid_actions == 1 and rec.forfeit_by is Side.RED
    assert rec.result is GameResult.BLACK_WINS and rec.plies == 0
    assert rec.termination == "illegal_action"


def test_run_match_counts_sum_and_colour_swap() -> None:
    cfg = GameConfig(max_plies=40)
    stats, records = run_match(
        HeuristicAgent(seed=1), RandomAgent(2), 6, seed=10, game_config=cfg
    )
    assert stats.games == 6 == len(records)
    assert stats.red_wins + stats.black_wins + stats.draws + stats.truncated == 6
    assert stats.invalid_actions == 0
    # Colours alternate: heuristic is RED in games 0, 2, 4.
    assert [r.red for r in records] == ["heuristic", "random"] * 3
    assert stats.by_agent_color[("heuristic", Side.RED)].games == 3
    assert stats.by_agent_color[("heuristic", Side.BLACK)].games == 3
    assert stats.average_plies == pytest.approx(sum(r.plies for r in records) / 6)
    assert all(r.plies <= 40 for r in records)


def test_run_match_without_swap_and_validation() -> None:
    _, records = run_match(
        RandomAgent(1, name="a"), RandomAgent(2, name="b"), 4, swap_colors=False,
        game_config=GameConfig(max_plies=10),
    )
    assert {r.red for r in records} == {"a"} and {r.black for r in records} == {"b"}
    with pytest.raises(ValueError):
        run_match(RandomAgent(1), RandomAgent(2), 1)  # same default name
    assert run_match(RandomAgent(1, name="a"), RandomAgent(2, name="b"), 0)[0].games == 0


def test_run_match_is_reproducible() -> None:
    def go() -> list[tuple[GameResult, int]]:
        _, recs = run_match(
            HeuristicAgent(seed=1), RandomAgent(2), 4, seed=3, game_config=GameConfig(max_plies=60)
        )
        return [(r.result, r.plies) for r in recs]

    assert go() == go()


def test_external_limit_counts_as_truncated_not_draw() -> None:
    stats, _ = run_match(
        RandomAgent(1, name="a"), RandomAgent(2, name="b"), 3, max_episode_steps=5
    )
    assert stats.truncated == 3 and stats.draws == 0 and stats.average_plies == 5.0
