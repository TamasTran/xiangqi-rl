from __future__ import annotations

import pytest

from xiangqi_rl.engine import GameResult, Side
from xiangqi_rl.evaluation import GameRecord, summarize


def rec(red: str, black: str, result: GameResult, plies: int, **kw) -> GameRecord:
    return GameRecord(red=red, black=black, result=result, plies=plies, **kw)


def test_counters_and_averages() -> None:
    records = [
        rec("A", "B", GameResult.RED_WINS, 40),
        rec("B", "A", GameResult.RED_WINS, 20),  # B wins as RED
        rec("A", "B", GameResult.DRAW, 100),
        rec("B", "A", GameResult.BLACK_WINS, 60),  # A wins as BLACK
        rec("A", "B", GameResult.IN_PROGRESS, 10, truncated=True),
    ]
    s = summarize(records)
    assert (s.games, s.red_wins, s.black_wins, s.draws, s.truncated) == (5, 2, 1, 1, 1)
    assert s.red_wins + s.black_wins + s.draws + s.truncated == s.games
    assert s.average_plies == pytest.approx(46.0)
    a, b = s.by_agent["A"], s.by_agent["B"]
    assert (a.wins, a.losses, a.draws, a.truncated, a.games) == (2, 1, 1, 1, 5)
    assert (b.wins, b.losses, b.draws, b.truncated, b.games) == (1, 2, 1, 1, 5)
    assert a.win_rate == pytest.approx(0.4)
    assert a.score == pytest.approx((2 + 0.5 * 2) / 5)
    assert s.by_agent_color[("A", Side.RED)].games == 3
    assert s.by_agent_color[("A", Side.BLACK)].wins == 1
    assert s.by_agent_color[("B", Side.RED)].wins == 1


def test_invalid_actions_summed_and_empty_input() -> None:
    s = summarize(
        [
            rec("A", "B", GameResult.BLACK_WINS, 3, invalid_actions=1, forfeit_by=Side.RED),
            rec("A", "B", GameResult.DRAW, 5),
        ]
    )
    assert s.invalid_actions == 1
    empty = summarize([])
    assert empty.games == 0 and empty.average_plies == 0.0 and empty.by_agent == {}
