"""Game records and aggregate statistics for agent-vs-agent matches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from xiangqi_rl.engine import GameResult, Side


@dataclass(frozen=True)
class GameRecord:
    """Outcome of one played game.

    ``result`` is the engine result (``IN_PROGRESS`` only if the game was cut
    off by an external step limit, i.e. ``truncated``). ``forfeit_by`` is set
    when an agent submitted an illegal action: that agent loses and the game
    stops (``result`` is then the opponent's win).
    """

    red: str
    black: str
    result: GameResult
    plies: int
    truncated: bool = False
    invalid_actions: int = 0
    forfeit_by: Side | None = None
    termination: str | None = None


@dataclass(frozen=True)
class ColorStats:
    """One agent's record when playing a given colour."""

    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    truncated: int = 0

    @property
    def score(self) -> float:
        """Win = 1, draw or truncation = 0.5, loss = 0, averaged (0 if no games)."""
        return (self.wins + 0.5 * (self.draws + self.truncated)) / self.games if self.games else 0.0

    @property
    def win_rate(self) -> float:
        """Fraction of games won (0 if no games)."""
        return self.wins / self.games if self.games else 0.0


@dataclass(frozen=True)
class MatchStats:
    """Aggregate statistics of a set of games.

    ``red_wins + black_wins + draws + truncated == games``.
    """

    games: int
    red_wins: int
    black_wins: int
    draws: int
    truncated: int
    invalid_actions: int
    average_plies: float
    by_agent: dict[str, ColorStats]
    by_agent_color: dict[tuple[str, Side], ColorStats]


def summarize(records: Sequence[GameRecord]) -> MatchStats:
    """Aggregate ``records`` into :class:`MatchStats` (per agent *name*)."""
    red_wins = sum(r.result is GameResult.RED_WINS for r in records)
    black_wins = sum(r.result is GameResult.BLACK_WINS for r in records)
    truncated = sum(r.truncated for r in records)
    draws = sum(r.result is GameResult.DRAW and not r.truncated for r in records)

    per_key: dict[tuple[str, Side], dict[str, int]] = {}
    for r in records:
        for side, name in ((Side.RED, r.red), (Side.BLACK, r.black)):
            c = per_key.setdefault(
                (name, side), {"games": 0, "wins": 0, "losses": 0, "draws": 0, "truncated": 0}
            )
            c["games"] += 1
            if r.truncated:
                c["truncated"] += 1
            elif r.result is GameResult.DRAW:
                c["draws"] += 1
            elif r.result is GameResult.win_for(side):
                c["wins"] += 1
            else:
                c["losses"] += 1

    by_agent_color = {key: ColorStats(**counts) for key, counts in per_key.items()}
    by_agent: dict[str, ColorStats] = {}
    for (name, _side), stats in by_agent_color.items():
        prev = by_agent.get(name, ColorStats())
        by_agent[name] = ColorStats(
            games=prev.games + stats.games,
            wins=prev.wins + stats.wins,
            losses=prev.losses + stats.losses,
            draws=prev.draws + stats.draws,
            truncated=prev.truncated + stats.truncated,
        )
    n = len(records)
    return MatchStats(
        games=n,
        red_wins=red_wins,
        black_wins=black_wins,
        draws=draws,
        truncated=truncated,
        invalid_actions=sum(r.invalid_actions for r in records),
        average_plies=sum(r.plies for r in records) / n if n else 0.0,
        by_agent=by_agent,
        by_agent_color=by_agent_color,
    )
