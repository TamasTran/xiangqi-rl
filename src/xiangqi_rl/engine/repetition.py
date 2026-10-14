"""Repetition adjudication.

This module is intentionally isolated from :class:`~xiangqi_rl.engine.game.Game`
so the simplified policy below can later be swapped for a full implementation
of the official (Asian / Chinese) perpetual-check and perpetual-chase rules.

Implemented policy (deterministic)
----------------------------------
A *position* is the piece placement plus the side to move (the position key).

1. Repetition is adjudicated when the current position has occurred
   ``threshold`` times (default 3, counting the game's starting position).
2. The *window* is every move played from the earliest of those
   ``threshold`` occurrences up to now.
3. If ``perpetual_check_loses`` is true and exactly one side gave check with
   **every** one of its moves in the window, that side loses (perpetual check).
4. Otherwise the game is a draw. This includes mutual perpetual check.

Not implemented: perpetual *chase* (repeatedly attacking an unprotected
piece), "idle" versus "forcing" move classification, and the detailed
tournament adjudication tables. A perpetual chase is therefore scored as a
draw by this policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from xiangqi_rl.engine.constants import GameResult, Side, TerminationReason


@dataclass(frozen=True, slots=True)
class RepetitionVerdict:
    """The outcome of a repetition adjudication."""

    result: GameResult
    reason: TerminationReason


def adjudicate_repetition(
    position_keys: Sequence[str],
    movers: Sequence[Side],
    gave_check: Sequence[bool],
    threshold: int = 3,
    perpetual_check_loses: bool = True,
) -> RepetitionVerdict | None:
    """Adjudicate the latest position of a game, or return ``None`` to continue.

    Args:
        position_keys: Key of the position after each ply; ``position_keys[0]``
            is the starting position, so ``len(position_keys) == plies + 1``.
        movers: ``movers[i]`` is the side that played move ``i`` (from position
            ``i`` to position ``i + 1``).
        gave_check: ``gave_check[i]`` is whether move ``i`` left the opponent in check.
        threshold: Number of occurrences of the same position that triggers
            adjudication (at least 2).
        perpetual_check_loses: Whether a sole perpetual checker loses
            (otherwise every repetition is a draw).
    """
    if threshold < 2:
        raise ValueError(f"threshold must be at least 2, got {threshold}")
    if len(movers) != len(gave_check) or len(position_keys) != len(movers) + 1:
        raise ValueError("position_keys must have exactly one more entry than movers/gave_check")

    current = position_keys[-1]
    occurrences = [index for index, key in enumerate(position_keys) if key == current]
    if len(occurrences) < threshold:
        return None

    if perpetual_check_loses:
        window = range(occurrences[-threshold], len(movers))
        perpetual = {
            side: _checked_every_move(side, window, movers, gave_check) for side in Side
        }
        if perpetual[Side.RED] and not perpetual[Side.BLACK]:
            return RepetitionVerdict(GameResult.BLACK_WINS, TerminationReason.PERPETUAL_CHECK)
        if perpetual[Side.BLACK] and not perpetual[Side.RED]:
            return RepetitionVerdict(GameResult.RED_WINS, TerminationReason.PERPETUAL_CHECK)
    return RepetitionVerdict(GameResult.DRAW, TerminationReason.REPETITION)


def _checked_every_move(
    side: Side, window: range, movers: Sequence[Side], gave_check: Sequence[bool]
) -> bool:
    flags = [gave_check[i] for i in window if movers[i] is side]
    return bool(flags) and all(flags)
