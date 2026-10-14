"""Reward convention for the two-player zero-sum game.

Rewards are sparse: ``+1`` win, ``-1`` loss, ``0`` draw, and ``0`` on every
non-terminal ply. A reward is always expressed *from a stated perspective*:

* ``perspective=None`` (default, "mover"): from the side that played the move
  this step. The step reward is therefore ``+1`` when the move ended the game
  in the mover's favour, ``-1`` if it lost the game (e.g. perpetual check),
  and ``0`` otherwise. Consecutive steps alternate perspective.
* ``perspective=Side.RED`` / ``Side.BLACK``: from that fixed side, whoever moved.

Truncation by an external step limit yields reward ``0``: no result is
fabricated.
"""

from __future__ import annotations

from xiangqi_rl.engine import GameResult, Side

WIN_REWARD = 1.0
LOSS_REWARD = -1.0
DRAW_REWARD = 0.0


def result_for_side(result: GameResult, side: Side) -> float:
    """Value of ``result`` from ``side``'s point of view (0 while in progress)."""
    if result is GameResult.RED_WINS:
        return WIN_REWARD if side is Side.RED else LOSS_REWARD
    if result is GameResult.BLACK_WINS:
        return WIN_REWARD if side is Side.BLACK else LOSS_REWARD
    return DRAW_REWARD  # DRAW or IN_PROGRESS


def step_reward(result: GameResult, mover: Side, perspective: Side | None = None) -> float:
    """Reward for one step: the result seen from ``perspective`` (default: ``mover``)."""
    return result_for_side(result, mover if perspective is None else perspective)
