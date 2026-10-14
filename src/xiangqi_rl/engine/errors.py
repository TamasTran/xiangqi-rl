"""Exception hierarchy for the Xiangqi engine.

Every engine-specific error derives from :class:`XiangqiError` so callers (for
example a future RL environment wrapper) can catch engine failures precisely
without resorting to broad ``except Exception`` clauses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from xiangqi_rl.engine.move import Move


class XiangqiError(Exception):
    """Base class for all engine errors."""


class InvalidSquareError(XiangqiError, ValueError):
    """A square index, coordinate pair, or square name is outside the board."""


class InvalidMoveFormatError(XiangqiError, ValueError):
    """A move is malformed (bad notation, identical source and destination, ...)."""


class InvalidPositionError(XiangqiError, ValueError):
    """A board/position cannot occur in a legal game (e.g. missing general)."""


class IllegalMoveError(XiangqiError):
    """A well-formed move that is not legal in the current position.

    Attributes:
        move: The rejected move.
        reason: Human-readable explanation of why the move is illegal.
    """

    def __init__(self, move: Move, reason: str) -> None:
        self.move = move
        self.reason = reason
        super().__init__(f"Illegal move {move}: {reason}")


class GameOverError(XiangqiError):
    """A move was submitted to a game that has already ended."""
