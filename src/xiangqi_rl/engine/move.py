"""Move and move-record models.

A :class:`Move` is only a source and destination square. It deliberately knows
nothing about which piece moves or whether the move is legal; that depends on
the position and is the job of :mod:`xiangqi_rl.engine.rules`.

For future RL integration a move maps bijectively onto a flat action index::

    action = 90 * from_sq + to_sq      (0 <= action < 8100)

Most of these 8100 actions are never legal; an RL environment is expected to
mask them using the engine's legal move list.
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass

from xiangqi_rl.engine.constants import (
    NUM_SQUARES,
    Side,
    make_square,
    parse_square,
    square_name,
    validate_square,
)
from xiangqi_rl.engine.errors import InvalidMoveFormatError
from xiangqi_rl.engine.pieces import Piece

ACTION_SPACE_SIZE = NUM_SQUARES * NUM_SQUARES

_UCI_PATTERN = re.compile(r"^([a-i][0-9])([a-i][0-9])$")


@dataclass(frozen=True, slots=True, order=True)
class Move:
    """A move from ``from_sq`` to ``to_sq`` (square indices ``0..89``).

    Equality, hashing and ordering are by ``(from_sq, to_sq)`` only, so sorting
    a list of moves gives a deterministic order.
    """

    from_sq: int
    to_sq: int

    def __post_init__(self) -> None:
        # Normalise integer-likes (e.g. numpy ints) to plain ints and validate.
        object.__setattr__(self, "from_sq", validate_square(self.from_sq))
        object.__setattr__(self, "to_sq", validate_square(self.to_sq))
        if self.from_sq == self.to_sq:
            raise InvalidMoveFormatError(
                f"Move source and destination are both {square_name(self.from_sq)}"
            )

    @classmethod
    def from_coords(cls, from_row: int, from_col: int, to_row: int, to_col: int) -> Move:
        """Build a move from ``(row, col)`` coordinates."""
        return cls(make_square(from_row, from_col), make_square(to_row, to_col))

    @classmethod
    def from_uci(cls, text: str) -> Move:
        """Parse UCCI-style notation such as ``'h2e2'`` (Red's central cannon)."""
        match = _UCI_PATTERN.match(text) if isinstance(text, str) else None
        if match is None:
            raise InvalidMoveFormatError(f"Invalid move notation {text!r}")
        return cls(parse_square(match.group(1)), parse_square(match.group(2)))

    @classmethod
    def from_action(cls, action: int) -> Move:
        """Inverse of :meth:`to_action`."""
        if isinstance(action, bool):
            raise InvalidMoveFormatError(f"Action must be an integer, got bool {action!r}")
        try:
            index = operator.index(action)
        except TypeError:
            raise InvalidMoveFormatError(f"Action must be an integer, got {action!r}") from None
        if not 0 <= index < ACTION_SPACE_SIZE:
            raise InvalidMoveFormatError(f"Action {index} outside 0..{ACTION_SPACE_SIZE - 1}")
        from_sq, to_sq = divmod(index, NUM_SQUARES)
        # from_sq == to_sq (e.g. action 0) is not a move; Move() raises for it.
        return cls(from_sq, to_sq)

    def to_action(self) -> int:
        """Flat action index ``90 * from_sq + to_sq``."""
        return self.from_sq * NUM_SQUARES + self.to_sq

    def uci(self) -> str:
        """UCCI-style notation, e.g. ``'h2e2'``."""
        return square_name(self.from_sq) + square_name(self.to_sq)

    def __str__(self) -> str:
        return self.uci()

    def __repr__(self) -> str:
        return f"Move({self.uci()!r})"


@dataclass(frozen=True, slots=True)
class MoveRecord:
    """A move as it was played in a game, with the information needed to
    inspect or replay history.

    Attributes:
        ply: Zero-based index of this move within the game.
        move: The move itself.
        piece: The piece that moved.
        captured: The captured piece, or ``None`` for a quiet move.
        gave_check: Whether the move left the opponent in check.
    """

    ply: int
    move: Move
    piece: Piece
    captured: Piece | None
    gave_check: bool

    @property
    def side(self) -> Side:
        """The side that made the move."""
        return self.piece.side

    @property
    def is_capture(self) -> bool:
        """Whether the move captured a piece."""
        return self.captured is not None
