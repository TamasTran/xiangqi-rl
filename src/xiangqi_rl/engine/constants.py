"""Sides, piece types, results, and board geometry for Xiangqi.

Coordinate convention (used everywhere in the engine)
-----------------------------------------------------
* The board has ``NUM_ROWS = 10`` rows and ``NUM_COLS = 9`` columns (files).
* Row 0 is the top of the board (Black's back rank); row 9 is the bottom
  (Red's back rank).
* Column 0 is the leftmost file as seen from Red's side of the board.
* A *square* is a plain ``int`` in ``0..89``: ``square = row * NUM_COLS + col``.
* Red starts at the bottom and moves "forward" toward decreasing rows; Black
  starts at the top and moves toward increasing rows.
* Black's half of the board is rows 0-4, Red's half is rows 5-9; the river lies
  between rows 4 and 5.

Square names use the common UCCI-style notation: a file letter ``a``-``i``
(column 0-8) followed by a rank digit ``0``-``9`` counted from Red's back rank,
so ``rank = 9 - row``. Red's general starts on ``e0`` and Black's on ``e9``.
"""

from __future__ import annotations

import operator
from enum import Enum, IntEnum

from xiangqi_rl.engine.errors import InvalidSquareError

NUM_ROWS = 10
NUM_COLS = 9
NUM_SQUARES = NUM_ROWS * NUM_COLS

FILE_LETTERS = "abcdefghi"


class Side(IntEnum):
    """The two players. Integer values are stable and usable as array indices."""

    RED = 0
    BLACK = 1

    @property
    def opponent(self) -> Side:
        """The other side."""
        return Side.BLACK if self is Side.RED else Side.RED

    @property
    def forward(self) -> int:
        """Row delta of one step "forward" (toward the enemy) for this side."""
        return -1 if self is Side.RED else 1


class PieceType(IntEnum):
    """The seven Xiangqi piece types. Integer values are stable."""

    GENERAL = 0
    ADVISOR = 1
    ELEPHANT = 2
    HORSE = 3
    CHARIOT = 4
    CANNON = 5
    SOLDIER = 6


class GameResult(Enum):
    """Outcome of a game from a neutral point of view."""

    IN_PROGRESS = "in_progress"
    RED_WINS = "red_wins"
    BLACK_WINS = "black_wins"
    DRAW = "draw"

    @staticmethod
    def win_for(side: Side) -> GameResult:
        """The result in which ``side`` wins."""
        return GameResult.RED_WINS if side is Side.RED else GameResult.BLACK_WINS


class TerminationReason(Enum):
    """Why a game ended. See ``docs/xiangqi_rules.md`` for the exact policy."""

    CHECKMATE = "checkmate"
    """Side to move is in check and has no legal move: it loses."""
    STALEMATE = "stalemate"
    """Side to move is not in check but has no legal move: it also loses."""
    PERPETUAL_CHECK = "perpetual_check"
    """Repetition where exactly one side checked on every move: that side loses."""
    REPETITION = "repetition"
    """Repetition without a sole perpetual checker: draw."""
    MAX_PLIES = "max_plies"
    """Configured ply limit reached: draw."""


PALACE_COLS = range(3, 6)
_PALACE_ROWS = {Side.RED: range(7, 10), Side.BLACK: range(0, 3)}
_HOME_ROWS = {Side.RED: range(5, 10), Side.BLACK: range(0, 5)}


def _as_int(value: object, what: str) -> int:
    """Coerce an integer-like value (incl. numpy ints) to ``int``; reject bools."""
    if isinstance(value, bool):
        raise InvalidSquareError(f"{what} must be an integer, got bool {value!r}")
    try:
        return operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise InvalidSquareError(f"{what} must be an integer, got {value!r}") from None


def is_on_board(row: int, col: int) -> bool:
    """Whether ``(row, col)`` lies on the 10x9 board."""
    return 0 <= row < NUM_ROWS and 0 <= col < NUM_COLS


def make_square(row: int, col: int) -> int:
    """Convert ``(row, col)`` to a square index, validating the coordinates."""
    row = _as_int(row, "row")
    col = _as_int(col, "col")
    if not is_on_board(row, col):
        raise InvalidSquareError(f"Coordinates ({row}, {col}) are off the board")
    return row * NUM_COLS + col


def validate_square(square: object) -> int:
    """Return ``square`` as an ``int`` if it is a valid index, else raise."""
    index = _as_int(square, "square")
    if not 0 <= index < NUM_SQUARES:
        raise InvalidSquareError(f"Square {index} is outside 0..{NUM_SQUARES - 1}")
    return index


def square_row(square: int) -> int:
    """Row of a (valid) square index."""
    return square // NUM_COLS


def square_col(square: int) -> int:
    """Column of a (valid) square index."""
    return square % NUM_COLS


def square_coords(square: int) -> tuple[int, int]:
    """``(row, col)`` of a (valid) square index."""
    return divmod(square, NUM_COLS)


def in_palace(side: Side, row: int, col: int) -> bool:
    """Whether ``(row, col)`` lies inside ``side``'s 3x3 palace."""
    return row in _PALACE_ROWS[side] and col in PALACE_COLS


def on_home_side(side: Side, row: int) -> bool:
    """Whether ``row`` is on ``side``'s own half of the river."""
    return row in _HOME_ROWS[side]


def has_crossed_river(side: Side, row: int) -> bool:
    """Whether a piece of ``side`` on ``row`` is across the river (enemy half)."""
    return not on_home_side(side, row)


def square_name(square: int) -> str:
    """Name of a square in UCCI-style notation, e.g. ``0 -> 'a9'``, ``85 -> 'e0'``."""
    row, col = square_coords(validate_square(square))
    return f"{FILE_LETTERS[col]}{NUM_ROWS - 1 - row}"


def parse_square(name: str) -> int:
    """Parse a square name such as ``'e0'`` into a square index."""
    if not isinstance(name, str) or len(name) != 2:
        raise InvalidSquareError(f"Invalid square name {name!r}")
    file_char, rank_char = name[0], name[1]
    if file_char not in FILE_LETTERS or not rank_char.isdigit():
        raise InvalidSquareError(f"Invalid square name {name!r}")
    col = FILE_LETTERS.index(file_char)
    row = NUM_ROWS - 1 - int(rank_char)
    return make_square(row, col)
