"""Action encoding: ``action = 90 * source_square + destination_square``.

The action space is deliberately much larger (8100) than the number of legal
moves in any position (at most a few dozen). Which actions are allowed right
now is described by the *legal-action mask*, built from the engine's own legal
move list so that it can never disagree with the engine.

Squares are the engine's indices ``row * 9 + col`` (row 0 = Black's back rank).
"""

from __future__ import annotations

import operator

import numpy as np

from xiangqi_rl.engine import (
    ACTION_SPACE_SIZE,
    NUM_COLS,
    NUM_ROWS,
    NUM_SQUARES,
    Game,
    InvalidSquareError,
    Move,
    make_square,
    square_coords,
)

ACTION_DIM = ACTION_SPACE_SIZE
"""Size of the discrete action space (90 * 90 = 8100)."""

__all__ = [
    "ACTION_DIM",
    "action_to_move",
    "index_to_square",
    "legal_action_mask",
    "move_to_action",
    "square_to_index",
]


def square_to_index(row: int, col: int) -> int:
    """``(row, col) -> 0..89``; raises :class:`InvalidSquareError` if off the board."""
    return make_square(row, col)


def index_to_square(index: int) -> tuple[int, int]:
    """``0..89 -> (row, col)``; raises :class:`InvalidSquareError` if out of range."""
    if isinstance(index, bool):
        raise InvalidSquareError(f"Square index must be an integer, got bool {index!r}")
    try:
        value = operator.index(index)
    except TypeError:
        raise InvalidSquareError(f"Square index must be an integer, got {index!r}") from None
    if not 0 <= value < NUM_SQUARES:
        raise InvalidSquareError(f"Square index {value} outside 0..{NUM_SQUARES - 1}")
    row, col = square_coords(value)
    assert 0 <= row < NUM_ROWS and 0 <= col < NUM_COLS
    return row, col


def move_to_action(move: Move) -> int:
    """Encode a move as ``90 * from_sq + to_sq``."""
    return move.to_action()


def action_to_move(action: int) -> Move:
    """Decode an action index into a :class:`Move`.

    Accepts Python and NumPy integers only. Floats, bools, strings and
    out-of-range values (and the 90 "null" actions whose source equals the
    destination) raise :class:`InvalidMoveFormatError`; nothing is coerced.
    """
    return Move.from_action(action)


def legal_action_mask(game: Game) -> np.ndarray:
    """Boolean mask of shape ``(ACTION_DIM,)``; ``True`` marks a legal action.

    All ``False`` for a finished game (there are no legal moves, and none is
    invented). A fresh array is returned on every call.
    """
    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    for move in game.legal_moves():
        mask[move.to_action()] = True
    return mask


