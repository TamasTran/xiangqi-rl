"""Pseudo-legal move generation.

A *pseudo-legal* move obeys the moving piece's movement pattern (palace,
river, elephant eye, horse leg, cannon screen, ...) and does not capture a
piece of its own side, but may leave the mover's own general in check or
facing the enemy general. :mod:`xiangqi_rl.engine.rules` filters these into
*legal* moves.

Note on the "flying general": in a legal game the two generals never face
each other on an open file (the side that would expose them is not allowed
to), so a general can never actually capture the opposing general by flying
across the board. The rule is therefore enforced purely as a legality
constraint in :mod:`rules`, and the general's pseudo-legal moves here are only
its one-step palace moves.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from xiangqi_rl.engine.board import Board
from xiangqi_rl.engine.constants import (
    NUM_COLS,
    PieceType,
    Side,
    has_crossed_river,
    in_palace,
    is_on_board,
    on_home_side,
    square_coords,
    validate_square,
)
from xiangqi_rl.engine.move import Move

#: One-step orthogonal directions (row delta, col delta).
ORTHOGONAL_STEPS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
#: One-step diagonal directions.
DIAGONAL_STEPS: tuple[tuple[int, int], ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1))
#: Horse moves as ``(destination offset, leg offset)``, both relative to the
#: horse. The leg is the orthogonally adjacent square in the direction of the
#: two-square component of the move; it must be empty.
HORSE_STEPS: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((-2, -1), (-1, 0)),
    ((-2, 1), (-1, 0)),
    ((2, -1), (1, 0)),
    ((2, 1), (1, 0)),
    ((-1, -2), (0, -1)),
    ((1, -2), (0, -1)),
    ((-1, 2), (0, 1)),
    ((1, 2), (0, 1)),
)


def _square(row: int, col: int) -> int:
    """Unchecked ``(row, col) -> square`` for coordinates already known to be on board."""
    return row * NUM_COLS + col


def _can_land(cells: tuple, square: int, side: Side) -> bool:
    """A piece of ``side`` may end on ``square`` if it is empty or holds an enemy."""
    occupant = cells[square]
    return occupant is None or occupant.side is not side


def _general_targets(cells: tuple, row: int, col: int, side: Side) -> Iterator[int]:
    for dr, dc in ORTHOGONAL_STEPS:
        r, c = row + dr, col + dc
        if in_palace(side, r, c) and _can_land(cells, _square(r, c), side):
            yield _square(r, c)


def _advisor_targets(cells: tuple, row: int, col: int, side: Side) -> Iterator[int]:
    for dr, dc in DIAGONAL_STEPS:
        r, c = row + dr, col + dc
        if in_palace(side, r, c) and _can_land(cells, _square(r, c), side):
            yield _square(r, c)


def _elephant_targets(cells: tuple, row: int, col: int, side: Side) -> Iterator[int]:
    for dr, dc in DIAGONAL_STEPS:
        r, c = row + 2 * dr, col + 2 * dc
        if not is_on_board(r, c) or not on_home_side(side, r):
            continue  # elephants never cross the river
        if cells[_square(row + dr, col + dc)] is not None:
            continue  # the elephant's eye is blocked
        if _can_land(cells, _square(r, c), side):
            yield _square(r, c)


def _horse_targets(cells: tuple, row: int, col: int, side: Side) -> Iterator[int]:
    for (dr, dc), (leg_r, leg_c) in HORSE_STEPS:
        r, c = row + dr, col + dc
        if not is_on_board(r, c):
            continue
        if cells[_square(row + leg_r, col + leg_c)] is not None:
            continue  # the horse's leg is blocked ("hobbled horse")
        if _can_land(cells, _square(r, c), side):
            yield _square(r, c)


def _chariot_targets(cells: tuple, row: int, col: int, side: Side) -> Iterator[int]:
    for dr, dc in ORTHOGONAL_STEPS:
        r, c = row + dr, col + dc
        while is_on_board(r, c):
            occupant = cells[_square(r, c)]
            if occupant is None:
                yield _square(r, c)
            else:
                if occupant.side is not side:
                    yield _square(r, c)  # capture the first enemy piece met
                break
            r, c = r + dr, c + dc


def _cannon_targets(cells: tuple, row: int, col: int, side: Side) -> Iterator[int]:
    for dr, dc in ORTHOGONAL_STEPS:
        r, c = row + dr, col + dc
        # Quiet moves: slide over empty squares like a chariot.
        while is_on_board(r, c) and cells[_square(r, c)] is None:
            yield _square(r, c)
            r, c = r + dr, c + dc
        # (r, c) is now the screen (any piece of either side) or off the board.
        r, c = r + dr, c + dc
        # Capture: the first piece beyond exactly one screen, if it is an enemy.
        while is_on_board(r, c):
            occupant = cells[_square(r, c)]
            if occupant is not None:
                if occupant.side is not side:
                    yield _square(r, c)
                break
            r, c = r + dr, c + dc


def _soldier_targets(cells: tuple, row: int, col: int, side: Side) -> Iterator[int]:
    steps = [(side.forward, 0)]
    if has_crossed_river(side, row):
        steps += [(0, -1), (0, 1)]  # sideways only after crossing; never backward
    for dr, dc in steps:
        r, c = row + dr, col + dc
        if is_on_board(r, c) and _can_land(cells, _square(r, c), side):
            yield _square(r, c)


_TARGET_GENERATORS: dict[PieceType, Callable[[tuple, int, int, Side], Iterator[int]]] = {
    PieceType.GENERAL: _general_targets,
    PieceType.ADVISOR: _advisor_targets,
    PieceType.ELEPHANT: _elephant_targets,
    PieceType.HORSE: _horse_targets,
    PieceType.CHARIOT: _chariot_targets,
    PieceType.CANNON: _cannon_targets,
    PieceType.SOLDIER: _soldier_targets,
}


def pseudo_legal_moves_from(board: Board, square: int) -> list[Move]:
    """Pseudo-legal moves of the piece on ``square`` (empty list if no piece)."""
    square = validate_square(square)
    piece = board.squares[square]
    if piece is None:
        return []
    row, col = square_coords(square)
    targets = _TARGET_GENERATORS[piece.kind](board.squares, row, col, piece.side)
    return [Move(square, target) for target in sorted(targets)]


def generate_pseudo_legal_moves(board: Board, side: Side) -> list[Move]:
    """All pseudo-legal moves for ``side``, sorted by ``(from_sq, to_sq)``."""
    moves: list[Move] = []
    for square, _piece in board.occupied(side):
        moves.extend(pseudo_legal_moves_from(board, square))
    return moves
