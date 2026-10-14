"""Attack detection, check, and legal-move filtering.

Definitions used by the engine
------------------------------
* **Attack** (:func:`is_square_attacked`): a piece of side *S* attacks square
  *q* if, by its movement pattern on the current board, it could move to *q*
  were *q* occupied by an enemy piece. Pins are irrelevant: a piece pinned
  against its own general still attacks, because in Xiangqi a general may
  never be left capturable. Attacks are computed by looking *outward from the
  target square* (reverse lookup), not by generating the opponent's legal
  moves; that would wrongly ignore pinned attackers and would recurse.
* **Flying general** (:func:`generals_facing`): the two generals may never
  stand on the same file with no piece between them. This is modelled as a
  separate predicate rather than as a long-range "attack" of the general.
* **Check** (:func:`is_in_check`): a side is in check if its general is
  attacked, *or* the generals face each other.
* **Legal move**: a pseudo-legal move after which the mover is not in check.
  Legality is always decided on the *post-move* board. This automatically
  handles pins, discovered attacks, a general stepping along a line it is
  attacked on, a moving piece becoming (or ceasing to be) a cannon screen,
  and exposure of the generals to each other.
"""

from __future__ import annotations

from collections.abc import Iterator

from xiangqi_rl.engine.board import Board
from xiangqi_rl.engine.constants import (
    NUM_COLS,
    PieceType,
    Side,
    has_crossed_river,
    in_palace,
    is_on_board,
    on_home_side,
    square_col,
    square_coords,
    square_name,
    square_row,
    validate_square,
)
from xiangqi_rl.engine.errors import InvalidPositionError
from xiangqi_rl.engine.move import Move
from xiangqi_rl.engine.move_generator import (
    DIAGONAL_STEPS,
    HORSE_STEPS,
    ORTHOGONAL_STEPS,
    generate_pseudo_legal_moves,
    pseudo_legal_moves_from,
)


def _square(row: int, col: int) -> int:
    return row * NUM_COLS + col


def generals_facing(board: Board) -> bool:
    """Whether both generals are on the same file with no piece between them."""
    return _facing(board, board.find_general(Side.RED), board.find_general(Side.BLACK))


def _facing(board: Board, general_a: int | None, general_b: int | None) -> bool:
    if general_a is None or general_b is None or square_col(general_a) != square_col(general_b):
        return False
    col = square_col(general_a)
    low, high = sorted((square_row(general_a), square_row(general_b)))
    cells = board.squares
    return all(cells[_square(row, col)] is None for row in range(low + 1, high))


def _iter_attackers(board: Board, square: int, by_side: Side) -> Iterator[int]:
    """Yield the square of each ``by_side`` piece attacking ``square`` (each once)."""
    cells = board.squares
    tr, tc = square_coords(square)

    def holds(r: int, c: int, kind: PieceType) -> bool:
        piece = cells[_square(r, c)]
        return piece is not None and piece.side is by_side and piece.kind is kind

    # Chariots and cannons: walk outward along each ray from the target.
    # The first piece met attacks if it is a chariot; the second piece met
    # attacks if it is a cannon (the first piece being its screen).
    for dr, dc in ORTHOGONAL_STEPS:
        r, c = tr + dr, tc + dc
        pieces_seen = 0
        while is_on_board(r, c) and pieces_seen < 2:
            piece = cells[_square(r, c)]
            if piece is not None:
                pieces_seen += 1
                if piece.side is by_side:
                    if pieces_seen == 1 and piece.kind is PieceType.CHARIOT:
                        yield _square(r, c)
                    elif pieces_seen == 2 and piece.kind is PieceType.CANNON:
                        yield _square(r, c)
            r, c = r + dr, c + dc

    # Horses: a horse at target - offset reaches the target if *its* leg
    # (adjacent to the horse, not to the target) is empty.
    for (dr, dc), (leg_r, leg_c) in HORSE_STEPS:
        hr, hc = tr - dr, tc - dc
        if is_on_board(hr, hc) and holds(hr, hc, PieceType.HORSE):
            if cells[_square(hr + leg_r, hc + leg_c)] is None:
                yield _square(hr, hc)

    # Soldiers: one step forward, or sideways once across the river. The
    # sideways attacker stands on the target's row, so the river test uses it.
    sr = tr - by_side.forward
    if is_on_board(sr, tc) and holds(sr, tc, PieceType.SOLDIER):
        yield _square(sr, tc)
    if has_crossed_river(by_side, tr):
        for sc in (tc - 1, tc + 1):
            if is_on_board(tr, sc) and holds(tr, sc, PieceType.SOLDIER):
                yield _square(tr, sc)

    # General (one orthogonal step) and advisors (one diagonal step) can only
    # attack squares inside their own palace.
    if in_palace(by_side, tr, tc):
        for dr, dc in ORTHOGONAL_STEPS:
            r, c = tr - dr, tc - dc
            if is_on_board(r, c) and holds(r, c, PieceType.GENERAL):
                yield _square(r, c)
        for dr, dc in DIAGONAL_STEPS:
            r, c = tr - dr, tc - dc
            if is_on_board(r, c) and holds(r, c, PieceType.ADVISOR):
                yield _square(r, c)

    # Elephants: two diagonal steps with an empty eye, own half only.
    if on_home_side(by_side, tr):
        for dr, dc in DIAGONAL_STEPS:
            er, ec = tr - 2 * dr, tc - 2 * dc
            if is_on_board(er, ec) and holds(er, ec, PieceType.ELEPHANT):
                if cells[_square(tr - dr, tc - dc)] is None:
                    yield _square(er, ec)


def attackers_of(board: Board, square: int, by_side: Side) -> list[int]:
    """Sorted squares of ``by_side`` pieces attacking ``square``.

    The flying-general relation is *not* included; see :func:`generals_facing`.
    """
    return sorted(set(_iter_attackers(board, validate_square(square), by_side)))


def is_square_attacked(board: Board, square: int, by_side: Side) -> bool:
    """Whether any ``by_side`` piece attacks ``square`` (pins ignored).

    For an empty square this answers "could ``by_side`` capture an enemy
    piece placed there?" (e.g. a cannon needs a screen). Legality checks only
    ever query the square of a general on a post-move board.
    """
    return next(_iter_attackers(board, validate_square(square), by_side), None) is not None


def is_in_check(board: Board, side: Side) -> bool:
    """Whether ``side``'s general is attacked or faces the enemy general."""
    general = board.find_general(side)
    if general is None:
        raise InvalidPositionError(f"{side.name} has no general on the board")
    if _facing(board, general, board.find_general(side.opponent)):
        return True
    return next(_iter_attackers(board, general, side.opponent), None) is not None


def leaves_general_safe(board: Board, move: Move, side: Side) -> bool:
    """Whether playing (pseudo-legal) ``move`` leaves ``side`` out of check."""
    return not is_in_check(board.apply_move(move), side)


def generate_legal_moves(board: Board, side: Side) -> list[Move]:
    """All legal moves for ``side``, sorted by ``(from_sq, to_sq)``."""
    return [
        move
        for move in generate_pseudo_legal_moves(board, side)
        if leaves_general_safe(board, move, side)
    ]


def has_legal_move(board: Board, side: Side) -> bool:
    """Whether ``side`` has at least one legal move (stops at the first one)."""
    return any(
        leaves_general_safe(board, move, side)
        for move in generate_pseudo_legal_moves(board, side)
    )


def illegal_move_reason(board: Board, side: Side, move: Move) -> str | None:
    """Explain why ``move`` is illegal for ``side`` here, or ``None`` if legal."""
    piece = board.squares[move.from_sq]
    if piece is None:
        return f"no piece on {square_name(move.from_sq)}"
    if piece.side is not side:
        return f"the piece on {square_name(move.from_sq)} belongs to {piece.side.name}"
    target = board.squares[move.to_sq]
    if target is not None and target.side is side:
        return "cannot capture a piece of one's own side"
    if move not in pseudo_legal_moves_from(board, move.from_sq):
        return f"a {piece.kind.name.lower()} cannot move from {square_name(move.from_sq)} to {square_name(move.to_sq)}"
    after = board.apply_move(move)
    if generals_facing(after):
        return "the two generals would face each other"
    if is_in_check(after, side):
        return "it would leave the general in check"
    return None


__all__ = [
    "attackers_of",
    "generals_facing",
    "generate_legal_moves",
    "has_legal_move",
    "illegal_move_reason",
    "is_in_check",
    "is_square_attacked",
    "leaves_general_safe",
]
