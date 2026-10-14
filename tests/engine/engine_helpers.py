"""Shared helpers for the engine tests.

Includes a deliberately naive, independently written rules reference
(``naive_*`` functions). It decides moves with a per-move geometric predicate
instead of the engine's generators and reverse-lookup attack tables, so
agreement between the two is meaningful evidence of correctness.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping

from xiangqi_rl.engine import Board, Game, GameConfig, Move, PieceType, Side
from xiangqi_rl.engine.move_generator import pseudo_legal_moves_from


def board_from(pieces: Mapping[str, str]) -> Board:
    """``board_from({"e0": "K", "e9": "k"})``."""
    return Board.from_pieces(pieces)


def game_from(pieces: Mapping[str, str], side: Side = Side.RED, **config: object) -> Game:
    """A validated game from a ``{square name: FEN symbol}`` mapping."""
    return Game(board_from(pieces), side, GameConfig(**config) if config else None)


def ucis(moves: object) -> set[str]:
    """Set of UCCI strings for an iterable of moves."""
    return {move.uci() for move in moves}  # type: ignore[attr-defined]


def targets_from(board: Board, square_name: str) -> set[str]:
    """Destination square names of the pseudo-legal moves from one square."""
    from xiangqi_rl.engine import parse_square

    return {move.uci()[2:] for move in pseudo_legal_moves_from(board, parse_square(square_name))}


def legal_from(game: Game, square_name: str) -> set[str]:
    """Destination square names of the legal moves from one square."""
    return {m.uci()[2:] for m in game.legal_moves() if m.uci()[:2] == square_name}


# ----------------------------------------------------------------------------
# Naive reference implementation (independent of the engine's rules module).
# ----------------------------------------------------------------------------


def _grid(board: Board) -> list[list]:
    return [list(board.squares[r * 9 : r * 9 + 9]) for r in range(10)]


def _in_palace(side: Side, r: int, c: int) -> bool:
    rows = (7, 8, 9) if side is Side.RED else (0, 1, 2)
    return r in rows and 3 <= c <= 5


def _own_half(side: Side, r: int) -> bool:
    return r >= 5 if side is Side.RED else r <= 4


def _count_between(grid: list[list], fr: int, fc: int, tr: int, tc: int) -> int:
    """Pieces strictly between two squares on the same row or column."""
    if fr == tr:
        lo, hi = sorted((fc, tc))
        return sum(grid[fr][c] is not None for c in range(lo + 1, hi))
    lo, hi = sorted((fr, tr))
    return sum(grid[r][fc] is not None for r in range(lo + 1, hi))


def naive_can_move(grid: list[list], fr: int, fc: int, tr: int, tc: int) -> bool:
    """Geometric pseudo-legality of moving the piece on (fr, fc) to (tr, tc)."""
    piece = grid[fr][fc]
    if piece is None or (fr, fc) == (tr, tc):
        return False
    target = grid[tr][tc]
    if target is not None and target.side is piece.side:
        return False
    dr, dc = tr - fr, tc - fc
    side, kind = piece.side, piece.kind
    if kind is PieceType.GENERAL:
        return abs(dr) + abs(dc) == 1 and _in_palace(side, tr, tc)
    if kind is PieceType.ADVISOR:
        return abs(dr) == 1 and abs(dc) == 1 and _in_palace(side, tr, tc)
    if kind is PieceType.ELEPHANT:
        return (
            abs(dr) == 2
            and abs(dc) == 2
            and _own_half(side, tr)
            and grid[fr + dr // 2][fc + dc // 2] is None
        )
    if kind is PieceType.HORSE:
        if sorted((abs(dr), abs(dc))) != [1, 2]:
            return False
        leg = (fr + dr // 2, fc) if abs(dr) == 2 else (fr, fc + dc // 2)
        return grid[leg[0]][leg[1]] is None
    if kind in (PieceType.CHARIOT, PieceType.CANNON):
        if (dr == 0) == (dc == 0):
            return False
        between = _count_between(grid, fr, fc, tr, tc)
        if kind is PieceType.CHARIOT:
            return between == 0
        return between == 0 if target is None else between == 1
    if kind is PieceType.SOLDIER:
        forward = -1 if side is Side.RED else 1
        if dc == 0 and dr == forward:
            return True
        crossed = not _own_half(side, fr)
        return crossed and dr == 0 and abs(dc) == 1
    raise AssertionError(kind)


def naive_in_check(grid: list[list], side: Side) -> bool:
    """General attacked by any enemy piece's geometry, or generals facing."""
    spots = {
        p.side: (r, c)
        for r in range(10)
        for c in range(9)
        if (p := grid[r][c]) is not None and p.kind is PieceType.GENERAL
    }
    gr, gc = spots[side]
    er, ec = spots[side.opponent]
    if gc == ec and _count_between(grid, gr, gc, er, gc) == 0:
        return True
    return any(
        naive_can_move(grid, r, c, gr, gc)
        for r in range(10)
        for c in range(9)
        if grid[r][c] is not None and grid[r][c].side is side.opponent
    )


def naive_legal_moves(board: Board, side: Side) -> set[str]:
    """Legal moves by brute force over every (source, destination) pair."""
    grid = _grid(board)
    result = set()
    for fr in range(10):
        for fc in range(9):
            piece = grid[fr][fc]
            if piece is None or piece.side is not side:
                continue
            for tr in range(10):
                for tc in range(9):
                    if not naive_can_move(grid, fr, fc, tr, tc):
                        continue
                    after = [row[:] for row in grid]
                    after[tr][tc], after[fr][fc] = piece, None
                    if not naive_in_check(after, side):
                        result.add(Move.from_coords(fr, fc, tr, tc).uci())
    return result


def random_playout_states(
    seed: int, games: int, max_plies: int
) -> Iterator[tuple[Game, int]]:
    """Yield ``(game, game_index)`` after every position of seeded random games.

    Moves are chosen uniformly from the engine's legal moves with a fixed
    seed, so the sequence is reproducible.
    """
    rng = random.Random(seed)
    for index in range(games):
        game = Game.new_game(GameConfig(max_plies=max_plies))
        yield game, index
        while not game.is_terminal():
            game.make_move(rng.choice(game.legal_moves()))
            yield game, index
