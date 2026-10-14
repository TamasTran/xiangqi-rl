"""The standard starting position."""

from __future__ import annotations

import pytest

from engine_helpers import ucis
from xiangqi_rl.engine import (
    INITIAL_FEN,
    NUM_COLS,
    NUM_ROWS,
    NUM_SQUARES,
    Board,
    Game,
    GameResult,
    Piece,
    PieceType,
    Side,
    parse_square,
)
from xiangqi_rl.engine.rules import generate_legal_moves

# Expected placement written out square by square (not derived from the FEN
# constant the engine uses).
EXPECTED_PLACEMENT = {
    # Red, bottom of the board (ranks 0-3)
    "a0": "R", "b0": "N", "c0": "B", "d0": "A", "e0": "K", "f0": "A", "g0": "B", "h0": "N", "i0": "R",
    "b2": "C", "h2": "C",
    "a3": "P", "c3": "P", "e3": "P", "g3": "P", "i3": "P",
    # Black, top of the board (ranks 6-9)
    "a9": "r", "b9": "n", "c9": "b", "d9": "a", "e9": "k", "f9": "a", "g9": "b", "h9": "n", "i9": "r",
    "b7": "c", "h7": "c",
    "a6": "p", "c6": "p", "e6": "p", "g6": "p", "i6": "p",
}  # fmt: skip

# Red's 44 legal opening moves, enumerated by hand:
#   chariots a0/i0: two steps up each (blocked by own soldier on rank 3)  -> 4
#   horses b0/h0: two forward jumps each (sideways jump blocked by the
#     elephant on the leg square)                                          -> 4
#   elephants c0/g0: two diagonal jumps each                              -> 4
#   advisors d0/f0: only to the palace centre e1                          -> 2
#   general: e1                                                           -> 1
#   cannons b2/h2: 4 quiet up, 1 down, 1 + 5 sideways, plus the capture of
#     the black horse over the black cannon screen = 12 each              -> 24
#   soldiers: one step forward each                                       -> 5
EXPECTED_OPENING_MOVES = {
    "a0a1", "a0a2", "i0i1", "i0i2",
    "b0a2", "b0c2", "h0g2", "h0i2",
    "c0a2", "c0e2", "g0e2", "g0i2",
    "d0e1", "f0e1", "e0e1",
    "b2b3", "b2b4", "b2b5", "b2b6", "b2b9", "b2b1", "b2a2",
    "b2c2", "b2d2", "b2e2", "b2f2", "b2g2",
    "h2h3", "h2h4", "h2h5", "h2h6", "h2h9", "h2h1", "h2i2",
    "h2g2", "h2f2", "h2e2", "h2d2", "h2c2",
    "a3a4", "c3c4", "e3e4", "g3g4", "i3i4",
}  # fmt: skip


def perft(board: Board, side: Side, depth: int) -> int:
    """Count leaf nodes of the legal-move tree (standard engine test)."""
    moves = generate_legal_moves(board, side)
    if depth == 1:
        return len(moves)
    return sum(perft(board.apply_move(m), side.opponent, depth - 1) for m in moves)


def test_board_dimensions() -> None:
    assert (NUM_ROWS, NUM_COLS, NUM_SQUARES) == (10, 9, 90)
    assert len(Board.initial().squares) == 90


def test_each_side_has_sixteen_pieces_with_standard_counts() -> None:
    board = Board.initial()
    expected = {
        PieceType.GENERAL: 1,
        PieceType.ADVISOR: 2,
        PieceType.ELEPHANT: 2,
        PieceType.HORSE: 2,
        PieceType.CHARIOT: 2,
        PieceType.CANNON: 2,
        PieceType.SOLDIER: 5,
    }
    for side in Side:
        assert board.count(side) == 16
        for kind, count in expected.items():
            assert board.count(side, kind) == count, (side, kind)


def test_piece_locations_match_standard_setup() -> None:
    board = Board.initial()
    expected = {parse_square(name): Piece.from_symbol(sym) for name, sym in EXPECTED_PLACEMENT.items()}
    for square in range(90):
        assert board.piece_at(square) == expected.get(square), square


def test_red_moves_first_and_game_in_progress() -> None:
    game = Game.new_game()
    assert game.side_to_move is Side.RED
    assert game.get_result() is GameResult.IN_PROGRESS
    assert not game.is_in_check(Side.RED) and not game.is_in_check(Side.BLACK)
    assert game.to_fen() == INITIAL_FEN


def test_initial_legal_moves_match_hand_enumeration() -> None:
    assert len(EXPECTED_OPENING_MOVES) == 44
    assert ucis(Game.new_game().legal_moves()) == EXPECTED_OPENING_MOVES


def test_black_initial_moves_mirror_red() -> None:
    # Black's options from the start position are Red's mirrored vertically.
    board = Board.initial()
    black = ucis(generate_legal_moves(board, Side.BLACK))
    mirrored = {f"{m[0]}{9 - int(m[1])}{m[2]}{9 - int(m[3])}" for m in EXPECTED_OPENING_MOVES}
    assert black == mirrored


@pytest.mark.parametrize(("depth", "nodes"), [(1, 44), (2, 1920), (3, 79666)])
def test_perft_matches_published_reference(depth: int, nodes: int) -> None:
    # Reference values for the standard Xiangqi start position, widely
    # published for engine verification (e.g. the Chess Programming Wiki's
    # Chinese chess perft table): 44, 1920, 79666, 3290240.
    assert perft(Board.initial(), Side.RED, depth) == nodes


@pytest.mark.slow
def test_perft_depth_4() -> None:
    assert perft(Board.initial(), Side.RED, 4) == 3290240
