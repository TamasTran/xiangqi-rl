"""Xiangqi (Chinese chess) game engine.

The public API is re-exported here::

    from xiangqi_rl.engine import Game, Move, Side

See ``docs/xiangqi_rules.md`` for the coordinate convention and rule policies.
"""

from xiangqi_rl.engine.board import INITIAL_PLACEMENT, Board
from xiangqi_rl.engine.constants import (
    NUM_COLS,
    NUM_ROWS,
    NUM_SQUARES,
    GameResult,
    PieceType,
    Side,
    TerminationReason,
    make_square,
    parse_square,
    square_coords,
    square_name,
)
from xiangqi_rl.engine.errors import (
    GameOverError,
    IllegalMoveError,
    InvalidMoveFormatError,
    InvalidPositionError,
    InvalidSquareError,
    XiangqiError,
)
from xiangqi_rl.engine.game import Game, GameConfig, GameState, position_key
from xiangqi_rl.engine.move import ACTION_SPACE_SIZE, Move, MoveRecord
from xiangqi_rl.engine.pieces import Piece

INITIAL_FEN = f"{INITIAL_PLACEMENT} w - - 0 1"

__all__ = [
    "ACTION_SPACE_SIZE",
    "INITIAL_FEN",
    "INITIAL_PLACEMENT",
    "NUM_COLS",
    "NUM_ROWS",
    "NUM_SQUARES",
    "Board",
    "Game",
    "GameConfig",
    "GameOverError",
    "GameResult",
    "GameState",
    "IllegalMoveError",
    "InvalidMoveFormatError",
    "InvalidPositionError",
    "InvalidSquareError",
    "Move",
    "MoveRecord",
    "Piece",
    "PieceType",
    "Side",
    "TerminationReason",
    "XiangqiError",
    "make_square",
    "parse_square",
    "position_key",
    "square_coords",
    "square_name",
]
