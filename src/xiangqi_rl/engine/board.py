"""Immutable board representation.

A :class:`Board` is a tuple of 90 entries (``Piece`` or ``None``) indexed by
square (see :mod:`xiangqi_rl.engine.constants` for the coordinate convention).
Boards are immutable: :meth:`Board.apply_move` returns a *new* board. This makes
it impossible for one game state, a clone, or a search simulation to mutate
another by accident, at the cost of copying 90 references per move, which is
negligible for Phase 1.

The board knows piece placement only. It does not know whose turn it is and it
does not enforce movement rules; see :mod:`move_generator` and :mod:`rules`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping

from xiangqi_rl.engine.constants import (
    NUM_COLS,
    NUM_ROWS,
    NUM_SQUARES,
    PieceType,
    Side,
    has_crossed_river,
    in_palace,
    parse_square,
    square_coords,
    square_name,
    validate_square,
)
from xiangqi_rl.engine.errors import IllegalMoveError, InvalidPositionError
from xiangqi_rl.engine.move import Move
from xiangqi_rl.engine.pieces import STARTING_PIECE_COUNTS, Piece

#: FEN piece placement of the standard starting position (Black on top).
INITIAL_PLACEMENT = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"

# Squares a piece can ever legally occupy, per side. Advisors are confined to
# the five palace points on the diagonals; elephants to seven points on their
# own half. Used only to validate custom positions.
_ADVISOR_SQUARES = {
    Side.BLACK: frozenset({(0, 3), (0, 5), (1, 4), (2, 3), (2, 5)}),
    Side.RED: frozenset({(9, 3), (9, 5), (8, 4), (7, 3), (7, 5)}),
}
_ELEPHANT_SQUARES = {
    Side.BLACK: frozenset({(0, 2), (0, 6), (2, 0), (2, 4), (2, 8), (4, 2), (4, 6)}),
    Side.RED: frozenset({(9, 2), (9, 6), (7, 0), (7, 4), (7, 8), (5, 2), (5, 6)}),
}
# Soldiers start on these rows and can never move backward.
_SOLDIER_START_ROW = {Side.RED: 6, Side.BLACK: 3}
_PALACE_SQUARES = {
    side: tuple(
        row * NUM_COLS + col
        for row in range(NUM_ROWS)
        for col in range(NUM_COLS)
        if in_palace(side, row, col)
    )
    for side in Side
}


class Board:
    """An immutable 10x9 Xiangqi board.

    Construct boards with :meth:`initial`, :meth:`empty`, :meth:`from_pieces`
    or :meth:`from_fen_placement`. Equality and hashing are by placement.
    """

    __slots__ = ("_squares",)

    _squares: tuple[Piece | None, ...]

    def __init__(self, squares: Iterable[Piece | None]) -> None:
        cells = tuple(squares)
        if len(cells) != NUM_SQUARES:
            raise InvalidPositionError(f"A board needs {NUM_SQUARES} squares, got {len(cells)}")
        for index, cell in enumerate(cells):
            if cell is not None and not isinstance(cell, Piece):
                raise TypeError(f"Square {index} holds {cell!r}, expected Piece or None")
        self._squares = cells

    @classmethod
    def _from_trusted(cls, cells: tuple[Piece | None, ...]) -> Board:
        """Construct without re-validating (internal use on already-valid data)."""
        board = cls.__new__(cls)
        board._squares = cells
        return board

    # ------------------------------------------------------------------ factories

    @classmethod
    def empty(cls) -> Board:
        """A board with no pieces."""
        return cls._from_trusted((None,) * NUM_SQUARES)

    @classmethod
    def initial(cls) -> Board:
        """The standard starting position."""
        return cls.from_fen_placement(INITIAL_PLACEMENT)

    @classmethod
    def from_pieces(cls, placement: Mapping[int | str, Piece | str]) -> Board:
        """Build a board from ``{square: piece}``.

        Keys may be square indices or names (``'e0'``); values may be
        :class:`Piece` objects or FEN symbols (``'K'``, ``'r'``, ...). Handy
        for writing test positions::

            Board.from_pieces({"e0": "K", "e9": "k", "a0": "R"})
        """
        cells: list[Piece | None] = [None] * NUM_SQUARES
        for key, value in placement.items():
            square = parse_square(key) if isinstance(key, str) else validate_square(key)
            piece = Piece.from_symbol(value) if isinstance(value, str) else value
            if not isinstance(piece, Piece):
                raise TypeError(f"Expected Piece or symbol for {key!r}, got {value!r}")
            cells[square] = piece
        return cls._from_trusted(tuple(cells))

    @classmethod
    def from_fen_placement(cls, placement: str) -> Board:
        """Parse the piece-placement field of a Xiangqi FEN string.

        Ranks are listed from row 0 (Black's back rank) to row 9, separated by
        ``/``; digits denote runs of empty squares.
        """
        ranks = placement.split("/")
        if len(ranks) != NUM_ROWS:
            raise InvalidPositionError(f"FEN placement needs {NUM_ROWS} ranks, got {len(ranks)}")
        cells: list[Piece | None] = []
        for row, rank in enumerate(ranks):
            row_cells: list[Piece | None] = []
            for char in rank:
                if char.isdigit():
                    row_cells.extend([None] * int(char))
                else:
                    try:
                        row_cells.append(Piece.from_symbol(char))
                    except ValueError as exc:
                        raise InvalidPositionError(f"Row {row}: {exc}") from exc
            if len(row_cells) != NUM_COLS:
                raise InvalidPositionError(
                    f"FEN row {row} ({rank!r}) describes {len(row_cells)} squares, expected {NUM_COLS}"
                )
            cells.extend(row_cells)
        return cls._from_trusted(tuple(cells))

    # ------------------------------------------------------------------ queries

    @property
    def squares(self) -> tuple[Piece | None, ...]:
        """All 90 squares as an immutable tuple."""
        return self._squares

    def piece_at(self, square: int) -> Piece | None:
        """The piece on ``square``, or ``None`` if it is empty."""
        return self._squares[validate_square(square)]

    def is_empty(self, square: int) -> bool:
        """Whether ``square`` is empty."""
        return self.piece_at(square) is None

    def occupied(self, side: Side | None = None) -> Iterator[tuple[int, Piece]]:
        """Yield ``(square, piece)`` in increasing square order, optionally for one side."""
        for square, piece in enumerate(self._squares):
            if piece is not None and (side is None or piece.side is side):
                yield square, piece

    def count(self, side: Side | None = None, kind: PieceType | None = None) -> int:
        """Number of pieces, optionally filtered by side and/or type."""
        return sum(1 for _, piece in self.occupied(side) if kind is None or piece.kind is kind)

    def find_general(self, side: Side) -> int | None:
        """Square of ``side``'s general, or ``None`` if absent.

        Generals live in their palace in any valid position, so the nine
        palace squares are searched first; the rest of the board is scanned
        only as a fallback for unvalidated boards. This lookup sits on the hot
        path of every legality check.
        """
        cells = self._squares
        for square in (*_PALACE_SQUARES[side], *range(NUM_SQUARES)):
            piece = cells[square]
            if piece is not None and piece.kind is PieceType.GENERAL and piece.side is side:
                return square
        return None

    # ------------------------------------------------------------------ updates

    def apply_move(self, move: Move) -> Board:
        """Return a new board with ``move`` played (capturing whatever is on the
        destination).

        This is purely mechanical: it does **not** check Xiangqi movement rules
        or king safety. It only refuses moves from an empty square and captures
        of one's own piece, which can never be correct.
        """
        moving = self._squares[move.from_sq]
        if moving is None:
            raise IllegalMoveError(move, f"no piece on {square_name(move.from_sq)}")
        target = self._squares[move.to_sq]
        if target is not None and target.side is moving.side:
            raise IllegalMoveError(move, "cannot capture a piece of one's own side")
        cells = list(self._squares)
        cells[move.to_sq] = moving
        cells[move.from_sq] = None
        return Board._from_trusted(tuple(cells))

    # ------------------------------------------------------------------ serialization

    def to_fen_placement(self) -> str:
        """Piece-placement field of a Xiangqi FEN string (deterministic)."""
        ranks = []
        for row in range(NUM_ROWS):
            parts: list[str] = []
            empty_run = 0
            for col in range(NUM_COLS):
                piece = self._squares[row * NUM_COLS + col]
                if piece is None:
                    empty_run += 1
                    continue
                if empty_run:
                    parts.append(str(empty_run))
                    empty_run = 0
                parts.append(piece.symbol)
            if empty_run:
                parts.append(str(empty_run))
            ranks.append("".join(parts))
        return "/".join(ranks)

    # ------------------------------------------------------------------ validation

    def validation_errors(self) -> list[str]:
        """Describe every way this placement could not arise in a legal game.

        Checks: exactly one general per side inside its palace; no side has
        more pieces of a type than it starts with; advisors, elephants and
        soldiers stand only on squares they can reach. Facing generals and
        "side not to move is in check" depend on whose turn it is and are
        checked by :class:`~xiangqi_rl.engine.game.Game`.
        """
        errors: list[str] = []
        for side in Side:
            for kind, limit in STARTING_PIECE_COUNTS.items():
                count = self.count(side, kind)
                if kind is PieceType.GENERAL and count != 1:
                    errors.append(f"{side.name} must have exactly one general, has {count}")
                elif count > limit:
                    errors.append(f"{side.name} has {count} {kind.name.lower()}s (max {limit})")
        for square, piece in self.occupied():
            row, col = square_coords(square)
            where = f"{piece.side.name} {piece.kind.name.lower()} on {square_name(square)}"
            if piece.kind is PieceType.GENERAL and not in_palace(piece.side, row, col):
                errors.append(f"{where} is outside its palace")
            elif piece.kind is PieceType.ADVISOR and (row, col) not in _ADVISOR_SQUARES[piece.side]:
                errors.append(f"{where} is on a square advisors cannot reach")
            elif piece.kind is PieceType.ELEPHANT and (row, col) not in _ELEPHANT_SQUARES[piece.side]:
                errors.append(f"{where} is on a square elephants cannot reach")
            elif piece.kind is PieceType.SOLDIER:
                start_row = _SOLDIER_START_ROW[piece.side]
                behind_start = (row - start_row) * piece.side.forward < 0
                if behind_start:
                    errors.append(f"{where} is behind its starting rank")
                elif not has_crossed_river(piece.side, row) and col % 2 == 1:
                    # Before crossing, soldiers can only move straight ahead
                    # from their (even) starting files.
                    errors.append(f"{where} cannot have left its file before crossing the river")
        return errors

    # ------------------------------------------------------------------ dunder

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board):
            return NotImplemented
        return self._squares == other._squares

    def __hash__(self) -> int:
        return hash(self._squares)

    def __repr__(self) -> str:
        return f"Board.from_fen_placement({self.to_fen_placement()!r})"

    def __str__(self) -> str:
        """ASCII diagram with ranks/files labelled in UCCI-style notation."""
        lines = []
        for row in range(NUM_ROWS):
            cells = (
                piece.symbol if (piece := self._squares[row * NUM_COLS + col]) else "."
                for col in range(NUM_COLS)
            )
            lines.append(f"{NUM_ROWS - 1 - row}  " + " ".join(cells))
            if row == 4:
                lines.append("   ~~~~~ river ~~~~~")
        lines.append("   a b c d e f g h i")
        return "\n".join(lines)
