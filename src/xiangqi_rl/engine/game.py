"""Game: turn order, history, and terminal-state policy on top of the rules.

Terminal-state policy (evaluated after every move, in this order)
----------------------------------------------------------------
1. **No legal move**: the side to move loses. If it is in check this is
   ``CHECKMATE``, otherwise ``STALEMATE``; in Xiangqi stalemate is a loss,
   not a draw.
2. **Repetition**: delegated to :func:`repetition.adjudicate_repetition`
   (perpetual check loses, other repetitions draw), if enabled.
3. **Ply limit**: once ``config.max_plies`` plies have been played the game is
   a ``DRAW`` (``MAX_PLIES``), if enabled.

A checkmate on the last allowed ply therefore still counts as a win.
"""

from __future__ import annotations

from dataclasses import dataclass

from xiangqi_rl.engine.board import Board
from xiangqi_rl.engine.constants import GameResult, Side, TerminationReason
from xiangqi_rl.engine.errors import (
    GameOverError,
    IllegalMoveError,
    InvalidPositionError,
    XiangqiError,
)
from xiangqi_rl.engine.move import Move, MoveRecord
from xiangqi_rl.engine.repetition import adjudicate_repetition
from xiangqi_rl.engine.rules import (
    generate_legal_moves,
    illegal_move_reason,
    is_in_check,
)

_FEN_SIDE = {Side.RED: "w", Side.BLACK: "b"}
_SIDE_FROM_FEN = {"w": Side.RED, "r": Side.RED, "b": Side.BLACK}


def position_key(board: Board, side_to_move: Side) -> str:
    """Deterministic, collision-free key of a position.

    The key is ``"<FEN placement> <w|b>"``: a plain serialisation, not a
    hash, so two positions share a key if and only if they have the same
    placement and side to move. Used for repetition detection.
    """
    return f"{board.to_fen_placement()} {_FEN_SIDE[side_to_move]}"


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Configuration of the terminal-state policy.

    Attributes:
        max_plies: Draw once this many plies have been played (``None`` disables).
        repetition_threshold: Adjudicate when a position occurs this many times
            (``None`` disables repetition adjudication).
        perpetual_check_loses: During repetition adjudication, a side that
            checked on every move in the cycle loses (otherwise always draw).
    """

    max_plies: int | None = 300
    repetition_threshold: int | None = 3
    perpetual_check_loses: bool = True

    def __post_init__(self) -> None:
        if self.max_plies is not None and self.max_plies < 1:
            raise ValueError(f"max_plies must be positive or None, got {self.max_plies}")
        if self.repetition_threshold is not None and self.repetition_threshold < 2:
            raise ValueError(
                f"repetition_threshold must be at least 2 or None, got {self.repetition_threshold}"
            )


@dataclass(frozen=True, slots=True)
class GameState:
    """Immutable snapshot of a game's current position and status."""

    board: Board
    side_to_move: Side
    ply: int
    result: GameResult
    termination: TerminationReason | None

    @property
    def position_key(self) -> str:
        """See :func:`position_key`."""
        return position_key(self.board, self.side_to_move)

    @property
    def is_terminal(self) -> bool:
        """Whether the game has ended."""
        return self.result is not GameResult.IN_PROGRESS


class Game:
    """A Xiangqi game: position, side to move, history and result.

    Typical use::

        game = Game.new_game()
        moves = game.legal_moves()
        game.make_move(moves[0])          # or game.make_move("h2e2")
        game.is_terminal(), game.get_result()

    The board is immutable and all returned collections are tuples or frozen
    dataclasses, so inspecting a game cannot mutate it. A move is validated
    completely before any state changes; an illegal move leaves the game
    untouched.
    """

    def __init__(
        self,
        board: Board | None = None,
        side_to_move: Side = Side.RED,
        config: GameConfig | None = None,
    ) -> None:
        """Start a game from ``board`` (default: standard start) with
        ``side_to_move`` to play.

        Raises:
            InvalidPositionError: If the position could not occur in a legal
                game (see :meth:`Board.validation_errors`), or the side *not*
                to move is in check (its general could be captured).
        """
        board = Board.initial() if board is None else board
        if not isinstance(board, Board):
            raise TypeError(f"board must be a Board, got {board!r}")
        if not isinstance(side_to_move, Side):
            raise TypeError(f"side_to_move must be a Side, got {side_to_move!r}")
        problems = board.validation_errors()
        if not problems and is_in_check(board, side_to_move.opponent):
            problems.append(
                f"{side_to_move.opponent.name} is in check but it is {side_to_move.name}'s turn"
            )
        if problems:
            raise InvalidPositionError("Invalid position: " + "; ".join(problems))

        self._config = config if config is not None else GameConfig()
        self._initial_board = board
        self._initial_side = side_to_move
        self._start()

    # ------------------------------------------------------------------ construction

    @classmethod
    def new_game(cls, config: GameConfig | None = None) -> Game:
        """A game from the standard starting position, Red to move."""
        return cls(Board.initial(), Side.RED, config)

    @classmethod
    def from_fen(cls, fen: str, config: GameConfig | None = None) -> Game:
        """A game from a Xiangqi FEN string.

        Only the placement and side-to-move fields are used (``w``/``r`` for
        Red, ``b`` for Black); move counters, if present, are ignored and the
        new game's history starts empty.
        """
        fields = fen.split()
        if len(fields) < 2 or fields[1] not in _SIDE_FROM_FEN:
            raise InvalidPositionError(f"FEN needs a placement and a side to move: {fen!r}")
        return cls(Board.from_fen_placement(fields[0]), _SIDE_FROM_FEN[fields[1]], config)

    def _start(self) -> None:
        """(Re)initialise all mutable state from the initial position."""
        self._board = self._initial_board
        self._side_to_move = self._initial_side
        self._boards: list[Board] = [self._initial_board]
        self._records: list[MoveRecord] = []
        self._position_keys: list[str] = [position_key(self._board, self._side_to_move)]
        self._evaluate_status()

    def reset(self) -> None:
        """Restore the game's initial position and clear history."""
        self._start()

    def copy(self) -> Game:
        """An independent copy (sharing only immutable boards and records)."""
        clone = Game.__new__(Game)
        clone._config = self._config
        clone._initial_board = self._initial_board
        clone._initial_side = self._initial_side
        clone._board = self._board
        clone._side_to_move = self._side_to_move
        clone._boards = list(self._boards)
        clone._records = list(self._records)
        clone._position_keys = list(self._position_keys)
        clone._legal_moves = self._legal_moves
        clone._legal_set = self._legal_set
        clone._result = self._result
        clone._termination = self._termination
        return clone

    clone = copy

    def __copy__(self) -> Game:
        return self.copy()

    def __deepcopy__(self, memo: dict) -> Game:
        return self.copy()

    # ------------------------------------------------------------------ inspection

    @property
    def config(self) -> GameConfig:
        """The game's (immutable) configuration."""
        return self._config

    @property
    def board(self) -> Board:
        """The current (immutable) board."""
        return self._board

    @property
    def side_to_move(self) -> Side:
        """The side whose turn it is."""
        return self._side_to_move

    @property
    def ply(self) -> int:
        """Number of moves played so far in this game."""
        return len(self._records)

    @property
    def state(self) -> GameState:
        """Immutable snapshot of the current position and status."""
        return GameState(
            board=self._board,
            side_to_move=self._side_to_move,
            ply=self.ply,
            result=self._result,
            termination=self._termination,
        )

    @property
    def history(self) -> tuple[MoveRecord, ...]:
        """All moves played, oldest first."""
        return tuple(self._records)

    @property
    def position_keys(self) -> tuple[str, ...]:
        """Position key after each ply; element 0 is the starting position."""
        return tuple(self._position_keys)

    def position_key(self) -> str:
        """Key of the current position (see :func:`position_key`)."""
        return self._position_keys[-1]

    def to_fen(self) -> str:
        """Xiangqi FEN of the current position.

        The halfmove field counts plies since the last capture in this game and
        the fullmove field starts at 1 and increments after each Black move.
        """
        since_capture = 0
        for record in reversed(self._records):
            if record.is_capture:
                break
            since_capture += 1
        black_moves = sum(1 for record in self._records if record.side is Side.BLACK)
        return (
            f"{self._board.to_fen_placement()} {_FEN_SIDE[self._side_to_move]} - - "
            f"{since_capture} {1 + black_moves}"
        )

    def legal_moves(self) -> tuple[Move, ...]:
        """Legal moves for the side to move, sorted; empty once the game is over.

        These are exactly the moves :meth:`make_move` accepts.
        """
        return self._legal_moves if not self.is_terminal() else ()

    def is_legal(self, move: Move | str) -> bool:
        """Whether :meth:`make_move` would accept ``move`` right now."""
        return not self.is_terminal() and _coerce_move(move) in self._legal_set

    def is_in_check(self, side: Side | None = None) -> bool:
        """Whether ``side`` (default: side to move) is in check."""
        return is_in_check(self._board, self._side_to_move if side is None else side)

    def is_checkmate(self, side: Side | None = None) -> bool:
        """Whether ``side`` (default: side to move) is in check with no legal move."""
        side = self._side_to_move if side is None else side
        return self.is_in_check(side) and not self._legal_moves_for(side)

    def is_stalemate(self, side: Side | None = None) -> bool:
        """Whether ``side`` is *not* in check but has no legal move (a loss)."""
        side = self._side_to_move if side is None else side
        return not self.is_in_check(side) and not self._legal_moves_for(side)

    def is_terminal(self) -> bool:
        """Whether the game has ended."""
        return self._result is not GameResult.IN_PROGRESS

    def get_result(self) -> GameResult:
        """The current result (``IN_PROGRESS`` while the game continues)."""
        return self._result

    @property
    def result(self) -> GameResult:
        """Alias of :meth:`get_result`."""
        return self._result

    @property
    def termination_reason(self) -> TerminationReason | None:
        """Why the game ended, or ``None`` while it is in progress."""
        return self._termination

    @property
    def winner(self) -> Side | None:
        """The winning side, or ``None`` for a draw or an unfinished game."""
        if self._result is GameResult.RED_WINS:
            return Side.RED
        if self._result is GameResult.BLACK_WINS:
            return Side.BLACK
        return None

    # ------------------------------------------------------------------ mutation

    def make_move(self, move: Move | str) -> MoveRecord:
        """Play ``move`` (a :class:`Move` or notation such as ``'h2e2'``).

        Returns:
            The :class:`MoveRecord` appended to the history.

        Raises:
            GameOverError: If the game has already ended.
            IllegalMoveError: If the move is not legal; the game is unchanged.
        """
        if self.is_terminal():
            raise GameOverError(f"The game is over ({self._result.value}); no more moves allowed")
        move = _coerce_move(move)
        if move not in self._legal_set:
            reason = illegal_move_reason(self._board, self._side_to_move, move)
            raise IllegalMoveError(move, reason or "not a legal move in this position")

        # Compute everything first, then commit, so failures cannot leave a
        # partially applied move.
        piece = self._board.squares[move.from_sq]
        assert piece is not None  # guaranteed by legality
        captured = self._board.squares[move.to_sq]
        new_board = self._board.apply_move(move)
        new_side = self._side_to_move.opponent
        record = MoveRecord(
            ply=len(self._records),
            move=move,
            piece=piece,
            captured=captured,
            gave_check=is_in_check(new_board, new_side),
        )
        new_key = position_key(new_board, new_side)

        self._board = new_board
        self._side_to_move = new_side
        self._boards.append(new_board)
        self._records.append(record)
        self._position_keys.append(new_key)
        self._evaluate_status()
        return record

    def undo_move(self) -> MoveRecord:
        """Take back the last move and return its record.

        Raises:
            XiangqiError: If no move has been played.
        """
        if not self._records:
            raise XiangqiError("No move to undo")
        record = self._records.pop()
        self._boards.pop()
        self._position_keys.pop()
        self._board = self._boards[-1]
        self._side_to_move = record.side
        self._evaluate_status()
        return record

    # ------------------------------------------------------------------ internals

    def _legal_moves_for(self, side: Side) -> tuple[Move, ...]:
        if side is self._side_to_move:
            return self._legal_moves
        return tuple(generate_legal_moves(self._board, side))

    def _evaluate_status(self) -> None:
        """Recompute legal moves and the result for the current position."""
        self._legal_moves: tuple[Move, ...] = tuple(
            generate_legal_moves(self._board, self._side_to_move)
        )
        self._legal_set: frozenset[Move] = frozenset(self._legal_moves)
        self._result = GameResult.IN_PROGRESS
        self._termination: TerminationReason | None = None

        if not self._legal_moves:
            self._result = GameResult.win_for(self._side_to_move.opponent)
            self._termination = (
                TerminationReason.CHECKMATE
                if is_in_check(self._board, self._side_to_move)
                else TerminationReason.STALEMATE
            )
            return

        threshold = self._config.repetition_threshold
        if threshold is not None:
            verdict = adjudicate_repetition(
                self._position_keys,
                [record.side for record in self._records],
                [record.gave_check for record in self._records],
                threshold=threshold,
                perpetual_check_loses=self._config.perpetual_check_loses,
            )
            if verdict is not None:
                self._result = verdict.result
                self._termination = verdict.reason
                return

        max_plies = self._config.max_plies
        if max_plies is not None and self.ply >= max_plies:
            self._result = GameResult.DRAW
            self._termination = TerminationReason.MAX_PLIES

    def __repr__(self) -> str:
        return f"Game(fen={self.to_fen()!r}, result={self._result.value})"

    def __str__(self) -> str:
        return f"{self._board}\n{self._side_to_move.name} to move, ply {self.ply}, {self._result.value}"


def _coerce_move(move: Move | str) -> Move:
    if isinstance(move, Move):
        return move
    if isinstance(move, str):
        return Move.from_uci(move)
    raise TypeError(f"Expected Move or move notation string, got {move!r}")
