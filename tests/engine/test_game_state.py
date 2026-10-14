"""Game API, integrity guarantees, ply limit, repetition, models and serialization."""

from __future__ import annotations

import copy
import dataclasses

import pytest

from engine_helpers import board_from, game_from
from xiangqi_rl.engine import (
    ACTION_SPACE_SIZE,
    INITIAL_FEN,
    Board,
    Game,
    GameConfig,
    GameOverError,
    GameResult,
    IllegalMoveError,
    InvalidMoveFormatError,
    InvalidPositionError,
    InvalidSquareError,
    Move,
    Piece,
    PieceType,
    Side,
    TerminationReason,
    XiangqiError,
    make_square,
    parse_square,
    square_name,
)
from xiangqi_rl.engine.repetition import adjudicate_repetition

KNIGHT_SHUFFLE = ["b0c2", "b9c7", "c2b0", "c7b9"]  # returns to the start position


def play(game: Game, moves: list[str]) -> None:
    for move in moves:
        game.make_move(move)


# ----------------------------------------------------------------------------- models


def test_square_conversions() -> None:
    assert make_square(0, 0) == 0 and make_square(9, 8) == 89
    assert square_name(make_square(9, 4)) == "e0"
    assert square_name(make_square(0, 4)) == "e9"
    assert all(parse_square(square_name(sq)) == sq for sq in range(90))


@pytest.mark.parametrize("bad", [(-1, 0), (10, 0), (0, 9), (0, -1)])
def test_invalid_coordinates_rejected(bad) -> None:
    with pytest.raises(InvalidSquareError):
        make_square(*bad)


@pytest.mark.parametrize("bad", [-1, 90, 1000, True, "5", 1.0])
def test_invalid_square_indices_rejected(bad) -> None:
    with pytest.raises(InvalidSquareError):
        Move(bad, 10)


def test_move_equality_hash_and_order() -> None:
    assert Move(1, 2) == Move(1, 2) == Move.from_uci(Move(1, 2).uci())
    assert Move(1, 2) != Move(2, 1)
    assert len({Move(1, 2), Move(1, 2), Move(2, 1)}) == 2
    assert sorted([Move(5, 1), Move(1, 9), Move(1, 3)]) == [Move(1, 3), Move(1, 9), Move(5, 1)]


def test_move_action_encoding_roundtrip() -> None:
    assert ACTION_SPACE_SIZE == 8100
    move = Move.from_uci("h2e2")
    assert move.to_action() == 90 * move.from_sq + move.to_sq
    for action in range(ACTION_SPACE_SIZE):
        from_sq, to_sq = divmod(action, 90)
        if from_sq == to_sq:
            with pytest.raises(InvalidMoveFormatError):
                Move.from_action(action)
        else:
            assert Move.from_action(action).to_action() == action
    for bad in (-1, 8100, True):
        with pytest.raises(InvalidMoveFormatError):
            Move.from_action(bad)


@pytest.mark.parametrize("bad", ["", "e0", "e0e0", "j0a1", "a0a10", "E0E1", None])
def test_invalid_move_notation_rejected(bad) -> None:
    with pytest.raises((InvalidMoveFormatError, InvalidSquareError)):
        Move.from_uci(bad)


def test_piece_symbols_roundtrip() -> None:
    for side in Side:
        for kind in PieceType:
            piece = Piece(side, kind)
            assert Piece.from_symbol(piece.symbol) == piece
    assert Piece.from_symbol("H") == Piece(Side.RED, PieceType.HORSE)
    assert Piece.from_symbol("e") == Piece(Side.BLACK, PieceType.ELEPHANT)
    with pytest.raises(ValueError):
        Piece.from_symbol("x")


# ----------------------------------------------------------------------------- invalid positions


@pytest.mark.parametrize(
    ("pieces", "side"),
    [
        ({"e0": "K"}, Side.RED),  # missing black general
        ({"e0": "K", "d0": "K", "e9": "k"}, Side.RED),  # two red generals
        ({"e3": "K", "d9": "k"}, Side.RED),  # general outside palace
        ({"e0": "K", "d9": "k", "d5": "R"}, Side.RED),  # black (not to move) in check
        ({"e0": "K", "d9": "k", "e5": "A"}, Side.RED),  # advisor outside palace
        ({"e0": "K", "d9": "k", "d1": "A"}, Side.RED),  # advisor off the diagonals
        ({"e0": "K", "d9": "k", "e4": "B"}, Side.RED),  # unreachable elephant square
        ({"e0": "K", "d9": "k", "c2": "P"}, Side.RED),  # soldier behind its start
        ({"e0": "K", "d9": "k", "b3": "P"}, Side.RED),  # soldier off-file before river
        ({"e0": "K", "d9": "k", **{f"{f}5": "P" for f in "abcdef"}}, Side.RED),  # 6 soldiers
    ],
)
def test_invalid_positions_rejected(pieces, side) -> None:
    with pytest.raises(InvalidPositionError):
        game_from(pieces, side)


def test_fen_parsing_errors() -> None:
    with pytest.raises(InvalidPositionError):
        Board.from_fen_placement("9/9/9")
    with pytest.raises(InvalidPositionError):
        Board.from_fen_placement("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABN")
    with pytest.raises(InvalidPositionError):
        Board.from_fen_placement("rnbakabnx/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR")
    with pytest.raises(InvalidPositionError):
        Game.from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR x")


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        GameConfig(max_plies=0)
    with pytest.raises(ValueError):
        GameConfig(repetition_threshold=1)


# ----------------------------------------------------------------------------- API & integrity


@pytest.mark.parametrize(
    ("move", "reason_fragment"),
    [
        ("e5e4", "no piece"),
        ("a9a8", "belongs to BLACK"),
        ("a0b0", "own side"),
        ("e0d1", "cannot move"),  # general moving diagonally
        ("b2b8", "cannot move"),  # cannon onto the black cannon with no screen
    ],
)
def test_illegal_moves_raise_and_do_not_mutate(move, reason_fragment) -> None:
    game = Game.new_game()
    play(game, ["h2e2"])
    game.make_move("h9g7")
    before = (game.state, game.history, game.position_keys, game.legal_moves())
    with pytest.raises(IllegalMoveError) as excinfo:
        game.make_move(move)
    assert reason_fragment in excinfo.value.reason
    assert excinfo.value.move == Move.from_uci(move)
    assert (game.state, game.history, game.position_keys, game.legal_moves()) == before


def test_self_check_move_rejected_with_reason() -> None:
    game = game_from({"e0": "K", "e2": "N", "e7": "r", "d9": "k"})
    with pytest.raises(IllegalMoveError, match="leave the general in check"):
        game.make_move("e2c3")
    assert game.board.piece_at(parse_square("e2")) == Piece(Side.RED, PieceType.HORSE)


def test_make_move_accepts_move_objects_and_rejects_other_types() -> None:
    game = Game.new_game()
    game.make_move(Move.from_uci("h2e2"))
    assert game.side_to_move is Side.BLACK
    with pytest.raises(TypeError):
        game.make_move(1234)  # type: ignore[arg-type]
    with pytest.raises(InvalidMoveFormatError):
        game.make_move("nonsense")


def test_is_legal_matches_legal_moves() -> None:
    game = Game.new_game()
    assert game.is_legal("h2e2") and game.is_legal(Move.from_uci("b0c2"))
    assert not game.is_legal("h2h8") and not game.is_legal("e0e2")


def test_capture_updates_board_and_history() -> None:
    game = Game.new_game()
    record = game.make_move("b2b9")  # cannon jumps the b7 cannon, takes the horse
    assert record.captured == Piece(Side.BLACK, PieceType.HORSE)
    assert record.piece == Piece(Side.RED, PieceType.CANNON)
    assert record.is_capture and record.side is Side.RED and record.ply == 0
    assert game.board.piece_at(parse_square("b9")) == Piece(Side.RED, PieceType.CANNON)
    assert game.board.is_empty(parse_square("b2"))
    assert game.board.count(Side.BLACK) == 15 and game.board.count(Side.RED) == 16
    assert game.history == (record,)
    assert game.to_fen().endswith(" b - - 0 1")


def test_history_records_alternating_sides() -> None:
    game = Game.new_game()
    play(game, ["h2e2", "h9g7", "h0g2"])
    assert [r.move.uci() for r in game.history] == ["h2e2", "h9g7", "h0g2"]
    assert [r.side for r in game.history] == [Side.RED, Side.BLACK, Side.RED]
    assert [r.ply for r in game.history] == [0, 1, 2]
    assert game.ply == 3 and game.side_to_move is Side.BLACK


def test_reset_restores_exact_initial_state() -> None:
    game = Game.new_game()
    play(game, ["b2b9", "a9b9", "h2e2"])
    game.reset()
    fresh = Game.new_game()
    assert game.state == fresh.state
    assert game.history == () and game.position_keys == fresh.position_keys
    assert game.legal_moves() == fresh.legal_moves()
    assert game.to_fen() == INITIAL_FEN


def test_reset_of_custom_position_returns_to_that_position() -> None:
    start = {"e0": "K", "i5": "R", "d9": "k"}
    game = game_from(start)
    game.make_move("i5i8")
    assert game.is_terminal()
    game.reset()
    assert game.board == board_from(start) and not game.is_terminal()


def test_clone_does_not_share_mutable_state() -> None:
    game = Game.new_game()
    game.make_move("h2e2")
    for clone in (game.copy(), game.clone(), copy.copy(game), copy.deepcopy(game)):
        clone.make_move("h9g7")
        assert game.ply == 1 and game.side_to_move is Side.BLACK
        assert clone.ply == 2
    original_key = game.position_key()
    clone = game.copy()
    game.make_move("b9c7")
    assert clone.position_key() == original_key and clone.ply == 1


def test_state_snapshot_is_immutable() -> None:
    game = Game.new_game()
    state = game.state
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.ply = 5  # type: ignore[misc]
    assert isinstance(game.board.squares, tuple)
    assert isinstance(game.history, tuple) and isinstance(game.legal_moves(), tuple)
    game.make_move("h2e2")
    assert state.ply == 0 and state.board == Board.initial()  # snapshot unaffected


def test_undo_move_restores_previous_positions() -> None:
    game = Game.new_game()
    states = [game.state]
    for move in ["b2b9", "a9b9", "h2e2"]:
        game.make_move(move)
        states.append(game.state)
    for expected in reversed(states[:-1]):
        game.undo_move()
        assert game.state == expected
    with pytest.raises(XiangqiError):
        game.undo_move()


def test_undo_after_checkmate_resumes_game() -> None:
    game = game_from({"d0": "K", "a8": "R", "i5": "R", "e9": "k"})
    game.make_move("i5i9")
    assert game.is_terminal()
    game.undo_move()
    assert game.get_result() is GameResult.IN_PROGRESS and game.side_to_move is Side.RED


# ----------------------------------------------------------------------------- serialization


def test_position_key_is_deterministic_and_transposition_aware() -> None:
    a, b = Game.new_game(), Game.new_game()
    play(a, ["b0c2", "b9c7", "h0g2", "h9g7"])
    play(b, ["h0g2", "h9g7", "b0c2", "b9c7"])
    assert a.position_key() == b.position_key()
    assert Game.new_game().position_key() == Game.new_game().position_key()
    assert Game.new_game().position_key().endswith(" w")


def test_position_key_includes_side_to_move() -> None:
    board = board_from({"e0": "K", "d9": "k", "a3": "P"})
    assert Game(board, Side.RED).position_key() != Game(board, Side.BLACK).position_key()


def test_fen_roundtrip() -> None:
    game = Game.new_game()
    play(game, ["h2e2", "h9g7", "e2e6"])
    restored = Game.from_fen(game.to_fen())
    assert restored.board == game.board and restored.side_to_move is game.side_to_move
    assert Board.from_fen_placement(game.board.to_fen_placement()) == game.board


# ----------------------------------------------------------------------------- ply limit


def test_max_ply_limit_declares_draw() -> None:
    game = Game.new_game(GameConfig(max_plies=4))
    play(game, ["b0c2", "b9c7", "h0g2"])
    assert not game.is_terminal()
    game.make_move("h9g7")
    assert game.get_result() is GameResult.DRAW
    assert game.termination_reason is TerminationReason.MAX_PLIES
    assert game.winner is None
    with pytest.raises(GameOverError):
        game.make_move("c2d4")


def test_ply_limit_can_be_disabled() -> None:
    game = Game.new_game(GameConfig(max_plies=None, repetition_threshold=None))
    play(game, KNIGHT_SHUFFLE * 50)
    assert game.ply == 200 and not game.is_terminal()


# ----------------------------------------------------------------------------- repetition


def test_threefold_repetition_without_checks_is_draw() -> None:
    game = Game.new_game()
    play(game, KNIGHT_SHUFFLE + KNIGHT_SHUFFLE[:3])
    assert not game.is_terminal()  # start position seen twice so far
    game.make_move(KNIGHT_SHUFFLE[3])  # third occurrence
    assert game.get_result() is GameResult.DRAW
    assert game.termination_reason is TerminationReason.REPETITION


def test_repetition_threshold_is_configurable() -> None:
    game = Game.new_game(GameConfig(repetition_threshold=2))
    play(game, KNIGHT_SHUFFLE)
    assert game.termination_reason is TerminationReason.REPETITION


def test_repetition_can_be_disabled() -> None:
    game = Game.new_game(GameConfig(repetition_threshold=None))
    play(game, KNIGHT_SHUFFLE * 3)
    assert not game.is_terminal()


PERPETUAL = {"d0": "K", "a8": "R", "e9": "k"}
# Red checks on every move (a9 on rank 9, a8 on rank 8); Black just shuffles.
PERPETUAL_CYCLE = ["a8a9", "e9e8", "a9a8", "e8e9"]


def test_perpetual_check_loses() -> None:
    game = game_from(PERPETUAL)
    play(game, PERPETUAL_CYCLE + PERPETUAL_CYCLE[:3])
    assert not game.is_terminal()
    assert all(r.gave_check for r in game.history if r.side is Side.RED)
    game.make_move(PERPETUAL_CYCLE[3])
    assert game.get_result() is GameResult.BLACK_WINS
    assert game.termination_reason is TerminationReason.PERPETUAL_CHECK


def test_perpetual_check_draw_when_rule_disabled() -> None:
    game = game_from(PERPETUAL, perpetual_check_loses=False)
    play(game, PERPETUAL_CYCLE * 2)
    assert game.get_result() is GameResult.DRAW
    assert game.termination_reason is TerminationReason.REPETITION


def test_adjudicator_mutual_perpetual_check_is_draw() -> None:
    keys = ["A", "B", "C", "D", "A", "B", "C", "D", "A"]
    movers = [Side.RED, Side.BLACK] * 4
    verdict = adjudicate_repetition(keys, movers, [True] * 8)
    assert verdict is not None and verdict.result is GameResult.DRAW


def test_adjudicator_black_perpetual_check_loses() -> None:
    keys = ["A", "B", "C", "D", "A", "B", "C", "D", "A"]
    movers = [Side.RED, Side.BLACK] * 4
    verdict = adjudicate_repetition(keys, movers, [False, True] * 4)
    assert verdict is not None
    assert verdict.result is GameResult.RED_WINS
    assert verdict.reason is TerminationReason.PERPETUAL_CHECK


def test_adjudicator_window_requires_check_on_every_move() -> None:
    # Red checked on all but one move inside the window: not perpetual.
    keys = ["A", "B", "C", "D", "A", "B", "C", "D", "A"]
    movers = [Side.RED, Side.BLACK] * 4
    checks = [True, False, True, False, False, False, True, False]
    verdict = adjudicate_repetition(keys, movers, checks)
    assert verdict is not None and verdict.result is GameResult.DRAW


def test_adjudicator_below_threshold_and_validation() -> None:
    assert adjudicate_repetition(["A", "B", "A"], [Side.RED, Side.BLACK], [True, True]) is None
    with pytest.raises(ValueError):
        adjudicate_repetition(["A"], [Side.RED], [True])
    with pytest.raises(ValueError):
        adjudicate_repetition(["A"], [], [], threshold=1)
