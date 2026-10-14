"""Regression tests for subtle cases plus invariants checked over random games.

The regression cases cover mistakes that are easy to make in a Xiangqi engine
(several are classic bugs in hobby engines), and one guards a performance
change made during implementation (see ``test_find_general_outside_palace``).
"""

from __future__ import annotations

from engine_helpers import board_from, game_from, legal_from, random_playout_states, ucis
from xiangqi_rl.engine import Board, Game, GameConfig, Piece, PieceType, Side, parse_square
from xiangqi_rl.engine.rules import is_in_check


# ----------------------------------------------------------------------------- regression cases


def test_find_general_outside_palace() -> None:
    # find_general() searches the palace first for speed; it must still find
    # a general elsewhere on an (unvalidated) board.
    board = board_from({"e5": "K", "a0": "k"})
    assert board.find_general(Side.RED) == parse_square("e5")
    assert board.find_general(Side.BLACK) == parse_square("a0")
    assert Board.empty().find_general(Side.RED) is None


def test_general_cannot_retreat_along_chariot_line() -> None:
    # Classic bug: evaluating attacks on the pre-move board, where the general
    # itself blocks the ray to the square behind it.
    game = game_from({"e1": "K", "e6": "r", "f9": "k"})
    assert "e1e0" not in ucis(game.legal_moves())
    assert "e1e2" not in ucis(game.legal_moves())
    assert legal_from(game, "e1") == {"d1"}  # f1 would face the black general


def test_capturing_cannon_screen_on_same_square_keeps_check() -> None:
    # The black soldier on e2 screens the e7 cannon. Capturing it with the
    # chariot puts a new screen on e2, so the check persists.
    game = game_from({"e0": "K", "a2": "R", "e2": "p", "e7": "c", "d9": "k"})
    assert game.is_in_check()
    assert "a2e2" not in ucis(game.legal_moves())
    assert ucis(game.legal_moves()) == {"e0f0"}


def test_discovered_check_is_recorded() -> None:
    game = game_from({"d0": "K", "e3": "R", "e6": "N", "e9": "k"})
    record = game.make_move("e6c7")
    assert record.gave_check and game.is_in_check(Side.BLACK)


def test_elephant_cannot_capture_across_river() -> None:
    game = game_from({"e0": "K", "d9": "k", "c4": "B", "a6": "r", "e6": "p"})
    assert legal_from(game, "c4") == {"a2", "e2"}


def test_soldier_on_own_river_bank_has_no_sideways_capture() -> None:
    game = game_from({"e0": "K", "d9": "k", "e4": "P", "d4": "p", "f4": "p"})
    assert legal_from(game, "e4") == {"e5"}


def test_illegal_move_after_terminal_does_not_change_result() -> None:
    game = game_from({"e0": "K", "i5": "R", "d9": "k"})
    game.make_move("i5i8")
    result, key = game.get_result(), game.position_key()
    assert not game.is_legal("d9d8")
    assert game.get_result() is result and game.position_key() == key


# ----------------------------------------------------------------------------- invariants


def _assert_board_invariants(board: Board) -> None:
    assert len(board.squares) == 90
    for cell in board.squares:
        assert cell is None or (isinstance(cell, Piece) and cell.side in (Side.RED, Side.BLACK))
    for side in Side:
        assert board.count(side, PieceType.GENERAL) == 1
        assert board.count(side) <= 16
    # Legal play can only ever reach placements that pass validation.
    assert board.validation_errors() == []


def test_invariants_hold_throughout_random_games() -> None:
    for game, _ in random_playout_states(seed=11, games=5, max_plies=150):
        board, side = game.board, game.side_to_move
        _assert_board_invariants(board)
        assert len(game.position_keys) == game.ply + 1
        captures = sum(record.is_capture for record in game.history)
        assert board.count() == 32 - captures
        # The side that just moved never leaves its own general in check.
        assert not is_in_check(board, side.opponent)
        if game.is_terminal():
            continue
        for move in game.legal_moves():
            after = board.apply_move(move)
            # Only the source and destination squares change.
            changed = {sq for sq in range(90) if after.squares[sq] != board.squares[sq]}
            assert changed == {move.from_sq, move.to_sq}
            assert after.squares[move.to_sq] == board.squares[move.from_sq]
            assert not is_in_check(after, side)


def test_every_legal_move_applies_through_game_api() -> None:
    for game, _ in random_playout_states(seed=5, games=2, max_plies=60):
        if game.is_terminal() or game.ply % 10:
            continue
        for move in game.legal_moves():
            trial = game.copy()
            record = trial.make_move(move)
            assert record.move == move and trial.ply == game.ply + 1
        assert game.ply == len(game.history)  # trials did not touch the original


def test_random_games_are_reproducible() -> None:
    def keys(seed: int) -> list[tuple[str, ...]]:
        out: list[tuple[str, ...]] = []
        last_index = -1
        for game, index in random_playout_states(seed=seed, games=2, max_plies=80):
            if index != last_index:
                out.append(())
                last_index = index
            out[-1] = game.position_keys
        return out

    assert keys(99) == keys(99)


def test_long_random_game_terminates_under_ply_limit() -> None:
    import random

    rng = random.Random(3)
    game = Game.new_game(GameConfig(max_plies=200))
    while not game.is_terminal():
        game.make_move(rng.choice(game.legal_moves()))
    assert game.ply <= 200
    assert game.termination_reason is not None
