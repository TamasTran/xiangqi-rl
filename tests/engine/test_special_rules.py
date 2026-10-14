"""Xiangqi-specific constraints: flying general, cannon screens, palace, river."""

from __future__ import annotations

import pytest

from engine_helpers import board_from, game_from, legal_from, ucis
from xiangqi_rl.engine import InvalidPositionError, Move, Side
from xiangqi_rl.engine.rules import generals_facing, illegal_move_reason, is_in_check


# ----------------------------------------------------------------------------- flying general


def test_generals_facing_on_open_file_counts_as_check_for_both() -> None:
    board = board_from({"e0": "K", "e9": "k"})
    assert generals_facing(board)
    assert is_in_check(board, Side.RED) and is_in_check(board, Side.BLACK)


def test_generals_not_facing_with_piece_between_or_different_files() -> None:
    assert not generals_facing(board_from({"e0": "K", "e9": "k", "e5": "p"}))
    assert not generals_facing(board_from({"e0": "K", "d9": "k"}))


def test_game_rejects_position_with_facing_generals() -> None:
    for side in Side:
        with pytest.raises(InvalidPositionError):
            game_from({"e0": "K", "e9": "k"}, side)


def test_general_cannot_step_onto_open_file_of_enemy_general() -> None:
    game = game_from({"d0": "K", "e9": "k"})
    assert legal_from(game, "d0") == {"d1"}
    assert illegal_move_reason(game.board, Side.RED, Move.from_uci("d0e0")) == (
        "the two generals would face each other"
    )


def test_only_blocker_between_generals_cannot_leave_file() -> None:
    # The red horse is the only piece between the generals: every horse move
    # leaves the e-file and would expose the generals.
    game = game_from({"e0": "K", "e9": "k", "e4": "N"})
    assert legal_from(game, "e4") == set()


def test_blocker_between_generals_may_move_along_file() -> None:
    # A cannon with no screen does not attack the black general, so this is
    # a valid position; the cannon may slide along the file but not off it.
    game = game_from({"e0": "K", "e9": "k", "e4": "C"})
    assert legal_from(game, "e4") == {"e1", "e2", "e3", "e5", "e6", "e7", "e8"}


def test_capture_that_exposes_generals_is_illegal() -> None:
    game = game_from({"e0": "K", "e9": "k", "e4": "N", "g5": "p"})
    assert "e4g5" not in ucis(game.legal_moves())


def test_second_blocker_frees_the_first() -> None:
    game = game_from({"e0": "K", "e9": "k", "e4": "N", "e6": "p"})
    assert legal_from(game, "e4") == {"d6", "f6", "d2", "f2", "c5", "g5", "c3", "g3"}


# ----------------------------------------------------------------------------- cannon screens


def test_cannon_check_needs_exactly_one_screen() -> None:
    no_screen = board_from({"e0": "K", "d9": "k", "e7": "c"})
    one_screen = board_from({"e0": "K", "d9": "k", "e7": "c", "e3": "P"})
    two_screens = board_from({"e0": "K", "d9": "k", "e7": "c", "e3": "P", "e5": "p"})
    assert not is_in_check(no_screen, Side.RED)
    assert is_in_check(one_screen, Side.RED)
    assert not is_in_check(two_screens, Side.RED)


def test_moving_piece_may_not_become_a_cannon_screen() -> None:
    # Putting the chariot on the e-file would give the black cannon a screen.
    game = game_from({"e0": "K", "d9": "k", "e7": "c", "a4": "R"})
    assert "a4e4" not in ucis(game.legal_moves())
    assert illegal_move_reason(game.board, Side.RED, Move.from_uci("a4e4")) == (
        "it would leave the general in check"
    )


def test_general_retreating_along_cannon_line_stays_in_check() -> None:
    # Cannon e7 with screen e3 checks the general on e1. Retreating to e0 or
    # advancing to e2 stays on the cannon's line (still exactly one screen);
    # d1 would face the black general on d9. Only f1 escapes, and the soldier
    # screen can only move forward along the file, which does not help.
    game = game_from({"e1": "K", "d9": "k", "e7": "c", "e3": "P"}, Side.RED)
    assert game.is_in_check()
    assert ucis(game.legal_moves()) == {"e1f1"}


# ----------------------------------------------------------------------------- palace / river at legal level


def test_general_and_advisor_never_leave_palace_in_legal_moves() -> None:
    game = game_from({"d2": "K", "f2": "A", "f9": "k"})
    assert legal_from(game, "d2") == {"d1", "e2"}
    assert legal_from(game, "f2") == {"e1"}


def test_elephant_river_restriction_in_legal_moves() -> None:
    game = game_from({"e0": "K", "d9": "k", "c4": "B"})
    assert legal_from(game, "c4") == {"a2", "e2"}
