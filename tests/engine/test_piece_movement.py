"""Pseudo-legal movement of each piece type, tested in isolation.

These tests use :func:`pseudo_legal_moves_from` on hand-built boards so each
piece's movement pattern is checked independently of king safety. Square
names: file a-i left to right, rank 0 (Red's back rank) to 9 (Black's).
"""

from __future__ import annotations

import pytest

from engine_helpers import board_from, targets_from


# ----------------------------------------------------------------------------- general


@pytest.mark.parametrize(
    ("pieces", "origin", "expected"),
    [
        ({"e1": "K"}, "e1", {"e0", "e2", "d1", "f1"}),  # palace centre
        ({"d0": "K"}, "d0", {"d1", "e0"}),  # palace corner
        ({"e2": "K"}, "e2", {"e1", "d2", "f2"}),  # cannot leave via e3
        ({"e8": "k"}, "e8", {"e9", "e7", "d8", "f8"}),  # black palace centre
        ({"f7": "k"}, "f7", {"f8", "e7"}),  # black corner: not g7 / f6
    ],
)
def test_general_moves_one_step_orthogonally_inside_palace(pieces, origin, expected) -> None:
    assert targets_from(board_from(pieces), origin) == expected


def test_general_cannot_capture_own_piece_but_can_capture_enemy() -> None:
    board = board_from({"e1": "K", "e0": "R", "d1": "r"})
    assert targets_from(board, "e1") == {"e2", "d1", "f1"}


# ----------------------------------------------------------------------------- advisor


@pytest.mark.parametrize(
    ("pieces", "origin", "expected"),
    [
        ({"e1": "A"}, "e1", {"d0", "f0", "d2", "f2"}),
        ({"d0": "A"}, "d0", {"e1"}),
        ({"f2": "A"}, "f2", {"e1"}),
        ({"e8": "a"}, "e8", {"d9", "f9", "d7", "f7"}),
        ({"d7": "a"}, "d7", {"e8"}),
    ],
)
def test_advisor_moves_one_step_diagonally_inside_palace(pieces, origin, expected) -> None:
    assert targets_from(board_from(pieces), origin) == expected


def test_advisor_blocked_by_own_piece() -> None:
    board = board_from({"e1": "A", "d0": "A", "f2": "n"})
    assert targets_from(board, "e1") == {"f0", "d2", "f2"}


# ----------------------------------------------------------------------------- elephant


@pytest.mark.parametrize(
    ("pieces", "origin", "expected"),
    [
        ({"e2": "B"}, "e2", {"c0", "g0", "c4", "g4"}),
        ({"a2": "B"}, "a2", {"c0", "c4"}),
        ({"e7": "b"}, "e7", {"c9", "g9", "c5", "g5"}),
    ],
)
def test_elephant_moves_two_steps_diagonally(pieces, origin, expected) -> None:
    assert targets_from(board_from(pieces), origin) == expected


@pytest.mark.parametrize(
    ("pieces", "origin", "expected"),
    [
        ({"c4": "B"}, "c4", {"a2", "e2"}),  # red elephant on the river bank
        ({"g4": "B"}, "g4", {"e2", "i2"}),
        ({"c5": "b"}, "c5", {"a7", "e7"}),  # black elephant on the river bank
    ],
)
def test_elephant_cannot_cross_river(pieces, origin, expected) -> None:
    assert targets_from(board_from(pieces), origin) == expected


def test_elephant_eye_blocks_regardless_of_colour() -> None:
    # Eye of e2->c0 is d1, eye of e2->g4 is f3.
    board = board_from({"e2": "B", "d1": "n", "f3": "P"})
    assert targets_from(board, "e2") == {"g0", "c4"}


def test_elephant_captures_enemy_not_own() -> None:
    board = board_from({"e2": "B", "c0": "r", "g0": "R"})
    assert targets_from(board, "e2") == {"c0", "c4", "g4"}


# ----------------------------------------------------------------------------- horse

ALL_HORSE_TARGETS_FROM_E4 = {"d6", "f6", "d2", "f2", "c5", "g5", "c3", "g3"}


def test_horse_has_eight_targets_on_open_board() -> None:
    assert targets_from(board_from({"e4": "N"}), "e4") == ALL_HORSE_TARGETS_FROM_E4


@pytest.mark.parametrize(
    ("leg", "blocked"),
    [
        ("e5", {"d6", "f6"}),  # leg toward Black
        ("e3", {"d2", "f2"}),  # leg toward Red
        ("d4", {"c5", "c3"}),  # leg to the left
        ("f4", {"g5", "g3"}),  # leg to the right
    ],
)
@pytest.mark.parametrize("blocker", ["P", "p"])
def test_horse_leg_blocks_exactly_two_targets(leg, blocked, blocker) -> None:
    board = board_from({"e4": "N", leg: blocker})
    assert targets_from(board, "e4") == ALL_HORSE_TARGETS_FROM_E4 - blocked


def test_diagonal_neighbours_do_not_hobble_horse() -> None:
    # Only orthogonal neighbours are legs; diagonal ones are irrelevant.
    board = board_from({"e4": "N", "d5": "P", "f5": "p", "d3": "P", "f3": "p"})
    assert targets_from(board, "e4") == ALL_HORSE_TARGETS_FROM_E4


def test_horse_in_corner_and_captures() -> None:
    assert targets_from(board_from({"a0": "N"}), "a0") == {"b2", "c1"}
    board = board_from({"a0": "N", "b2": "R", "c1": "r"})
    assert targets_from(board, "a0") == {"c1"}


# ----------------------------------------------------------------------------- chariot


def test_chariot_slides_until_blocked_and_captures_first_enemy() -> None:
    board = board_from({"e4": "R", "e6": "P", "b4": "n", "d0": "K", "f9": "k"})
    assert targets_from(board, "e4") == {
        "e5",  # up: stops before own soldier on e6
        "d4", "c4", "b4",  # left: captures b4, cannot reach a4
        "f4", "g4", "h4", "i4",  # right: to the edge
        "e3", "e2", "e1", "e0",  # down: to the edge
    }  # fmt: skip


def test_chariot_cannot_jump() -> None:
    board = board_from({"a0": "R", "a1": "r", "a2": "r"})
    assert "a2" not in targets_from(board, "a0")
    assert "a1" in targets_from(board, "a0")


# ----------------------------------------------------------------------------- cannon


def test_cannon_quiet_moves_screens_and_captures() -> None:
    board = board_from(
        {
            "e4": "C",
            "e7": "r", "e9": "n",  # up: enemy screen, enemy target
            "c4": "P", "a4": "r",  # left: own screen, enemy target
            "g4": "n", "h4": "P", "i4": "r",  # right: second piece is own -> no capture
            "e2": "P",  # down: screen with nothing behind it
        }
    )  # fmt: skip
    assert targets_from(board, "e4") == {"e5", "e6", "e9", "d4", "a4", "f4", "e3"}


def test_cannon_cannot_capture_without_screen() -> None:
    board = board_from({"e4": "C", "e7": "r"})
    assert targets_from(board, "e4") == {
        "e5", "e6", "e3", "e2", "e1", "e0",
        "d4", "c4", "b4", "a4", "f4", "g4", "h4", "i4",
    }  # fmt: skip


def test_cannon_cannot_jump_two_screens() -> None:
    board = board_from({"a0": "C", "a1": "P", "a2": "p", "a3": "r"})
    targets = targets_from(board, "a0")
    assert "a2" in targets  # exactly one screen (a1)
    assert "a3" not in targets  # two screens
    assert "a1" not in targets  # adjacent piece cannot be captured directly


def test_cannon_cannot_capture_own_piece_over_screen() -> None:
    board = board_from({"a0": "C", "a1": "p", "a2": "R"})
    assert targets_from(board, "a0") == {"b0", "c0", "d0", "e0", "f0", "g0", "h0", "i0"}


# ----------------------------------------------------------------------------- soldier


@pytest.mark.parametrize(
    ("pieces", "origin", "expected"),
    [
        ({"e3": "P"}, "e3", {"e4"}),  # home side: forward only
        ({"e4": "P"}, "e4", {"e5"}),  # on Red's river bank: still home side
        ({"e5": "P"}, "e5", {"e6", "d5", "f5"}),  # crossed: sideways too
        ({"a5": "P"}, "a5", {"a6", "b5"}),
        ({"e9": "P"}, "e9", {"d9", "f9"}),  # last rank: no forward, no promotion
        ({"e6": "p"}, "e6", {"e5"}),
        ({"e5": "p"}, "e5", {"e4"}),  # on Black's river bank
        ({"e4": "p"}, "e4", {"e3", "d4", "f4"}),
        ({"a0": "p"}, "a0", {"b0"}),
    ],
)
def test_soldier_movement(pieces, origin, expected) -> None:
    assert targets_from(board_from(pieces), origin) == expected


def test_soldier_never_moves_backward() -> None:
    for symbol, origin, backward in [("P", "e6", "e5"), ("p", "e3", "e4")]:
        assert backward not in targets_from(board_from({origin: symbol}), origin)


def test_soldier_blocked_by_own_piece_and_captures_enemy() -> None:
    board = board_from({"e5": "P", "e6": "R", "d5": "n"})
    assert targets_from(board, "e5") == {"d5", "f5"}


def test_empty_square_has_no_moves() -> None:
    assert targets_from(board_from({"e0": "K"}), "e5") == set()
