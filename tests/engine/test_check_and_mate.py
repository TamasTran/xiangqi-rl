"""Attack detection, check, checkmate and no-legal-move (stalemate) losses."""

from __future__ import annotations

import pytest

from engine_helpers import board_from, game_from, random_playout_states, ucis
from xiangqi_rl.engine import (
    Board,
    Game,
    GameOverError,
    GameResult,
    Side,
    TerminationReason,
    parse_square,
)
from xiangqi_rl.engine.move_generator import generate_pseudo_legal_moves
from xiangqi_rl.engine.rules import attackers_of, is_in_check, is_square_attacked


def attacked(pieces: dict[str, str], square: str, by: Side) -> bool:
    return is_square_attacked(board_from(pieces), parse_square(square), by)


# ----------------------------------------------------------------------------- attack detection by piece


def test_chariot_attack_and_blocking() -> None:
    assert is_in_check(board_from({"e0": "K", "d9": "k", "e6": "r"}), Side.RED)
    assert not is_in_check(board_from({"e0": "K", "d9": "k", "e6": "r", "e3": "P"}), Side.RED)


def test_horse_attack_respects_leg_adjacent_to_horse() -> None:
    base = {"e0": "K", "f9": "k", "d2": "n"}  # d2 -> e0 uses leg d1
    assert is_in_check(board_from(base), Side.RED)
    assert not is_in_check(board_from({**base, "d1": "A"}), Side.RED)
    # A piece next to the *target* (e1) is not the leg and does not block.
    assert is_in_check(board_from({**base, "e1": "A"}), Side.RED)


def test_horse_sideways_attack_leg() -> None:
    base = {"e0": "K", "f9": "k", "c1": "n"}  # c1 -> e0 uses leg d1
    assert is_in_check(board_from(base), Side.RED)
    assert is_in_check(board_from({**base, "d0": "A"}), Side.RED)  # d0 is not the leg
    assert not is_in_check(board_from({**base, "d1": "A"}), Side.RED)


def test_soldier_attacks_forward_and_sideways_after_crossing() -> None:
    assert is_in_check(board_from({"e0": "K", "f9": "k", "e1": "p"}), Side.RED)
    assert is_in_check(board_from({"e0": "K", "f9": "k", "d0": "p"}), Side.RED)
    # A soldier never attacks backward.
    assert not is_in_check(board_from({"e1": "K", "f9": "k", "e0": "p"}), Side.RED)


def test_soldier_does_not_attack_sideways_before_crossing() -> None:
    assert not attacked({"e3": "P"}, "d3", Side.RED)
    assert attacked({"e3": "P"}, "e4", Side.RED)
    assert attacked({"e5": "P"}, "d5", Side.RED)


def test_general_advisor_elephant_attacks_limited_to_their_zones() -> None:
    assert attacked({"e1": "K"}, "d1", Side.RED)
    assert not attacked({"e2": "K"}, "e3", Side.RED)  # outside palace
    assert attacked({"e1": "A"}, "d2", Side.RED)
    assert not attacked({"d2": "A"}, "c3", Side.RED)  # outside palace
    assert attacked({"e2": "B"}, "c4", Side.RED)
    assert not attacked({"e2": "B", "d3": "p"}, "c4", Side.RED)  # eye blocked
    assert not attacked({"c4": "B"}, "e6", Side.RED)  # across the river


def test_cannon_attack_on_empty_square_needs_screen() -> None:
    assert not attacked({"e9": "c"}, "e5", Side.BLACK)
    assert attacked({"e9": "c", "e7": "P"}, "e5", Side.BLACK)


def test_flying_general_is_check_but_not_an_ordinary_attack() -> None:
    board = board_from({"e0": "K", "e9": "k"})
    assert is_in_check(board, Side.RED)
    assert attackers_of(board, parse_square("e0"), Side.BLACK) == []


def test_double_check_lists_both_attackers() -> None:
    board = board_from({"e0": "K", "f9": "k", "e5": "r", "d2": "n"})
    expected = sorted([parse_square("e5"), parse_square("d2")])
    assert attackers_of(board, parse_square("e0"), Side.BLACK) == expected


def test_attack_detection_matches_pseudo_legal_captures_on_random_games() -> None:
    """Reverse-lookup attacks == "some enemy pseudo-legal move captures here".

    For an occupied square the two definitions coincide (pins are ignored by
    both), so this cross-checks the reverse lookup against the generators.
    """
    for game, _ in random_playout_states(seed=7, games=6, max_plies=120):
        board: Board = game.board
        for side in Side:
            targets = {m.to_sq for m in generate_pseudo_legal_moves(board, side)}
            for square, piece in board.occupied(side.opponent):
                assert is_square_attacked(board, square, side) == (square in targets), (
                    game.to_fen(),
                    square,
                )


# ----------------------------------------------------------------------------- checkmate & stalemate

MATE_IN_ONE = {"d0": "K", "a8": "R", "i5": "R", "e9": "k"}


def test_check_is_not_checkmate() -> None:
    game = game_from({"d0": "K", "i5": "R", "e9": "k"})
    game.make_move("i5i9")
    assert game.is_in_check(Side.BLACK)
    assert not game.is_checkmate(Side.BLACK)
    assert ucis(game.legal_moves()) == {"e9e8"}
    assert game.get_result() is GameResult.IN_PROGRESS


def test_checkmate_detected_and_red_wins() -> None:
    game = game_from(MATE_IN_ONE)
    record = game.make_move("i5i9")
    assert record.gave_check
    assert game.is_in_check(Side.BLACK)
    assert game.is_checkmate(Side.BLACK)
    assert game.is_terminal()
    assert game.get_result() is GameResult.RED_WINS
    assert game.termination_reason is TerminationReason.CHECKMATE
    assert game.winner is Side.RED
    assert game.legal_moves() == ()


def test_black_can_checkmate_red() -> None:
    game = game_from({"e0": "K", "a1": "r", "i4": "r", "d9": "k"}, Side.BLACK)
    game.make_move("i4i0")
    assert game.get_result() is GameResult.BLACK_WINS
    assert game.termination_reason is TerminationReason.CHECKMATE


def test_terminal_game_rejects_moves() -> None:
    game = game_from(MATE_IN_ONE)
    game.make_move("i5i9")
    with pytest.raises(GameOverError):
        game.make_move("e9e8")
    assert game.ply == 1


def test_no_legal_move_without_check_is_a_loss() -> None:
    # After i5i8 Black's general on d9 is not in check, but d8 is covered by
    # the chariot and e9 would face the red general: no legal move -> loss.
    game = game_from({"e0": "K", "i5": "R", "d9": "k"})
    game.make_move("i5i8")
    assert not game.is_in_check(Side.BLACK)
    assert game.is_stalemate(Side.BLACK)
    assert not game.is_checkmate(Side.BLACK)
    assert game.get_result() is GameResult.RED_WINS
    assert game.termination_reason is TerminationReason.STALEMATE


def test_starting_position_without_legal_moves_is_terminal_immediately() -> None:
    game = game_from({"e0": "K", "i8": "R", "d9": "k"}, Side.BLACK)
    assert game.is_terminal()
    assert game.get_result() is GameResult.RED_WINS
    assert game.termination_reason is TerminationReason.STALEMATE


def test_checkmate_on_last_allowed_ply_is_still_a_win() -> None:
    game = game_from(MATE_IN_ONE, max_plies=1)
    game.make_move("i5i9")
    assert game.get_result() is GameResult.RED_WINS


def test_mate_queries_for_side_not_to_move() -> None:
    game = Game.new_game()
    for side in Side:
        assert not game.is_checkmate(side)
        assert not game.is_stalemate(side)
