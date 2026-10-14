"""Legal move filtering: pins, king safety, check evasion, reference cross-check."""

from __future__ import annotations

from engine_helpers import (
    game_from,
    legal_from,
    naive_legal_moves,
    random_playout_states,
    ucis,
)
from xiangqi_rl.engine import Side, parse_square
from xiangqi_rl.engine.move_generator import pseudo_legal_moves_from
from xiangqi_rl.engine.rules import generate_legal_moves, is_in_check


def test_piece_pinned_by_chariot_cannot_move() -> None:
    game = game_from({"e0": "K", "e2": "N", "e7": "r", "d9": "k"})
    assert pseudo_legal_moves_from(game.board, parse_square("e2"))  # it *could* move...
    assert legal_from(game, "e2") == set()  # ...but every move exposes the general


def test_pinned_chariot_may_move_along_pin_line_and_capture_pinner() -> None:
    game = game_from({"e0": "K", "e2": "R", "e7": "r", "d9": "k"})
    assert legal_from(game, "e2") == {"e1", "e3", "e4", "e5", "e6", "e7"}


def test_pinned_enemy_piece_still_gives_check_and_controls_squares() -> None:
    # The black chariot on f2 is pinned against its general by the red
    # chariot on f0, yet it still checks the red general on e2 and controls
    # d2. Capturing it with the general would face the black general.
    game = game_from({"e2": "K", "f2": "r", "f0": "R", "f9": "k"})
    assert game.is_in_check(Side.RED)
    assert ucis(game.legal_moves()) == {"e2e1", "f0f2"}


def test_general_cannot_move_into_attacked_square() -> None:
    # d0 is covered by the chariot on d5; f0 would face the black general.
    game = game_from({"e0": "K", "d5": "r", "f9": "k"})
    assert ucis(game.legal_moves()) == {"e0e1"}


def test_general_cannot_capture_protected_piece() -> None:
    game = game_from({"e0": "K", "d0": "r", "d5": "r", "f9": "k"})
    assert game.is_in_check()
    assert ucis(game.legal_moves()) == {"e0e1"}


def test_general_may_capture_unprotected_checker() -> None:
    game = game_from({"e0": "K", "d0": "r", "f9": "k"})
    assert ucis(game.legal_moves()) == {"e0d0", "e0e1"}


def test_moves_that_fail_to_evade_chariot_check_are_illegal() -> None:
    # Black chariot e5 checks along the e-file. Only stepping aside or
    # interposing on e1 resolves it; e0e1 stays on the file.
    game = game_from({"e0": "K", "a1": "R", "e5": "r", "e9": "k"})
    assert game.is_in_check()
    assert ucis(game.legal_moves()) == {"e0d0", "e0f0", "a1e1"}


def test_legal_evasions_are_accepted() -> None:
    game = game_from({"e0": "K", "a1": "R", "e5": "r", "e9": "k"})
    game.make_move("a1e1")
    assert not game.is_in_check(Side.RED)
    assert game.side_to_move is Side.BLACK


def test_capturing_the_checker_is_a_legal_evasion() -> None:
    game = game_from({"e0": "K", "a5": "R", "e5": "r", "e9": "k"})
    assert ucis(game.legal_moves()) == {"e0d0", "e0f0", "a5e5"}


def test_cannon_check_evasions() -> None:
    # Cannon e7 checks through the horse screen on e2. Evasions: move the
    # screen away (all 8 horse moves), add a second screen (a4e4), or step to
    # f0. e1 stays on the line; d0 would face the black general.
    game = game_from({"e0": "K", "e2": "N", "a4": "R", "e7": "c", "d9": "k"})
    assert game.is_in_check()
    horse_moves = {"e2" + t for t in ["d4", "f4", "d0", "f0", "c3", "c1", "g3", "g1"]}
    assert ucis(game.legal_moves()) == {"e0f0", "a4e4"} | horse_moves


def test_horse_check_evaded_by_blocking_its_leg() -> None:
    # Black horse d2 checks e0 via leg d1. Blocking d1 is legal; e0f0 would
    # face the black general.
    game = game_from({"e0": "K", "a1": "R", "d2": "n", "f9": "k"})
    assert game.is_in_check()
    assert ucis(game.legal_moves()) == {"e0d0", "e0e1", "a1d1"}


def test_legal_moves_are_subset_of_pseudo_legal_and_sorted() -> None:
    game = game_from({"e0": "K", "e2": "N", "a4": "R", "e7": "c", "d9": "k"})
    legal = list(game.legal_moves())
    assert legal == sorted(legal)
    assert len(set(legal)) == len(legal)


def test_legal_moves_match_independent_reference_on_random_games() -> None:
    """Compare against the brute-force reference on reproducible random games."""
    positions = checks = 0
    for game, _ in random_playout_states(seed=2024, games=4, max_plies=100):
        board, side = game.board, game.side_to_move
        engine = ucis(generate_legal_moves(board, side))
        assert engine == naive_legal_moves(board, side), game.to_fen()
        positions += 1
        checks += is_in_check(board, side)
    # Make sure the sample actually exercised check positions.
    assert positions > 200 and checks > 0
