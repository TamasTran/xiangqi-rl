from __future__ import annotations

import numpy as np
import pytest

from xiangqi_rl.engine import (
    Game,
    InvalidMoveFormatError,
    InvalidSquareError,
    Move,
    Side,
)
from xiangqi_rl.env import (
    ACTION_DIM,
    action_to_move,
    index_to_square,
    legal_action_mask,
    move_to_action,
    square_to_index,
)


def test_action_dim() -> None:
    assert ACTION_DIM == 8100


def test_square_roundtrip_and_bounds() -> None:
    for index in range(90):
        assert square_to_index(*index_to_square(index)) == index
    assert square_to_index(0, 0) == 0 and square_to_index(9, 8) == 89
    for bad in (-1, 90, True, 1.0, "3"):
        with pytest.raises(InvalidSquareError):
            index_to_square(bad)  # type: ignore[arg-type]
    for row, col in ((10, 0), (0, 9), (-1, 0)):
        with pytest.raises(InvalidSquareError):
            square_to_index(row, col)


def test_move_action_roundtrip() -> None:
    for action in range(ACTION_DIM):
        from_sq, to_sq = divmod(action, 90)
        if from_sq == to_sq:
            continue
        move = action_to_move(action)
        assert (move.from_sq, move.to_sq) == (from_sq, to_sq)
        assert move_to_action(move) == action == 90 * from_sq + to_sq


@pytest.mark.parametrize("bad", [-1, 8100, 0, 91 * 0 + 91, True, 3.0, "7", None])
def test_invalid_actions_rejected_without_coercion(bad) -> None:
    # 0 and 91 are "null" moves (source == destination).
    with pytest.raises(InvalidMoveFormatError):
        action_to_move(bad)


def test_numpy_integers_accepted() -> None:
    assert action_to_move(np.int64(Move.from_uci("h2e2").to_action())) == Move.from_uci("h2e2")


def test_mask_shape_dtype_and_contents() -> None:
    game = Game.new_game()
    mask = legal_action_mask(game)
    assert mask.shape == (ACTION_DIM,) and mask.dtype == np.bool_
    assert mask.sum() == 44
    decoded = {action_to_move(int(a)) for a in np.flatnonzero(mask)}
    assert decoded == set(game.legal_moves())
    for action in np.flatnonzero(mask):
        assert game.is_legal(action_to_move(int(action)))
    assert np.array_equal(mask, legal_action_mask(game))  # deterministic


def test_mask_matches_engine_over_a_game() -> None:
    import random

    rng = random.Random(4)
    game = Game.new_game()
    while not game.is_terminal() and game.ply < 80:
        mask = legal_action_mask(game)
        assert {action_to_move(int(a)) for a in np.flatnonzero(mask)} == set(game.legal_moves())
        # Every non-masked action is rejected by the engine.
        for action in rng.sample(range(ACTION_DIM), 50):
            if not mask[action]:
                try:
                    move = action_to_move(action)
                except InvalidMoveFormatError:
                    continue
                assert not game.is_legal(move)
        game.make_move(rng.choice(game.legal_moves()))


def test_mask_all_false_when_game_over(make_fen) -> None:
    game = Game.from_fen(make_fen({"d0": "K", "a8": "R", "i5": "R", "e9": "k"}))
    game.make_move("i5i9")
    mask = legal_action_mask(game)
    assert mask.shape == (ACTION_DIM,) and not mask.any()
    assert game.side_to_move is Side.BLACK
