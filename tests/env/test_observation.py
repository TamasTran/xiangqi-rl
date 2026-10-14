from __future__ import annotations

import numpy as np

from xiangqi_rl.engine import Game, GameConfig, Piece, PieceType, Side, parse_square
from xiangqi_rl.env import (
    CHANNEL_NAMES,
    NUM_CHANNELS,
    OBSERVATION_SHAPE,
    encode_observation,
    observation_space,
    piece_channel,
)
from xiangqi_rl.env.observation import (
    PLY_PROGRESS_CHANNEL,
    REPETITION_CHANNEL,
    SIDE_TO_MOVE_CHANNEL,
)


def test_shape_dtype_and_space_membership() -> None:
    obs = encode_observation(Game.new_game())
    assert obs.shape == OBSERVATION_SHAPE == (17, 10, 9)
    assert obs.dtype == np.float32
    assert observation_space.contains(obs)


def test_channel_order_is_fixed() -> None:
    assert NUM_CHANNELS == len(CHANNEL_NAMES) == 17
    assert CHANNEL_NAMES[0] == "red_general" and CHANNEL_NAMES[6] == "red_soldier"
    assert CHANNEL_NAMES[7] == "black_general" and CHANNEL_NAMES[13] == "black_soldier"
    assert CHANNEL_NAMES[14:] == ("side_to_move_is_red", "repetition", "ply_progress")
    assert piece_channel(Side.RED, PieceType.CANNON) == 5
    assert piece_channel(Side.BLACK, PieceType.GENERAL) == 7
    assert piece_channel(Side.BLACK, PieceType.SOLDIER) == 13


def test_both_sides_encoded_at_correct_squares() -> None:
    game = Game.new_game()
    obs = encode_observation(game)
    # Piece planes hold exactly the pieces of the engine board.
    for square, piece in game.board.occupied():
        row, col = divmod(square, 9)
        assert obs[piece_channel(piece.side, piece.kind), row, col] == 1.0
    assert obs[:14].sum() == 32
    for side in Side:
        assert obs[side * 7 : side * 7 + 7].sum() == 16
    # Spot checks written independently of the encoder: e0 is Red's general
    # (row 9, col 4); e9 is Black's general (row 0, col 4).
    assert obs[0, 9, 4] == 1.0 and obs[7, 0, 4] == 1.0
    # Red cannons on b2/h2 -> row 7, cols 1 and 7.
    assert obs[5, 7, 1] == obs[5, 7, 7] == 1.0


def test_side_to_move_plane_and_scalar_planes() -> None:
    game = Game.new_game()
    obs = encode_observation(game)
    assert (obs[SIDE_TO_MOVE_CHANNEL] == 1.0).all()
    assert (obs[REPETITION_CHANNEL] == 0.0).all() and (obs[PLY_PROGRESS_CHANNEL] == 0.0).all()
    game.make_move("h2e2")
    obs = encode_observation(game)
    assert (obs[SIDE_TO_MOVE_CHANNEL] == 0.0).all()
    assert np.allclose(obs[PLY_PROGRESS_CHANNEL], 1 / 300)


def test_repetition_plane_counts_occurrences() -> None:
    game = Game.new_game()  # threshold 3
    for move in ["b0c2", "b9c7", "c2b0", "c7b9"]:
        game.make_move(move)
    # Initial position occurred twice -> (2 - 1) / (3 - 1)
    assert np.allclose(encode_observation(game)[REPETITION_CHANNEL], 0.5)


def test_planes_disabled_without_engine_limits() -> None:
    game = Game.new_game(GameConfig(max_plies=None, repetition_threshold=None))
    game.make_move("h2e2")
    obs = encode_observation(game)
    assert (obs[REPETITION_CHANNEL] == 0).all() and (obs[PLY_PROGRESS_CHANNEL] == 0).all()


def test_encoding_is_deterministic_and_returns_fresh_arrays() -> None:
    game = Game.new_game()
    a, b = encode_observation(game), encode_observation(game)
    assert np.array_equal(a, b) and a is not b
    a[:] = 7.0  # mutating a returned observation must not affect the game
    assert np.array_equal(encode_observation(game), b)
    assert game.board.piece_at(parse_square("e0")) == Piece(Side.RED, PieceType.GENERAL)


def test_equal_positions_give_equal_observations() -> None:
    a, b = Game.new_game(), Game.new_game()
    for m in ["b0c2", "b9c7", "h0g2", "h9g7"]:
        a.make_move(m)
    for m in ["h0g2", "h9g7", "b0c2", "b9c7"]:
        b.make_move(m)
    assert np.array_equal(encode_observation(a), encode_observation(b))
