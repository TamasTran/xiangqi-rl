"""Observation encoder: engine position -> ``float32`` array of shape (17, 10, 9).

Board axes use the engine's *absolute* coordinates (row 0 = Black's back rank
at the top, Red at the bottom). The board is **not** flipped for the side to
move; a plane tells the network whose turn it is. Phase 3 may flip the board
itself if it wants a side-to-move-relative view.

Channel layout (fixed, see :data:`CHANNEL_NAMES`)::

     0.. 6   RED   general, advisor, elephant, horse, chariot, cannon, soldier
     7..13   BLACK general, advisor, elephant, horse, chariot, cannon, soldier
    14       side to move: 1.0 everywhere if RED to move, 0.0 if BLACK
    15       repetition: (occurrences of the current position - 1) /
             (repetition_threshold - 1), clipped to [0, 1]; 0 if disabled
    16       ply progress: plies played / max_plies, clipped to [0, 1]; 0 if
             the engine ply limit is disabled

Why the last two planes exist: the engine's repetition and ply-limit rules
depend on history, so the board alone is not a Markov state. Planes 15 and 16
expose how close the game is to those rules. They are still *not* complete:
perpetual-check adjudication also depends on which earlier moves gave check,
which the observation does not include (see ``docs/rl_environment.md``).
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from xiangqi_rl.engine import NUM_COLS, NUM_ROWS, Game, PieceType, Side

PIECES_PER_SIDE = len(PieceType)
SIDE_TO_MOVE_CHANNEL = 2 * PIECES_PER_SIDE  # 14
REPETITION_CHANNEL = SIDE_TO_MOVE_CHANNEL + 1  # 15
PLY_PROGRESS_CHANNEL = SIDE_TO_MOVE_CHANNEL + 2  # 16
NUM_CHANNELS = PLY_PROGRESS_CHANNEL + 1  # 17
OBSERVATION_SHAPE = (NUM_CHANNELS, NUM_ROWS, NUM_COLS)
OBSERVATION_DTYPE = np.float32

CHANNEL_NAMES: tuple[str, ...] = (
    *(f"{side.name.lower()}_{kind.name.lower()}" for side in Side for kind in PieceType),
    "side_to_move_is_red",
    "repetition",
    "ply_progress",
)

observation_space = spaces.Box(
    low=0.0, high=1.0, shape=OBSERVATION_SHAPE, dtype=OBSERVATION_DTYPE
)


def piece_channel(side: Side, kind: PieceType) -> int:
    """Plane index of ``(side, kind)``: ``side * 7 + kind``."""
    return int(side) * PIECES_PER_SIDE + int(kind)


def encode_observation(game: Game) -> np.ndarray:
    """Encode the current position of ``game``. Returns a new array each call."""
    obs = np.zeros(OBSERVATION_SHAPE, dtype=OBSERVATION_DTYPE)
    for square, piece in game.board.occupied():
        row, col = divmod(square, NUM_COLS)
        obs[piece_channel(piece.side, piece.kind), row, col] = 1.0

    if game.side_to_move is Side.RED:
        obs[SIDE_TO_MOVE_CHANNEL] = 1.0

    threshold = game.config.repetition_threshold
    if threshold is not None:
        occurrences = game.position_keys.count(game.position_key())
        obs[REPETITION_CHANNEL] = min(1.0, (occurrences - 1) / (threshold - 1))

    max_plies = game.config.max_plies
    if max_plies is not None:
        obs[PLY_PROGRESS_CHANNEL] = min(1.0, game.ply / max_plies)
    return obs
