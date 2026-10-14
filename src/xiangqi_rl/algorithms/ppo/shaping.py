"""Potential-based reward shaping from material balance (optional).

Shaping term for one transition of the trained agent (decision -> next decision)::

    F = gamma * Phi(s') - Phi(s),     Phi(s) = coef * material_balance(s)

with ``Phi(terminal) = 0``. For potential-based shaping the shaped return of
an episode differs from the original one only by ``-Phi(s_0)`` (Ng, Harada &
Russell, 1999), so the optimal policy is unchanged; the shaping only gives an
earlier, denser signal when material is won or lost. Truncated episodes are not
terminal: ``Phi(s')`` of the final observation is used and the value of that
position is bootstrapped as usual.

``material_balance`` is (agent material - opponent material) / 48, where 48 is
one side's starting material without the general, so it lies in [-1, 1].
"""

from __future__ import annotations

from collections.abc import Mapping

from xiangqi_rl.engine import Board, PieceType, Side

#: Material values used by the potential (general excluded: always present).
SHAPING_PIECE_VALUES: Mapping[PieceType, float] = {
    PieceType.GENERAL: 0.0,
    PieceType.CHARIOT: 9.0,
    PieceType.CANNON: 4.5,
    PieceType.HORSE: 4.0,
    PieceType.ADVISOR: 2.0,
    PieceType.ELEPHANT: 2.0,
    PieceType.SOLDIER: 1.0,
}
#: One side's starting material: 2*9 + 2*4.5 + 2*4 + 2*2 + 2*2 + 5*1.
STARTING_MATERIAL = 48.0


def material_balance(board: Board, side: Side) -> float:
    """``(material(side) - material(opponent)) / 48``, in ``[-1, 1]``."""
    total = 0.0
    for _, piece in board.occupied():
        value = SHAPING_PIECE_VALUES[piece.kind]
        total += value if piece.side is side else -value
    return total / STARTING_MATERIAL
