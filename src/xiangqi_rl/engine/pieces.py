"""Piece model.

Pieces have *value semantics*: a :class:`Piece` is just ``(side, kind)``. The
rules of Xiangqi never need to distinguish two red horses from each other, so
the engine does not track per-piece identity. Pieces are immutable and
hashable, which makes boards safe to share between game states.

Text symbols follow the common Xiangqi FEN convention: Red is upper case,
Black is lower case, and

    K general, A advisor, B elephant (bishop), N horse (knight),
    R chariot (rook), C cannon, P soldier (pawn).

``E`` (elephant) and ``H`` (horse) are accepted as input aliases.
"""

from __future__ import annotations

from dataclasses import dataclass

from xiangqi_rl.engine.constants import PieceType, Side

_SYMBOL_BY_KIND: dict[PieceType, str] = {
    PieceType.GENERAL: "k",
    PieceType.ADVISOR: "a",
    PieceType.ELEPHANT: "b",
    PieceType.HORSE: "n",
    PieceType.CHARIOT: "r",
    PieceType.CANNON: "c",
    PieceType.SOLDIER: "p",
}
_KIND_BY_SYMBOL: dict[str, PieceType] = {
    **{symbol: kind for kind, symbol in _SYMBOL_BY_KIND.items()},
    "e": PieceType.ELEPHANT,
    "h": PieceType.HORSE,
}

#: Number of pieces of each type a side has at the start (and at most ever).
STARTING_PIECE_COUNTS: dict[PieceType, int] = {
    PieceType.GENERAL: 1,
    PieceType.ADVISOR: 2,
    PieceType.ELEPHANT: 2,
    PieceType.HORSE: 2,
    PieceType.CHARIOT: 2,
    PieceType.CANNON: 2,
    PieceType.SOLDIER: 5,
}


@dataclass(frozen=True, slots=True)
class Piece:
    """An immutable Xiangqi piece: a side and a piece type."""

    side: Side
    kind: PieceType

    def __post_init__(self) -> None:
        if not isinstance(self.side, Side):
            raise TypeError(f"side must be a Side, got {self.side!r}")
        if not isinstance(self.kind, PieceType):
            raise TypeError(f"kind must be a PieceType, got {self.kind!r}")

    @property
    def symbol(self) -> str:
        """FEN symbol: upper case for Red, lower case for Black."""
        symbol = _SYMBOL_BY_KIND[self.kind]
        return symbol.upper() if self.side is Side.RED else symbol

    @classmethod
    def from_symbol(cls, symbol: str) -> Piece:
        """Create a piece from its FEN symbol (e.g. ``'K'`` or ``'p'``)."""
        if not isinstance(symbol, str) or len(symbol) != 1:
            raise ValueError(f"Invalid piece symbol {symbol!r}")
        kind = _KIND_BY_SYMBOL.get(symbol.lower())
        if kind is None:
            raise ValueError(f"Unknown piece symbol {symbol!r}")
        side = Side.RED if symbol.isupper() else Side.BLACK
        return cls(side, kind)

    def __str__(self) -> str:
        return self.symbol
