"""Fixtures shared by the env/agents/evaluation tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest

from xiangqi_rl.engine import Board, Side


@pytest.fixture
def make_fen() -> Callable[[Mapping[str, str], Side], str]:
    """``make_fen({"e0": "K", "e9": "k"}, Side.RED)`` -> FEN string."""

    def build(pieces: Mapping[str, str], side: Side = Side.RED) -> str:
        placement = Board.from_pieces(pieces).to_fen_placement()
        return f"{placement} {'w' if side is Side.RED else 'b'} - - 0 1"

    return build


#: Red to move, ``i5i9`` is checkmate.
RED_MATE_IN_ONE = {"d0": "K", "a8": "R", "i5": "R", "e9": "k"}
#: Black to move, ``i4i0`` is checkmate.
BLACK_MATE_IN_ONE = {"e0": "K", "a1": "r", "i4": "r", "d9": "k"}
