"""Uniformly random legal-move baseline."""

from __future__ import annotations

import random

from xiangqi_rl.agents.base import NoLegalActionError
from xiangqi_rl.engine import Game


class RandomAgent:
    """Picks uniformly among the engine's legal moves.

    Uses a private ``random.Random(seed)``, never the global generator, so two
    agents with the same seed make the same choices in the same positions.
    """

    def __init__(self, seed: int | None = None, name: str = "random") -> None:
        self.name = name
        self._rng = random.Random(seed)

    @property
    def rng_state(self) -> tuple:
        """State of the agent's private RNG (for exact checkpoint/resume)."""
        return self._rng.getstate()

    @rng_state.setter
    def rng_state(self, state: tuple) -> None:
        self._rng.setstate(state)

    def select_action(self, game: Game) -> int:
        """A uniformly random legal action; raises if the game is over."""
        moves = game.legal_moves()  # sorted, so the choice depends only on the seed
        if not moves:
            raise NoLegalActionError("No legal moves: the game is over")
        return self._rng.choice(moves).to_action()
