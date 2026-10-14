"""Minimal agent protocol shared by the baseline agents and the match runner."""

from __future__ import annotations

from typing import Protocol

from xiangqi_rl.engine import Game


class NoLegalActionError(ValueError):
    """The agent was asked to act in a position with no legal move (game over)."""


class Agent(Protocol):
    """Anything with a ``name`` and ``select_action(game) -> action``.

    ``select_action`` must not mutate ``game`` and returns an action index in
    the encoding of :mod:`xiangqi_rl.env.action_space`.
    """

    name: str

    def select_action(self, game: Game) -> int: ...
