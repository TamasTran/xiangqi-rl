"""One-ply material heuristic baseline (not a strong Xiangqi engine).

For every legal move the agent scores the *resulting position* statically:

    score = value of the captured piece
          + soldier bonus (advancing, crossing the river)
          + winning bonus if the move leaves the opponent with no legal move

and plays the best-scoring move. It looks one ply deep only: it does not see
recaptures, so it happily moves a piece onto a defended square, and it does not
avoid being captured. Ties are broken uniformly by a private seeded RNG
(``tie_break="random"``) or by the smallest move (``"first"``).
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from xiangqi_rl.agents.base import NoLegalActionError
from xiangqi_rl.engine import Game, PieceType, Side
from xiangqi_rl.engine.constants import has_crossed_river
from xiangqi_rl.engine.rules import has_legal_move

DEFAULT_PIECE_VALUES: Mapping[PieceType, float] = {
    PieceType.GENERAL: 100.0,
    PieceType.CHARIOT: 9.0,
    PieceType.CANNON: 4.5,
    PieceType.HORSE: 4.0,
    PieceType.ADVISOR: 2.0,
    PieceType.ELEPHANT: 2.0,
    PieceType.SOLDIER: 1.0,
}


@dataclass(frozen=True)
class HeuristicConfig:
    """Tunable evaluation parameters (not claims of correct Xiangqi values)."""

    piece_values: Mapping[PieceType, float] = field(
        default_factory=lambda: dict(DEFAULT_PIECE_VALUES)
    )
    soldier_advance_bonus: float = 0.05
    """Bonus per step a soldier moves forward."""
    soldier_cross_bonus: float = 0.5
    """Extra bonus for the move that carries a soldier across the river."""
    win_bonus: float = 1000.0
    """Bonus for a move after which the opponent has no legal move."""


class HeuristicAgent:
    """Greedy one-ply material agent. See the module docstring for limits."""

    def __init__(
        self,
        config: HeuristicConfig | None = None,
        seed: int | None = None,
        tie_break: Literal["random", "first"] = "random",
        name: str = "heuristic",
    ) -> None:
        if tie_break not in ("random", "first"):
            raise ValueError(f"tie_break must be 'random' or 'first', got {tie_break!r}")
        self.name = name
        self.config = config if config is not None else HeuristicConfig()
        self._tie_break = tie_break
        self._rng = random.Random(seed)

    @property
    def rng_state(self) -> tuple:
        """State of the agent's private RNG (for exact checkpoint/resume)."""
        return self._rng.getstate()

    @rng_state.setter
    def rng_state(self, state: tuple) -> None:
        self._rng.setstate(state)

    def score_moves(self, game: Game) -> list[tuple[float, int]]:
        """``(score, action)`` for each legal move, in move order.

        Only reads ``game``; candidate positions are built from the immutable
        board, so the real game is never modified.
        """
        board = game.board
        side = game.side_to_move
        cfg = self.config
        scored: list[tuple[float, int]] = []
        for move in game.legal_moves():
            piece = board.squares[move.from_sq]
            captured = board.squares[move.to_sq]
            score = 0.0
            if captured is not None:
                score += cfg.piece_values[captured.kind]
            if piece is not None and piece.kind is PieceType.SOLDIER:
                score += self._soldier_bonus(side, move.from_sq, move.to_sq)
            if not has_legal_move(board.apply_move(move), side.opponent):
                score += cfg.win_bonus
            scored.append((score, move.to_action()))
        return scored

    def _soldier_bonus(self, side: Side, from_sq: int, to_sq: int) -> float:
        from_row, to_row = from_sq // 9, to_sq // 9
        bonus = 0.0
        if (to_row - from_row) * side.forward > 0:
            bonus += self.config.soldier_advance_bonus
            if not has_crossed_river(side, from_row) and has_crossed_river(side, to_row):
                bonus += self.config.soldier_cross_bonus
        return bonus

    def select_action(self, game: Game) -> int:
        """The best-scoring legal action; raises if the game is over."""
        scored = self.score_moves(game)
        if not scored:
            raise NoLegalActionError("No legal moves: the game is over")
        best = max(score for score, _ in scored)
        candidates = [action for score, action in scored if score == best]
        if self._tie_break == "first":
            return min(candidates)
        return self._rng.choice(candidates)
