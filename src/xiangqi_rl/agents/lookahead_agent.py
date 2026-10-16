"""Two-ply material search baseline: stronger than :class:`HeuristicAgent`.

For every legal move ``m`` the agent assumes the opponent then plays its best
reply and scores ``m`` by the worst outcome (minimax over two plies)::

    score(m) = gain(m) - max_r loss(r)

* ``gain(m)``: value of the piece ``m`` captures (+ the soldier bonus).
* ``loss(r)``: value of the piece the reply ``r`` captures, or a "mated" score
  if ``r`` checkmates us.
* If the opponent has *no* legal reply after ``m`` (checkmate or stalemate),
  ``m`` wins outright.
* Repetition (``avoid_repetition_loss``, default on): if ``m`` -- or the
  opponent's reply to ``m`` -- would repeat a position often enough for the
  engine's repetition rule to end the game, the outcome is taken from the
  engine itself: a loss (e.g. perpetual check by us) scores ``-MATE_SCORE``, a
  win ``+MATE_SCORE`` and a draw ``0``.

So, unlike the one-ply heuristic, it does not take a defended piece with a
more valuable one, it moves attacked pieces away, it sees mate in one and it
avoids moves that allow mate in one.

Limits: two plies only (no recapture chains, no deeper tactics); replies that
stalemate us are not detected; only the *immediate* effect of repetition is
seen (it does not plan to avoid a repetition several moves ahead) and the ply
limit is ignored. Without ``avoid_repetition_loss`` it is blind to repetition
and will happily check forever and lose by the perpetual-check rule. It is
a baseline, not a strong engine. It is about 6x slower per move than
:class:`HeuristicAgent` (~25 ms vs ~4 ms in pure Python, measured on the
development CPU), because it generates the opponent's legal replies for every
candidate move.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Literal

from xiangqi_rl.agents.base import NoLegalActionError
from xiangqi_rl.agents.heuristic_agent import HeuristicConfig
from xiangqi_rl.engine import Game, PieceType, Side, TerminationReason, position_key
from xiangqi_rl.engine.constants import has_crossed_river
from xiangqi_rl.engine.rules import generate_legal_moves, has_legal_move, is_in_check

#: Score of a move that wins at once / allows mate in one. Larger than any
#: possible material swing (a side has 48 points of material).
MATE_SCORE = 1000.0


class LookaheadAgent:
    """Minimax over the agent's move and the opponent's best reply."""

    def __init__(
        self,
        config: HeuristicConfig | None = None,
        seed: int | None = None,
        tie_break: Literal["random", "first"] = "random",
        name: str = "lookahead",
        avoid_repetition_loss: bool = True,
    ) -> None:
        if tie_break not in ("random", "first"):
            raise ValueError(f"tie_break must be 'random' or 'first', got {tie_break!r}")
        self.name = name
        self.config = config if config is not None else HeuristicConfig()
        self._tie_break = tie_break
        self.avoid_repetition_loss = avoid_repetition_loss
        self._rng = random.Random(seed)

    @property
    def rng_state(self) -> tuple:
        """State of the agent's private RNG (for exact checkpoint/resume)."""
        return self._rng.getstate()

    @rng_state.setter
    def rng_state(self, state: tuple) -> None:
        self._rng.setstate(state)

    def _value(self, kind: PieceType) -> float:
        # The general is never captured (it is mated instead); its material
        # value is irrelevant here.
        return 0.0 if kind is PieceType.GENERAL else self.config.piece_values[kind]

    def _soldier_bonus(self, side: Side, from_sq: int, to_sq: int) -> float:
        from_row, to_row = from_sq // 9, to_sq // 9
        if (to_row - from_row) * side.forward <= 0:
            return 0.0
        bonus = self.config.soldier_advance_bonus
        if not has_crossed_river(side, from_row) and has_crossed_river(side, to_row):
            bonus += self.config.soldier_cross_bonus
        return bonus

    def score_moves(self, game: Game) -> list[tuple[float, int]]:
        """``(score, action)`` per legal move.

        Scores of the best move(s) are exact. A move is abandoned as soon as
        one reply shows it is strictly worse than the best move found so far,
        so its score is then only an upper bound. The game is never modified
        (all positions are built from the immutable board).
        """
        board = game.board
        side, opponent = game.side_to_move, game.side_to_move.opponent
        cells = board.squares

        threshold = game.config.repetition_threshold if self.avoid_repetition_loss else None
        seen = Counter(game.position_keys) if threshold is not None else Counter()

        def repeats(position, to_move: Side) -> bool:
            """Would reaching ``position`` complete ``threshold`` occurrences?"""
            return threshold is not None and seen[position_key(position, to_move)] + 1 >= threshold

        def repetition_value(*moves) -> float | None:
            """Our value if playing ``moves`` ends the game by the engine's
            repetition rule (asked of the engine itself); else ``None``."""
            trial = game.copy()
            for m in moves:
                trial.make_move(m)
            if trial.termination_reason not in (TerminationReason.REPETITION,
                                                TerminationReason.PERPETUAL_CHECK):
                return None
            if trial.winner is side:
                return MATE_SCORE
            if trial.winner is opponent:
                return -MATE_SCORE
            return 0.0

        def gain(move) -> float:
            target = cells[move.to_sq]
            g = 0.0 if target is None else self._value(target.kind)
            piece = cells[move.from_sq]
            if piece is not None and piece.kind is PieceType.SOLDIER:
                g += self._soldier_bonus(side, move.from_sq, move.to_sq)
            return g

        # Captures first so a good "best so far" prunes the rest early.
        moves = sorted(game.legal_moves(), key=lambda m: -gain(m))
        best = -math.inf
        scored: list[tuple[float, int]] = []
        for move in moves:
            after = board.apply_move(move)
            replies = generate_legal_moves(after, opponent)
            if not replies:  # opponent is checkmated or stalemated: both lose
                score = MATE_SCORE
            elif repeats(after, opponent) and (forced := repetition_value(move)) is not None:
                score = forced  # the move itself ends the game by repetition
            else:
                g = gain(move)
                after_cells = after.squares

                def loss(reply) -> float:
                    victim = after_cells[reply.to_sq]
                    return 0.0 if victim is None else self._value(victim.kind)

                score = math.inf
                for reply in sorted(replies, key=lambda r: -loss(r)):
                    reply_board = after.apply_move(reply)
                    if is_in_check(reply_board, side) and not has_legal_move(reply_board, side):
                        score = -MATE_SCORE  # this reply mates us
                        break
                    value = g - loss(reply)
                    if repeats(reply_board, side):
                        ended = repetition_value(move, reply)
                        if ended is not None:
                            value = ended  # this reply ends the game by repetition
                    score = min(score, value)
                    if score < best:  # already worse than the best move: prune
                        break
            best = max(best, score)
            scored.append((score, move.to_action()))
        return scored

    def select_action(self, game: Game) -> int:
        """The best legal action under two-ply search; raises if the game is over."""
        scored = self.score_moves(game)
        if not scored:
            raise NoLegalActionError("No legal moves: the game is over")
        best = max(score for score, _ in scored)
        candidates = sorted(action for score, action in scored if score == best)
        if self._tie_break == "first":
            return candidates[0]
        return self._rng.choice(candidates)
