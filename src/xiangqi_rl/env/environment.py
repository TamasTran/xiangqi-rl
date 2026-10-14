"""Gymnasium environment wrapping the Phase 1 engine.

One ``step(action)`` plays exactly **one ply** by the side to move and then the
turn passes. The environment never plays an opponent move itself; whoever
drives it (a self-play trainer or the match runner) supplies the actions of
both sides, so the same environment serves training and evaluation as either
colour. See ``docs/rl_environment.md`` for the full contract.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from xiangqi_rl.engine import (
    Game,
    GameConfig,
    GameState,
    InvalidMoveFormatError,
    Side,
    XiangqiError,
)
from xiangqi_rl.env.action_space import ACTION_DIM, action_to_move, legal_action_mask
from xiangqi_rl.env.observation import encode_observation, observation_space
from xiangqi_rl.env.rewards import step_reward


class IllegalActionError(ValueError):
    """The action is not an in-range integer or is not legal in this position.

    The environment is left untouched when this is raised.
    """


class EpisodeOverError(RuntimeError):
    """``step()`` was called before ``reset()`` or after the episode ended."""


class XiangqiEnv(gym.Env[np.ndarray, int]):
    """Single-environment, side-to-move Xiangqi (Gymnasium API).

    Args:
        game_config: Engine terminal-state policy (ply limit, repetition).
            A natural result under it sets ``terminated=True``.
        max_episode_steps: Optional *external* limit on steps per episode.
            Reaching it without a natural result sets ``truncated=True``.
        reward_perspective: ``None`` for rewards from the mover's perspective
            (default); ``Side.RED``/``Side.BLACK`` for a fixed perspective.
        initial_fen: Start every episode from this FEN instead of the standard
            position (can be overridden per reset via ``options["fen"]``).
        render_mode: ``None`` or ``"ansi"`` (a text board).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        *,
        game_config: GameConfig | None = None,
        max_episode_steps: int | None = None,
        reward_perspective: Side | None = None,
        initial_fen: str | None = None,
        render_mode: str | None = None,
    ) -> None:
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError(f"max_episode_steps must be positive or None, got {max_episode_steps}")
        if render_mode not in (None, "ansi"):
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        self.observation_space = observation_space
        self.action_space = spaces.Discrete(ACTION_DIM)
        self.render_mode = render_mode
        self._game_config = game_config if game_config is not None else GameConfig()
        self._max_episode_steps = max_episode_steps
        self._reward_perspective = reward_perspective
        self._initial_fen = initial_fen
        self._game: Game | None = None
        self._steps = 0
        self._episode_over = False

    # ------------------------------------------------------------------ gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        ``options["fen"]`` starts from a custom position. ``seed`` seeds
        ``self.np_random`` as usual; the engine itself is deterministic and
        consumes no randomness.
        """
        super().reset(seed=seed)
        fen = (options or {}).get("fen", self._initial_fen)
        try:
            self._game = (
                Game.from_fen(fen, self._game_config)
                if fen is not None
                else Game.new_game(self._game_config)
            )
        except XiangqiError as exc:
            raise ValueError(f"Cannot start episode from FEN {fen!r}: {exc}") from exc
        self._steps = 0
        self._episode_over = self._game.is_terminal()
        return self._observation(), self._info(mover=None, truncated=False)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Play one legal move for the side to move.

        Returns ``(observation, reward, terminated, truncated, info)``.

        Raises:
            EpisodeOverError: Not reset yet, or the episode already ended.
            IllegalActionError: ``action`` is not an in-range integer or not
                legal here. The game, step counter and history are unchanged.
        """
        game = self._require_game()
        if self._episode_over:
            raise EpisodeOverError("Episode has ended; call reset() before step()")

        move = self._decode_legal(game, action)
        mover = game.side_to_move
        game.make_move(move)
        self._steps += 1

        terminated = game.is_terminal()
        truncated = (
            not terminated
            and self._max_episode_steps is not None
            and self._steps >= self._max_episode_steps
        )
        self._episode_over = terminated or truncated
        reward = step_reward(game.get_result(), mover, self._reward_perspective)
        return self._observation(), reward, terminated, truncated, self._info(mover, truncated)

    def render(self) -> str | None:
        """Text diagram of the board when ``render_mode == "ansi"``."""
        if self.render_mode != "ansi":
            return None
        return str(self._require_game())

    def close(self) -> None:
        """Release the game. No external resources are held."""
        self._game = None

    # ------------------------------------------------------------------ inspection

    def action_masks(self) -> np.ndarray:
        """Boolean legal-action mask, shape ``(8100,)``; all ``False`` once the game is over."""
        return legal_action_mask(self._require_game())

    @property
    def state(self) -> GameState:
        """Immutable snapshot of the engine state (board, side, ply, result)."""
        return self._require_game().state

    @property
    def side_to_move(self) -> Side:
        """The side that must move next."""
        return self._require_game().side_to_move

    @property
    def steps(self) -> int:
        """Steps (plies) taken in the current episode."""
        return self._steps

    def game_copy(self) -> Game:
        """An independent copy of the engine game, safe for agents to inspect."""
        return self._require_game().copy()

    # ------------------------------------------------------------------ internals

    def _require_game(self) -> Game:
        if self._game is None:
            raise EpisodeOverError("Environment has not been reset; call reset() first")
        return self._game

    @staticmethod
    def _decode_legal(game: Game, action: int):
        try:
            move = action_to_move(action)
        except InvalidMoveFormatError as exc:
            raise IllegalActionError(f"Invalid action {action!r}: {exc}") from exc
        if not game.is_legal(move):
            raise IllegalActionError(
                f"Action {int(action)} ({move.uci()}) is not legal for "
                f"{game.side_to_move.name} in this position"
            )
        return move

    def _observation(self) -> np.ndarray:
        return encode_observation(self._require_game())

    def _info(self, mover: Side | None, truncated: bool) -> dict[str, Any]:
        game = self._require_game()
        return {
            "action_mask": legal_action_mask(game),
            "current_player": int(game.side_to_move),
            "mover": None if mover is None else int(mover),
            "ply": game.ply,
            "steps": self._steps,
            "result": game.get_result().value,
            "termination_reason": (
                None if game.termination_reason is None else game.termination_reason.value
            ),
            "engine_terminal": game.is_terminal(),
            "externally_truncated": truncated,
        }
