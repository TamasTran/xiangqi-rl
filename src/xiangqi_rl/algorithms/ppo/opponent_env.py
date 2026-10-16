"""Fixed-perspective training wrapper around :class:`XiangqiEnv`.

The base environment is untouched (one ply per ``step``, both sides supplied
by the caller). This wrapper turns it into a single-agent problem *for
training only*:

* the trained agent plays one side (``agent_side``), a baseline opponent the
  other;
* ``step(action)`` plays the agent's move, then lets the opponent move until
  it is the agent's turn again (or the game ends);
* every returned observation is therefore a decision point of the trained
  agent (side to move == ``agent_side``), so rewards and value estimates share
  one perspective and ordinary GAE applies;
* reward is the game result from ``agent_side``'s point of view, given only
  when the game ends naturally; an external truncation yields reward 0;
* optionally, a potential-based material shaping term is added (see
  :mod:`shaping`); ``info["game_reward"]`` always holds the unshaped result
  reward and ``info["shaping_reward"]`` the shaping term.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import numpy as np

from xiangqi_rl.agents.base import Agent
from xiangqi_rl.algorithms.ppo.shaping import material_balance
from xiangqi_rl.engine import GameConfig, Side
from xiangqi_rl.env import EpisodeOverError, XiangqiEnv, result_for_side

_SCHEDULES = ("alternate", "red", "black", "random")


class OpponentEnv:
    """Single-agent view of Xiangqi against sampled baseline opponents.

    Args:
        opponents: Candidate opponents; one is drawn per episode.
        game_config: Engine rules (draw by ply limit, repetition).
        max_episode_plies: External limit in plies (causes ``truncated``).
        color_schedule: ``alternate`` (RED on even episodes), ``red``,
            ``black`` or ``random``.
        seed: Seed of the wrapper's own RNG (opponent / colour draws).
        initial_fen: Optional custom start position (mainly for tests).
        material_shaping_coef: Scale of the material potential (0 disables).
        shaping_gamma: Discount used in ``gamma * Phi(s') - Phi(s)``; must equal
            the PPO ``gamma`` (per agent decision) for the shaping to be
            potential-based.
        color_offset: Shifts the ``alternate`` schedule (odd: BLACK first), so
            parallel environments do not all start with the same colour.
    """

    def __init__(
        self,
        opponents: Sequence[Agent],
        *,
        game_config: GameConfig | None = None,
        max_episode_plies: int | None = None,
        color_schedule: str = "alternate",
        seed: int = 0,
        initial_fen: str | None = None,
        material_shaping_coef: float = 0.0,
        shaping_gamma: float = 1.0,
        color_offset: int = 0,
    ) -> None:
        if not opponents:
            raise ValueError("at least one opponent is required")
        if material_shaping_coef < 0:
            raise ValueError(f"material_shaping_coef must be >= 0, got {material_shaping_coef}")
        if not 0.0 < shaping_gamma <= 1.0:
            raise ValueError(f"shaping_gamma must be in (0, 1], got {shaping_gamma}")
        if color_schedule not in _SCHEDULES:
            raise ValueError(f"color_schedule must be one of {_SCHEDULES}, got {color_schedule!r}")
        self._env = XiangqiEnv(game_config=game_config, max_episode_steps=max_episode_plies,
                               initial_fen=initial_fen)
        self._opponents = list(opponents)
        self._schedule = color_schedule
        self._rng = random.Random(seed)
        self.episodes_started = 0
        self.agent_side: Side = Side.RED
        self.opponent: Agent = self._opponents[0]
        self._over = True
        self._opponent_moves = 0
        self._shaping_coef = material_shaping_coef
        self._shaping_gamma = shaping_gamma
        self._color_offset = color_offset
        self._potential = 0.0

    # ------------------------------------------------------------------ RNG state

    @property
    def rng_state(self) -> tuple:
        """State of the wrapper's RNG (for exact checkpoint/resume)."""
        return self._rng.getstate()

    @rng_state.setter
    def rng_state(self, state: tuple) -> None:
        self._rng.setstate(state)

    # ------------------------------------------------------------------ API

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return tuple(self._env.observation_space.shape)

    @property
    def action_dim(self) -> int:
        return int(self._env.action_space.n)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an episode; if the agent plays BLACK the opponent opens."""
        self.agent_side = self._pick_side()
        self.opponent = (
            self._opponents[0]
            if len(self._opponents) == 1
            else self._opponents[self._rng.randrange(len(self._opponents))]
        )
        self.episodes_started += 1
        self._opponent_moves = 0
        obs, _ = self._env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated) and self._env.side_to_move is not self.agent_side:
            obs, terminated, truncated = self._opponent_move()
        self._over = terminated or truncated
        if self._over:
            raise RuntimeError("episode ended before the trained agent could move")
        self._potential = self._phi()
        return obs, self._info(terminated=False, truncated=False)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Play ``action`` for the agent, then the opponent's replies.

        Returns ``(obs, reward, terminated, truncated, info)`` where ``obs`` is
        the next decision point (or the final observation if the episode
        ended) and ``info["action_mask"]`` its legal-action mask (all False
        when the episode ended).

        Raises:
            IllegalActionError: for an illegal ``action`` (state unchanged).
            EpisodeOverError: if the episode already ended.
        """
        if self._over:
            raise EpisodeOverError("episode ended; call reset()")
        obs, _, terminated, truncated, _ = self._env.step(action)
        while not (terminated or truncated) and self._env.side_to_move is not self.agent_side:
            obs, terminated, truncated = self._opponent_move()
        self._over = terminated or truncated
        game_reward = result_for_side(self._env.state.result, self.agent_side) if terminated else 0.0
        shaping = 0.0
        if self._shaping_coef > 0:
            next_potential = 0.0 if terminated else self._phi()  # Phi(terminal) = 0
            shaping = self._shaping_gamma * next_potential - self._potential
            self._potential = next_potential
        info = self._info(terminated, truncated)
        info["game_reward"] = game_reward
        info["shaping_reward"] = shaping
        return obs, game_reward + shaping, terminated, truncated, info

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------ internals

    def _phi(self) -> float:
        if self._shaping_coef == 0:
            return 0.0
        return self._shaping_coef * material_balance(self._env.state.board, self.agent_side)

    def _pick_side(self) -> Side:
        if self._schedule == "red":
            return Side.RED
        if self._schedule == "black":
            return Side.BLACK
        if self._schedule == "random":
            return Side.RED if self._rng.random() < 0.5 else Side.BLACK
        return Side.RED if (self.episodes_started + self._color_offset) % 2 == 0 else Side.BLACK

    def _opponent_move(self) -> tuple[np.ndarray, bool, bool]:
        action = self.opponent.select_action(self._env.game_copy())
        obs, _, terminated, truncated, _ = self._env.step(action)
        self._opponent_moves += 1
        return obs, terminated, truncated

    def _info(self, terminated: bool, truncated: bool) -> dict[str, Any]:
        state = self._env.state
        return {
            "action_mask": self._env.action_masks(),
            "agent_side": int(self.agent_side),
            "side_to_move": int(state.side_to_move),
            "opponent": self.opponent.name,
            "opponent_moves": self._opponent_moves,
            "ply": state.ply,
            "result": state.result.value,
            "termination_reason": None if state.termination is None else state.termination.value,
            "terminated": terminated,
            "truncated": truncated,
        }
