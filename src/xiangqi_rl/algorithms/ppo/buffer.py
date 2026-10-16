"""Fixed-size on-policy rollout buffer (NumPy, independent of the network).

Holds exactly one rollout of ``capacity`` decisions. It is *not* a replay
buffer: after an update it must be :meth:`reset` and refilled with data from
the updated policy. Minibatch iteration is refused until the buffer is full
and advantages have been computed, which prevents training on partial or
stale data.

With ``num_envs = E`` parallel environments the decisions are stored
time-major and interleaved: entry ``t * E + e`` is step ``t`` of environment
``e`` (so ``E = 1`` is a plain sequence). GAE runs separately along each
environment's own sequence ``arr[e::E]``; minibatches mix all of them.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from xiangqi_rl.algorithms.ppo.returns import compute_gae, normalize_advantages


class BufferStateError(RuntimeError):
    """The buffer was used in the wrong state (not full, stale, ...)."""


@dataclass(frozen=True)
class Minibatch:
    """Aligned slices of the buffer (copies, safe to modify)."""

    indices: np.ndarray
    observations: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


class RolloutBuffer:
    """Storage for one PPO rollout.

    Args:
        capacity: Number of decisions per rollout (all environments together).
        observation_shape: e.g. ``(17, 10, 9)``.
        action_dim: Size of the action space (mask length).
        num_envs: Parallel environments; must divide ``capacity``.
    """

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        action_dim: int,
        num_envs: int = 1,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if num_envs < 1 or capacity % num_envs:
            raise ValueError(f"num_envs={num_envs} must be >= 1 and divide capacity={capacity}")
        self.capacity = capacity
        self.num_envs = num_envs
        self.observation_shape = tuple(observation_shape)
        self.action_dim = action_dim
        n = capacity
        self.observations = np.zeros((n, *self.observation_shape), dtype=np.float32)
        self.masks = np.zeros((n, action_dim), dtype=np.bool_)
        self.actions = np.zeros(n, dtype=np.int64)
        self.log_probs = np.zeros(n, dtype=np.float32)
        self.values = np.zeros(n, dtype=np.float32)
        self.rewards = np.zeros(n, dtype=np.float32)
        self.terminated = np.zeros(n, dtype=np.bool_)
        self.truncated = np.zeros(n, dtype=np.bool_)
        self.truncation_values = np.zeros(n, dtype=np.float32)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.raw_advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)
        self._size = 0
        self._advantages_ready = False

    # ------------------------------------------------------------------ state

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        """Whether ``capacity`` decisions have been stored."""
        return self._size == self.capacity

    def reset(self) -> None:
        """Discard all data (call after every update)."""
        self._size = 0
        self._advantages_ready = False

    # ------------------------------------------------------------------ writing

    def add(
        self,
        observation: np.ndarray,
        mask: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        terminated: bool,
        truncated: bool,
        truncation_value: float = 0.0,
    ) -> None:
        """Append one decision. ``mask`` is the legal-action mask that was
        used to compute ``log_prob`` and must be reused at update time."""
        if self.is_full:
            raise BufferStateError("buffer is full; compute advantages, update, then reset()")
        if observation.shape != self.observation_shape:
            raise ValueError(f"observation shape {observation.shape} != {self.observation_shape}")
        if mask.shape != (self.action_dim,) or mask.dtype != np.bool_:
            raise ValueError(f"mask must be bool of shape ({self.action_dim},)")
        if not (0 <= int(action) < self.action_dim) or not mask[int(action)]:
            raise ValueError(f"action {action} is not legal under the stored mask")
        if terminated and truncated:
            raise ValueError("a step cannot be both terminated and truncated")
        if not all(np.isfinite(x) for x in (log_prob, value, reward, truncation_value)):
            raise ValueError("log_prob, value, reward and truncation_value must be finite")
        i = self._size
        self.observations[i] = observation
        self.masks[i] = mask
        self.actions[i] = int(action)
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.rewards[i] = reward
        self.terminated[i] = terminated
        self.truncated[i] = truncated
        self.truncation_values[i] = truncation_value
        self._size += 1
        self._advantages_ready = False

    def add_batch(
        self,
        observations: np.ndarray,
        masks: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        truncation_values: np.ndarray,
    ) -> None:
        """Append one step of all ``num_envs`` environments (row ``e`` = env ``e``)."""
        if len(self) % self.num_envs:
            raise BufferStateError("add_batch() cannot follow a partial add() row")
        if len(observations) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} rows, got {len(observations)}")
        for e in range(self.num_envs):
            self.add(observations[e], masks[e], int(actions[e]), float(log_probs[e]),
                     float(values[e]), float(rewards[e]), bool(terminated[e]),
                     bool(truncated[e]), float(truncation_values[e]))

    # ------------------------------------------------------------------ GAE

    def compute_advantages(
        self,
        last_value: float | np.ndarray,
        gamma: float,
        gae_lambda: float,
        normalize: bool = True,
    ) -> None:
        """Fill ``returns`` and ``advantages`` (``raw_advantages`` keeps the
        un-normalised values for diagnostics).

        ``last_value`` is the value of the observation *after* the last stored
        decision of each environment (a live decision point of the trained
        agent): a float for one environment or an array of ``num_envs``.
        """
        if not self.is_full:
            raise BufferStateError(f"buffer holds {self._size}/{self.capacity} decisions")
        last = np.broadcast_to(np.asarray(last_value, dtype=np.float64), (self.num_envs,))
        E = self.num_envs
        for e in range(E):
            adv, ret = compute_gae(
                self.rewards[e::E], self.values[e::E], self.terminated[e::E],
                self.truncated[e::E], float(last[e]), self.truncation_values[e::E],
                gamma, gae_lambda,
            )
            self.raw_advantages[e::E] = adv
            self.returns[e::E] = ret
        adv = self.raw_advantages
        self.advantages[:] = normalize_advantages(adv) if normalize else adv
        self._advantages_ready = True

    # ------------------------------------------------------------------ reading

    def minibatches(self, batch_size: int, rng: np.random.Generator) -> Iterator[Minibatch]:
        """Yield shuffled, aligned minibatches covering every stored decision
        exactly once (the last one may be smaller)."""
        if not self.is_full:
            raise BufferStateError(f"buffer holds {self._size}/{self.capacity} decisions")
        if not self._advantages_ready:
            raise BufferStateError("call compute_advantages() before iterating minibatches")
        if not 1 <= batch_size <= self.capacity:
            raise ValueError(f"batch_size must be in [1, {self.capacity}], got {batch_size}")
        order = rng.permutation(self.capacity)
        for start in range(0, self.capacity, batch_size):
            idx = order[start : start + batch_size]
            yield Minibatch(
                indices=idx,
                observations=self.observations[idx],
                masks=self.masks[idx],
                actions=self.actions[idx],
                old_log_probs=self.log_probs[idx],
                old_values=self.values[idx],
                advantages=self.advantages[idx],
                returns=self.returns[idx],
            )
