"""Generalised Advantage Estimation and related helpers (pure NumPy).

Transition ``t`` is one decision of the trained agent. With ``done_t`` meaning
the episode ended after step ``t``::

    next_value_t = 0                       if terminated_t  (natural end)
                 = truncation_values[t]    if truncated_t   (bootstrap from the
                                                              final observation)
                 = values[t + 1]           otherwise (same episode)
                 = last_value              if t is the last stored step
    delta_t = r_t + gamma * next_value_t - V(s_t)
    A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    R_t     = A_t + V(s_t)                 (the value-function target)

``(1 - done_t)`` cuts the recursion at episode boundaries so advantages never
leak between independent episodes. All values must be from the *trained
agent's* perspective (the training wrapper guarantees this).
"""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    last_value: float,
    truncation_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(advantages, returns)`` as float32 arrays of shape ``[n]``."""
    n = len(rewards)
    for name, arr in (("values", values), ("terminated", terminated),
                      ("truncated", truncated), ("truncation_values", truncation_values)):
        if len(arr) != n:
            raise ValueError(f"{name} has length {len(arr)}, expected {n}")
    if np.any(terminated & truncated):
        raise ValueError("a step cannot be both terminated and truncated")

    advantages = np.zeros(n, dtype=np.float64)
    running = 0.0
    for t in reversed(range(n)):
        done = bool(terminated[t] or truncated[t])
        if terminated[t]:
            next_value = 0.0
        elif truncated[t]:
            next_value = float(truncation_values[t])
        elif t == n - 1:
            next_value = float(last_value)
        else:
            next_value = float(values[t + 1])
        delta = float(rewards[t]) + gamma * next_value - float(values[t])
        running = delta + (0.0 if done else gamma * gae_lambda * running)
        advantages[t] = running
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def normalize_advantages(advantages: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean, unit-variance advantages (mean-centred only if std ~ 0)."""
    centred = advantages - advantages.mean()
    std = advantages.std()
    return (centred / (std + eps)).astype(np.float32)


def explained_variance(predicted: np.ndarray, target: np.ndarray) -> float:
    """``1 - Var(target - predicted) / Var(target)``; NaN if ``Var(target) == 0``."""
    var = np.var(target)
    if var < 1e-12:
        return float("nan")
    return float(1.0 - np.var(target - predicted) / var)
