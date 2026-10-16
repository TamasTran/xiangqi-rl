"""Acting and re-evaluating actions with one shared masking code path.

``act`` (rollout, no gradient) and ``evaluate_actions`` (update, with
gradient) both build :class:`~xiangqi_rl.nn.MaskedCategorical` from the model
logits and the **stored legal-action mask**, so the old and new
log-probabilities are always computed under the same mask. Raw logits, masked
logits, probabilities and log-probabilities are never mixed: only the masked,
normalised distribution's ``log_prob`` is used.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from xiangqi_rl.nn import MaskedCategorical


@torch.no_grad()
def act_batch(
    model: nn.Module,
    observations: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose a legal action for each of ``N`` positions in one forward pass.

    Returns ``(actions [N] int64, log_probs [N] float32, values [N] float32)``
    under the current (behaviour) policy; values are side-to-move.
    """
    obs = torch.from_numpy(np.ascontiguousarray(observations)).to(device, non_blocking=True)
    legal = torch.from_numpy(np.ascontiguousarray(masks)).to(device, non_blocking=True)
    logits, values = model(obs)
    dist = MaskedCategorical(logits, legal)  # raises on an all-False row
    actions = dist.mode() if greedy else dist.sample(generator)
    log_probs = dist.log_prob(actions)
    return (actions.cpu().numpy().astype(np.int64), log_probs.cpu().numpy().astype(np.float32),
            values.squeeze(-1).cpu().numpy().astype(np.float32))


@torch.no_grad()
def values_batch(model: nn.Module, observations: np.ndarray, device: torch.device) -> np.ndarray:
    """Value estimates ``[N]`` (side-to-move perspective) in one forward pass."""
    obs = torch.from_numpy(np.ascontiguousarray(observations)).to(device, non_blocking=True)
    return model(obs)[1].squeeze(-1).cpu().numpy().astype(np.float32)


def act(
    model: nn.Module,
    observation: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[int, float, float]:
    """Single-position version of :func:`act_batch`: ``(action, log_prob, value)``."""
    a, lp, v = act_batch(model, observation[None], mask[None], device, greedy, generator)
    return int(a[0]), float(lp[0]), float(v[0])


def state_value(model: nn.Module, observation: np.ndarray, device: torch.device) -> float:
    """Value estimate (side-to-move perspective) of a single observation."""
    return float(values_batch(model, observation[None], device)[0])


def evaluate_actions(
    model: nn.Module,
    observations: Tensor,
    masks: Tensor,
    actions: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Differentiable ``(log_prob [B], entropy [B], value [B])`` of ``actions``
    under the model and the given masks."""
    logits, values = model(observations)
    dist = MaskedCategorical(logits, masks)
    return dist.log_prob(actions), dist.entropy(), values.squeeze(-1)
