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
def act(
    model: nn.Module,
    observation: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[int, float, float]:
    """Choose a legal action for one position.

    Returns ``(action, log_prob, value)`` under the current (behaviour) policy;
    ``value`` is from the side-to-move perspective.
    """
    obs = torch.from_numpy(observation).unsqueeze(0).to(device)
    legal = torch.from_numpy(mask).unsqueeze(0).to(device)
    logits, value = model(obs)
    dist = MaskedCategorical(logits, legal)  # raises on an all-False mask
    action = dist.mode() if greedy else dist.sample(generator)
    log_prob = dist.log_prob(action)
    return int(action.item()), float(log_prob.item()), float(value.item())


@torch.no_grad()
def state_value(model: nn.Module, observation: np.ndarray, device: torch.device) -> float:
    """Value estimate (side-to-move perspective) of a single observation."""
    obs = torch.from_numpy(observation).unsqueeze(0).to(device)
    return float(model(obs)[1].item())


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
