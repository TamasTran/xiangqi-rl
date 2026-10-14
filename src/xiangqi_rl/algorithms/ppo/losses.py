"""PPO loss terms (pure tensor functions, easy to test by hand).

Minimised objective::

    L = -L_clip + vf_coef * L_value - ent_coef * H

* ``L_clip = mean(min(r * A, clip(r, 1 - eps, 1 + eps) * A))`` with
  ``r = exp(new_logp - old_logp)``. Both log-probabilities must come from the
  same legal-action mask.
* ``L_value = 0.5 * mean((V - R)^2)``. With value clipping (PPO paper /
  baselines): ``0.5 * mean(max((V - R)^2, (V_old + clip(V - V_old, -e, e) - R)^2))``.
* ``H`` is the mean entropy of the masked policy (nats).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PolicyLossOutput:
    """Policy loss and diagnostics (all detached except ``loss``)."""

    loss: Tensor
    ratio: Tensor
    clip_fraction: Tensor
    approx_kl: Tensor


def ppo_policy_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    clip_coef: float,
) -> PolicyLossOutput:
    """Clipped surrogate loss (to minimise): ``-mean(min(rA, clip(r)A))``."""
    log_ratio = new_log_probs - old_log_probs.detach()
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
    loss = -torch.min(unclipped, clipped).mean()
    with torch.no_grad():
        # k3 estimator of KL(old || new): (r - 1) - log r  (non-negative).
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()
    return PolicyLossOutput(loss, ratio.detach(), clip_fraction, approx_kl)


def value_loss(
    new_values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    clip_coef: float | None = None,
) -> Tensor:
    """``0.5 * MSE`` against ``returns``, optionally with PPO value clipping."""
    unclipped = (new_values - returns) ** 2
    if clip_coef is None:
        return 0.5 * unclipped.mean()
    old = old_values.detach()
    clipped_values = old + torch.clamp(new_values - old, -clip_coef, clip_coef)
    clipped = (clipped_values - returns) ** 2
    return 0.5 * torch.max(unclipped, clipped).mean()


def total_loss(policy_loss: Tensor, v_loss: Tensor, entropy: Tensor,
               vf_coef: float, ent_coef: float) -> Tensor:
    """``policy_loss + vf_coef * value_loss - ent_coef * entropy``."""
    return policy_loss + vf_coef * v_loss - ent_coef * entropy
