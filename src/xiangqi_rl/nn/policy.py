"""Legal-action masking and the masked categorical policy distribution.

Kept separate from the network on purpose: the network scores *every* action;
this module restricts the choice to legal ones.

Conventions
-----------
* ``mask`` is a ``bool`` tensor ``[B, action_dim]``, ``True`` = legal (exactly
  the Phase 2 ``action_masks()`` / ``info["action_mask"]``, batched).
* Illegal logits are replaced by the most negative finite value of the
  logits' dtype (not ``-inf``), so probabilities are exactly 0 while
  ``log_softmax``, entropy and gradients stay NaN-free.
* A row with no legal action (a terminal position) raises
  :class:`EmptyLegalMaskError`; it is never silently turned into NaNs or a
  uniform distribution. Use :func:`terminal_rows` to find such rows and keep
  them out of the policy (their value target is known from the game result).

PPO requirement (Phase 4): the **same mask** must be used when computing the
old log-probability (at rollout time) and the new one (at update time) for a
given state-action pair. Store the mask with the transition.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


class EmptyLegalMaskError(ValueError):
    """At least one row of the mask has no legal action (terminal position)."""


def _validate(logits: Tensor, mask: Tensor) -> None:
    if not isinstance(logits, Tensor) or not isinstance(mask, Tensor):
        raise TypeError("logits and mask must be torch.Tensors")
    if logits.dim() != 2:
        raise ValueError(f"logits must have shape [batch, action_dim], got {list(logits.shape)}")
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must have dtype torch.bool, got {mask.dtype}")
    if mask.shape != logits.shape:
        raise ValueError(
            f"mask shape {list(mask.shape)} does not match logits shape {list(logits.shape)}"
        )
    if not logits.is_floating_point():
        raise TypeError(f"logits must be floating point, got {logits.dtype}")


def terminal_rows(mask: Tensor) -> Tensor:
    """Bool ``[B]``: rows with no legal action, i.e. finished games."""
    return ~mask.any(dim=-1)


def masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    """Logits with illegal actions set to the dtype's most negative value.

    Raises:
        EmptyLegalMaskError: if any row has no legal action.
    """
    _validate(logits, mask)
    empty = terminal_rows(mask)
    if bool(empty.any()):
        rows = torch.nonzero(empty).flatten().tolist()
        raise EmptyLegalMaskError(
            f"Rows {rows} have no legal action (terminal positions); "
            "filter them out with terminal_rows() before building a policy"
        )
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


class MaskedCategorical:
    """Categorical distribution over the legal actions of each batch row.

    Args:
        logits: Raw network logits ``[B, action_dim]``.
        mask: Legal-action mask ``[B, action_dim]`` (bool).
    """

    def __init__(self, logits: Tensor, mask: Tensor) -> None:
        self.mask = mask
        self.logits = F.log_softmax(masked_logits(logits, mask), dim=-1)  # normalised

    @property
    def probs(self) -> Tensor:
        """Probabilities ``[B, action_dim]``; exactly 0 for illegal actions."""
        return self.logits.exp()

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        """Sample one legal action per row, shape ``[B]`` (``int64``)."""
        return torch.multinomial(self.probs, 1, generator=generator).squeeze(-1)

    def mode(self) -> Tensor:
        """Most probable legal action per row, shape ``[B]``."""
        return self.logits.argmax(dim=-1)

    def log_prob(self, actions: Tensor) -> Tensor:
        """Log-probability ``[B]`` of ``actions`` (``[B]`` integer tensor).

        Raises:
            ValueError: if an action is out of range or illegal under this
                mask. For PPO this catches an old/new mask mismatch early.
        """
        if actions.dtype not in (torch.int64, torch.int32) or actions.shape != self.logits.shape[:1]:
            raise ValueError(
                f"actions must be an integer tensor of shape [{self.logits.shape[0]}], "
                f"got {actions.dtype} {list(actions.shape)}"
            )
        actions = actions.long()
        if bool(((actions < 0) | (actions >= self.logits.shape[-1])).any()):
            raise ValueError("actions contain out-of-range indices")
        idx = actions.unsqueeze(-1)
        if not bool(self.mask.gather(-1, idx).all()):
            raise ValueError("actions contain illegal actions under the given mask")
        return self.logits.gather(-1, idx).squeeze(-1)

    def entropy(self) -> Tensor:
        """Entropy ``[B]`` in nats over the legal actions."""
        return -(self.probs * self.logits).sum(dim=-1)


def value_to_red_perspective(value: Tensor, observations: Tensor) -> Tensor:
    """Convert side-to-move values ``[B, 1]`` (or ``[B]``) to RED's perspective.

    Reads the side-to-move plane (channel 14, 1.0 = RED to move) of the Phase 2
    observation. Only for evaluation and logging; the network and PPO work in
    the side-to-move perspective.
    """
    from xiangqi_rl.env.observation import SIDE_TO_MOVE_CHANNEL

    red_to_move = observations[:, SIDE_TO_MOVE_CHANNEL].flatten(1).mean(dim=1)
    sign = torch.where(red_to_move > 0.5, 1.0, -1.0).to(value.dtype)
    return value * (sign.unsqueeze(-1) if value.dim() == 2 else sign)
