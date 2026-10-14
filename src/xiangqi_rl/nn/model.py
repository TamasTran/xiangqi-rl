"""Policy-value network: shared residual CNN with a policy and a value head.

::

    obs [B, C, 10, 9]
      -> Conv3x3(C -> H) -> Norm -> ReLU
      -> ResidualBlock x N          (Conv3x3 -> Norm -> ReLU -> Conv3x3 -> Norm, +skip, ReLU)
      -> features [B, H, 10, 9]
           |-- policy head: Conv1x1 -> Norm -> ReLU -> flatten -> Linear -> logits [B, A]
           `-- value head : Conv1x1 -> Norm -> ReLU -> flatten -> Linear -> ReLU
                                                          -> Linear -> tanh -> value [B, 1]

Outputs
-------
* ``policy_logits``: raw, unmasked scores for every action in the flat
  ``90 * from_sq + to_sq`` space. Legality is *not* known to the network.
* ``value``: expected game outcome in [-1, 1] **from the perspective of the
  side to move** in the input position (+1: side to move wins, -1: loses,
  0: draw). This is the same perspective as the sign-flipped Phase 2 rewards;
  see ``docs/policy_value_network.md``.

An untrained instance is just a randomly initialised function approximator.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from xiangqi_rl.nn.config import ModelConfig


def _norm(kind: str, channels: int, groups: int) -> nn.Module:
    if kind == "group":
        return nn.GroupNorm(groups, channels)
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    return nn.Identity()


class ResidualBlock(nn.Module):
    """``x -> Conv -> Norm -> ReLU -> Conv -> Norm -> (+x) -> ReLU``."""

    def __init__(self, channels: int, norm: str, groups: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=norm == "none")
        self.norm1 = _norm(norm, channels, groups)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=norm == "none")
        self.norm2 = _norm(norm, channels, groups)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + x)


class PolicyValueNet(nn.Module):
    """Shared-backbone policy-value network.

    ``forward(observations)`` takes a float tensor ``[B, C, H, W]`` matching
    ``config.observation_shape`` and returns ``(policy_logits [B, action_dim],
    value [B, 1])``. Use :mod:`xiangqi_rl.nn.policy` to apply legal-action
    masks to the logits.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else ModelConfig()
        cfg = self.config
        bias = cfg.norm == "none"
        cells = cfg.board_height * cfg.board_width

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, cfg.hidden_channels, 3, padding=1, bias=bias),
            _norm(cfg.norm, cfg.hidden_channels, cfg.norm_groups),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *(
                ResidualBlock(cfg.hidden_channels, cfg.norm, cfg.norm_groups)
                for _ in range(cfg.num_res_blocks)
            )
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(cfg.hidden_channels, cfg.policy_channels, 1, bias=bias),
            _norm(cfg.norm, cfg.policy_channels, cfg.norm_groups),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(cfg.policy_channels * cells, cfg.action_dim),
        )
        value_layers: list[nn.Module] = [
            nn.Conv2d(cfg.hidden_channels, cfg.value_channels, 1, bias=bias),
            _norm(cfg.norm, cfg.value_channels, cfg.norm_groups),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(cfg.value_channels * cells, cfg.value_hidden),
            nn.ReLU(),
            nn.Linear(cfg.value_hidden, 1),
        ]
        if cfg.value_activation == "tanh":
            value_layers.append(nn.Tanh())
        self.value_head = nn.Sequential(*value_layers)

        # Small final layers: the untrained policy starts near uniform and the
        # untrained value near 0, instead of arbitrary large outputs.
        policy_out = self.policy_head[-1]
        value_out = [m for m in self.value_head if isinstance(m, nn.Linear)][-1]
        nn.init.normal_(policy_out.weight, std=0.01)
        nn.init.zeros_(policy_out.bias)
        nn.init.normal_(value_out.weight, std=0.01)
        nn.init.zeros_(value_out.bias)

    def features(self, observations: Tensor) -> Tensor:
        """Shared feature map ``[B, hidden_channels, H, W]``."""
        self._check_input(observations)
        return self.blocks(self.stem(observations))

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(policy_logits [B, action_dim], value [B, 1])``."""
        feats = self.features(observations)
        return self.policy_head(feats), self.value_head(feats)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Number of (trainable) parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def _check_input(self, observations: Tensor) -> None:
        if not isinstance(observations, Tensor):
            raise TypeError(f"observations must be a torch.Tensor, got {type(observations).__name__}")
        expected = self.config.observation_shape
        if observations.dim() != 4 or tuple(observations.shape[1:]) != expected:
            raise ValueError(
                f"observations must have shape [batch, {expected[0]}, {expected[1]}, "
                f"{expected[2]}], got {list(observations.shape)}"
            )
        if not observations.is_floating_point():
            raise TypeError(f"observations must be floating point, got {observations.dtype}")


def create_model(config: ModelConfig | None = None, seed: int | None = None) -> PolicyValueNet:
    """Build a model; with ``seed`` the initial weights are reproducible and the
    global torch RNG state is left untouched."""
    if seed is None:
        return PolicyValueNet(config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return PolicyValueNet(config)
