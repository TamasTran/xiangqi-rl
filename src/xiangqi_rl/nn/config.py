"""Model hyper-parameters (validated at construction)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from xiangqi_rl.engine import NUM_COLS, NUM_ROWS
from xiangqi_rl.env import ACTION_DIM, NUM_CHANNELS

ARCHITECTURE = "xiangqi-policy-value-resnet-v1"
"""Identifier stored in checkpoints; bump when the module structure changes."""

NormKind = Literal["group", "batch", "none"]
ValueActivation = Literal["tanh", "none"]


@dataclass(frozen=True)
class ModelConfig:
    """Configuration of :class:`~xiangqi_rl.nn.model.PolicyValueNet`.

    The input/output sizes default to the real Phase 2 interfaces
    (observation planes, board size, action space), so the defaults can never
    drift from the environment.

    Attributes:
        in_channels: Observation planes (Phase 2: 17).
        board_height / board_width: Board size (10 x 9).
        action_dim: Policy logits per position (8100 = 90 * 90).
        hidden_channels: Width of the shared backbone.
        num_res_blocks: Residual blocks in the backbone.
        norm: ``"group"`` (default), ``"batch"`` or ``"none"``. GroupNorm is
            independent of batch composition, so a position gives the same
            output whether it is evaluated alone (rollout) or inside a large
            minibatch (PPO update), and train/eval modes are identical.
        norm_groups: Groups for GroupNorm (must divide the channel counts).
        policy_channels: Channels of the 1x1 conv in the policy head.
        value_channels: Channels of the 1x1 conv in the value head.
        value_hidden: Hidden units of the value MLP.
        value_activation: ``"tanh"`` bounds the value to [-1, 1] (matching the
            +1/-1/0 rewards); ``"none"`` returns the raw scalar.
    """

    in_channels: int = NUM_CHANNELS
    board_height: int = NUM_ROWS
    board_width: int = NUM_COLS
    action_dim: int = ACTION_DIM
    hidden_channels: int = 64
    num_res_blocks: int = 4
    norm: NormKind = "group"
    norm_groups: int = 8
    policy_channels: int = 8
    value_channels: int = 8
    value_hidden: int = 64
    value_activation: ValueActivation = "tanh"

    def __post_init__(self) -> None:
        positive = (
            "in_channels", "board_height", "board_width", "action_dim",
            "hidden_channels", "policy_channels", "value_channels", "value_hidden",
            "norm_groups",
        )  # fmt: skip
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if isinstance(self.num_res_blocks, bool) or not isinstance(self.num_res_blocks, int) \
                or self.num_res_blocks < 0:
            raise ValueError(f"num_res_blocks must be a non-negative integer, got {self.num_res_blocks!r}")
        if self.norm not in ("group", "batch", "none"):
            raise ValueError(f"norm must be 'group', 'batch' or 'none', got {self.norm!r}")
        if self.value_activation not in ("tanh", "none"):
            raise ValueError(f"value_activation must be 'tanh' or 'none', got {self.value_activation!r}")
        if self.norm == "group":
            for name in ("hidden_channels", "policy_channels", "value_channels"):
                if getattr(self, name) % self.norm_groups:
                    raise ValueError(
                        f"{name}={getattr(self, name)} is not divisible by "
                        f"norm_groups={self.norm_groups}; set norm_groups to a divisor "
                        "(e.g. 1, 2, 4) or use norm='batch'/'none'"
                    )

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """``(in_channels, board_height, board_width)``."""
        return (self.in_channels, self.board_height, self.board_width)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form (JSON / checkpoint friendly)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Inverse of :meth:`to_dict`; unknown keys raise ``TypeError``."""
        return cls(**data)
