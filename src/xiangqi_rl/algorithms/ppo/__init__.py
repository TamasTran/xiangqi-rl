"""Proximal Policy Optimization implemented from scratch."""

from xiangqi_rl.algorithms.ppo.buffer import BufferStateError, Minibatch, RolloutBuffer
from xiangqi_rl.algorithms.ppo.config import (
    SMOKE_OVERRIDES,
    PPOConfig,
    load_config,
    parse_override,
)
from xiangqi_rl.algorithms.ppo.distributions import act, evaluate_actions, state_value
from xiangqi_rl.algorithms.ppo.losses import ppo_policy_loss, total_loss, value_loss
from xiangqi_rl.algorithms.ppo.opponent_env import OpponentEnv
from xiangqi_rl.algorithms.ppo.returns import (
    compute_gae,
    explained_variance,
    normalize_advantages,
)
from xiangqi_rl.algorithms.ppo.seeding import resolve_device, seed_everything
from xiangqi_rl.algorithms.ppo.shaping import material_balance
from xiangqi_rl.algorithms.ppo.trainer import NonFiniteError, PPOTrainer

__all__ = [
    "SMOKE_OVERRIDES",
    "BufferStateError",
    "Minibatch",
    "NonFiniteError",
    "OpponentEnv",
    "PPOConfig",
    "PPOTrainer",
    "RolloutBuffer",
    "act",
    "compute_gae",
    "evaluate_actions",
    "explained_variance",
    "load_config",
    "material_balance",
    "normalize_advantages",
    "parse_override",
    "ppo_policy_loss",
    "resolve_device",
    "seed_everything",
    "state_value",
    "total_loss",
    "value_loss",
]
