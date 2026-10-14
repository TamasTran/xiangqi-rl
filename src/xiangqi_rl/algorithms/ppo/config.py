"""PPO experiment configuration (every field is used by the trainer)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from xiangqi_rl.nn import ModelConfig

OPPONENT_NAMES = ("random", "heuristic", "lookahead")
COLOR_SCHEDULES = ("alternate", "red", "black", "random")
OPTIMIZERS = ("adam", "adamw", "sgd")
LR_SCHEDULES = ("none", "linear")
DEVICES = ("auto", "cpu", "cuda")


@dataclass(frozen=True)
class PPOConfig:
    """Hyper-parameters and run settings for :class:`PPOTrainer`.

    Units: a *decision* is one action of the trained agent (two plies once
    the opponent's reply is included). ``rollout_steps`` counts decisions.
    """

    # --- run
    seed: int = 0
    device: str = "auto"
    """``auto`` (CUDA if available, else CPU), ``cpu`` or ``cuda``."""
    deterministic: bool = False
    """Request deterministic torch algorithms (slower; see docs)."""
    output_dir: str = "artifacts/ppo"
    total_iterations: int = 100
    """Number of rollout+update cycles."""

    # --- model
    model: ModelConfig = field(default_factory=ModelConfig)

    # --- optimisation
    optimizer: str = "adam"
    learning_rate: float = 3e-4
    adam_eps: float = 1e-5
    weight_decay: float = 0.0
    lr_schedule: str = "linear"
    """``linear`` decays the learning rate to 0 over ``total_iterations``."""

    # --- PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_clip_coef: float | None = None
    """If set, PPO value clipping with this epsilon; ``None`` disables it."""
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    target_kl: float | None = None
    """Stop the update epochs early once the mean approx-KL exceeds this."""
    rollout_steps: int = 2048
    minibatch_size: int = 256
    update_epochs: int = 4

    # --- opponents / episodes
    opponents: tuple[str, ...] = ("random", "heuristic")
    """Training opponents; one is drawn uniformly at random per episode."""
    color_schedule: str = "alternate"
    """Trained side per episode: ``alternate`` (RED, BLACK, ...), ``red``,
    ``black`` or ``random``."""
    game_max_plies: int | None = 200
    """Engine draw rule (a natural, terminal result)."""
    max_episode_plies: int | None = None
    """External step limit in plies (truncation, bootstrapped)."""
    material_shaping_coef: float = 0.0
    """Potential-based material shaping ``gamma*Phi(s') - Phi(s)`` with
    ``Phi = coef * material_balance`` (0 disables). Requires
    ``model.value_activation='none'`` because shaped values can exceed [-1, 1]."""

    # --- evaluation / logging / checkpoints
    eval_interval: int = 10
    """Evaluate every N iterations (0 disables, including the initial one)."""
    eval_games: int = 20
    """Games per opponent (even, so both colours are equally represented)."""
    eval_seed: int = 12345
    eval_opponents: tuple[str, ...] = ("random", "heuristic")
    eval_greedy: bool = True
    """Evaluate with the most probable legal move instead of sampling."""
    checkpoint_interval: int = 10
    log_interval: int = 1
    """Print a console line every N iterations (the CSV has every one)."""

    def __post_init__(self) -> None:
        def check(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(msg)

        check(self.device in DEVICES, f"device must be one of {DEVICES}, got {self.device!r}")
        check(self.optimizer in OPTIMIZERS, f"optimizer must be one of {OPTIMIZERS}")
        check(self.lr_schedule in LR_SCHEDULES, f"lr_schedule must be one of {LR_SCHEDULES}")
        check(self.color_schedule in COLOR_SCHEDULES,
              f"color_schedule must be one of {COLOR_SCHEDULES}")
        check(self.model.norm != "batch",
              "norm='batch' is not supported by PPO (rollout and update would use "
              "different statistics); use 'group' or 'none'")
        check(self.material_shaping_coef >= 0, "material_shaping_coef must be >= 0")
        check(self.material_shaping_coef == 0 or self.model.value_activation == "none",
              "material_shaping_coef > 0 needs model.value_activation='none': shaped "
              "returns range over [-1 - coef, 1 + coef], outside tanh's [-1, 1]")
        check(self.total_iterations >= 0, "total_iterations must be >= 0")
        check(self.learning_rate > 0, "learning_rate must be positive")
        check(0.0 < self.gamma <= 1.0, "gamma must be in (0, 1]")
        check(0.0 <= self.gae_lambda <= 1.0, "gae_lambda must be in [0, 1]")
        check(self.clip_coef > 0, "clip_coef must be positive")
        check(self.value_clip_coef is None or self.value_clip_coef > 0,
              "value_clip_coef must be positive or None")
        check(self.vf_coef >= 0 and self.ent_coef >= 0, "vf_coef and ent_coef must be >= 0")
        check(self.max_grad_norm > 0, "max_grad_norm must be positive")
        check(self.target_kl is None or self.target_kl > 0, "target_kl must be positive or None")
        check(self.rollout_steps >= 1, "rollout_steps must be >= 1")
        check(1 <= self.minibatch_size <= self.rollout_steps,
              "minibatch_size must be in [1, rollout_steps]")
        check(self.update_epochs >= 1, "update_epochs must be >= 1")
        check(len(self.opponents) >= 1 and set(self.opponents) <= set(OPPONENT_NAMES),
              f"opponents must be a non-empty subset of {OPPONENT_NAMES}")
        check(self.game_max_plies is None or self.game_max_plies >= 1, "game_max_plies must be >= 1")
        check(self.max_episode_plies is None or self.max_episode_plies >= 1,
              "max_episode_plies must be >= 1")
        check(self.eval_interval >= 0 and self.checkpoint_interval >= 1 and self.log_interval >= 1,
              "eval_interval must be >= 0; checkpoint_interval and log_interval >= 1")
        check(self.eval_games >= 2 and self.eval_games % 2 == 0,
              "eval_games must be an even number >= 2")
        check(len(self.eval_opponents) >= 1 and set(self.eval_opponents) <= set(OPPONENT_NAMES),
              f"eval_opponents must be a non-empty subset of {OPPONENT_NAMES}")

    # ------------------------------------------------------------------ (de)serialisation

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form (lists instead of tuples, nested model dict)."""
        data = asdict(self)
        data["opponents"] = list(self.opponents)
        data["eval_opponents"] = list(self.eval_opponents)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PPOConfig:
        """Build from a dict; unknown keys raise ``TypeError``."""
        data = dict(data)
        if "model" in data:
            data["model"] = ModelConfig.from_dict(dict(data["model"]))
        for key in ("opponents", "eval_opponents"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

    def with_overrides(self, overrides: dict[str, Any]) -> PPOConfig:
        """Return a copy with ``overrides`` applied; ``model.<field>`` keys
        update the nested model config."""
        top: dict[str, Any] = {}
        model: dict[str, Any] = {}
        for key, value in overrides.items():
            if key.startswith("model."):
                model[key.removeprefix("model.")] = value
            else:
                top[key] = value
        known = set(self.to_dict())
        unknown = sorted(set(top) - known)
        if unknown:
            raise TypeError(f"Unknown config keys: {unknown}")
        for key in ("opponents", "eval_opponents"):
            if key in top:
                top[key] = tuple(top[key])
        if model:
            top["model"] = ModelConfig.from_dict({**self.model.to_dict(), **model})
        return replace(self, **top)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> PPOConfig:
    """Load a YAML config file (and apply optional overrides)."""
    import yaml

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    config = PPOConfig.from_dict(data)
    return config.with_overrides(overrides) if overrides else config


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``key=value`` (value read as YAML: ``3``, ``true``, ``null``, ``[a, b]``)."""
    import yaml

    if "=" not in text:
        raise ValueError(f"Override must look like key=value, got {text!r}")
    key, value = text.split("=", 1)
    return key.strip(), yaml.safe_load(value)


SMOKE_OVERRIDES: dict[str, Any] = {
    "model.hidden_channels": 16,
    "model.num_res_blocks": 1,
    "model.norm_groups": 4,
    "model.policy_channels": 4,
    "model.value_channels": 4,
    "model.value_hidden": 16,
    "rollout_steps": 64,
    "minibatch_size": 32,
    "update_epochs": 2,
    "total_iterations": 2,
    "game_max_plies": 40,
    "eval_interval": 1,
    "eval_games": 2,
    "checkpoint_interval": 1,
}
"""Small, fast settings used by ``train_ppo.py --smoke-test``."""
