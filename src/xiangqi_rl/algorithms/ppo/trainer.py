"""PPO trainer: rollout collection, update loop, evaluation, checkpoints.

Data flow per iteration::

    OpponentEnv -> (obs, legal mask) -> model -> masked policy -> action
        -> RolloutBuffer -> GAE -> PPO epochs/minibatches -> updated model

The trainer owns one :class:`OpponentEnv`, one :class:`RolloutBuffer` and the
optimizer. The behaviour policy is frozen during a rollout (the model is only
modified inside :meth:`update`).
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from xiangqi_rl.algorithms.ppo.buffer import RolloutBuffer
from xiangqi_rl.algorithms.ppo.config import PPOConfig
from xiangqi_rl.algorithms.ppo.distributions import act_batch, evaluate_actions, values_batch
from xiangqi_rl.algorithms.ppo.logger import CSVLogger
from xiangqi_rl.algorithms.ppo.losses import ppo_policy_loss, total_loss, value_loss
from xiangqi_rl.algorithms.ppo.returns import explained_variance
from xiangqi_rl.algorithms.ppo.seeding import resolve_device, seed_everything
from xiangqi_rl.algorithms.ppo.vec_env import EnvSpec, VecEnv, make_vec_env
from xiangqi_rl.evaluation.ppo_evaluator import EvalResult, evaluate_against_baselines
from xiangqi_rl.nn import PolicyValueNet, create_model, load_checkpoint, save_checkpoint

TRAIN_COLUMNS = [
    "iteration", "decisions", "episodes", "policy_loss", "value_loss", "entropy",
    "approx_kl", "clip_fraction", "mean_advantage", "mean_return", "explained_variance",
    "grad_norm", "learning_rate", "epochs_run", "episode_reward", "episode_shaped_return",
    "episode_length",
    "wins", "draws", "losses", "truncated", "episodes_in_rollout",
    "env_seconds", "inference_seconds", "update_seconds", "wall_seconds",
]
EVAL_COLUMNS = [
    "iteration", "opponent", "games", "wins", "draws", "losses", "truncated", "win_rate",
    "draw_rate", "score", "average_plies", "invalid_actions",
    "red_games", "red_wins", "red_draws", "red_losses",
    "black_games", "black_wins", "black_draws", "black_losses", "checkpoint",
    # how games ended, from the policy's point of view
    "win_checkmate", "win_stalemate", "win_perpetual_check", "win_other",
    "loss_checkmate", "loss_stalemate", "loss_perpetual_check", "loss_other",
    "draw_repetition", "draw_max_plies", "draw_other", "truncated_external",
]


_REASON_COLUMNS = (
    "win_checkmate", "win_stalemate", "win_perpetual_check", "win_other",
    "loss_checkmate", "loss_stalemate", "loss_perpetual_check", "loss_other",
    "draw_repetition", "draw_max_plies", "draw_other", "truncated_external",
)


def reason_columns(reasons: dict[str, int]) -> dict[str, int]:
    """Map ``"<outcome>_<reason>"`` counts onto the fixed eval-log columns;
    reasons without their own column are summed into ``<outcome>_other``."""
    out = dict.fromkeys(_REASON_COLUMNS, 0)
    for key, count in reasons.items():
        if key.startswith("truncated_"):
            out["truncated_external"] += count
        elif key in out:
            out[key] += count
        else:
            out[key.split("_", 1)[0] + "_other"] += count
    return out


class NonFiniteError(FloatingPointError):
    """A loss, gradient or metric was NaN/inf; training stopped before it
    could corrupt the model."""


@dataclass
class EpisodeStats:
    """One finished training episode (from the trained agent's perspective)."""

    reward: float
    """Game result reward (+1/0/-1; 0 when truncated), without shaping."""
    shaped_return: float
    """Undiscounted sum of the rewards used for training (game + shaping)."""
    decisions: int
    plies: int
    outcome: str  # "win" | "draw" | "loss" | "truncated"
    agent_side: int
    opponent: str


class PPOTrainer:
    """Train a :class:`PolicyValueNet` with PPO against baseline opponents."""

    def __init__(
        self,
        config: PPOConfig,
        model: PolicyValueNet | None = None,
        *,
        resuming: bool = False,
    ) -> None:
        """Create a trainer.

        A *new* run refuses an ``output_dir`` that already holds a run (logs or
        checkpoints): continuing would append to the old CSV logs and overwrite
        its ``latest.pt``/``best.pt``. Resume with :meth:`from_checkpoint`
        (which passes ``resuming=True``) or pick a new directory.
        """
        existing = [
            name for name in ("train_log.csv", "eval_log.csv", "latest.pt", "best.pt")
            if (Path(config.output_dir) / name).exists()
        ]
        if existing and not resuming:
            raise FileExistsError(
                f"{config.output_dir} already contains a run ({', '.join(existing)}). "
                "Resume it (--resume <dir>/latest.pt) or use a new output_dir."
            )
        self.config = config
        self.device = resolve_device(config.device)
        seed_everything(config.seed, config.deterministic)

        self.model = (model if model is not None else create_model(config.model, seed=config.seed))
        if self.model.config != config.model:
            raise ValueError("model.config does not match config.model")
        self.model.to(self.device)
        self.optimizer = self._make_optimizer()
        total = max(config.total_iterations, 1)
        if config.lr_schedule == "linear":
            self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
                torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lambda it: max(0.0, 1.0 - it / total)
                )
            )
        else:
            self.scheduler = None

        self.envs: VecEnv = make_vec_env(self._env_specs(), self._resolve_workers())
        self.buffer = RolloutBuffer(config.rollout_steps, self.envs.observation_shape,
                                    self.envs.action_dim, num_envs=config.num_envs)
        self._np_rng = np.random.default_rng(config.seed)

        self.iteration = 0
        self.total_decisions = 0
        self.total_episodes = 0
        self.best_eval_score = -math.inf
        self._wall_seconds = 0.0

        # Current decision point of every environment, and per-episode
        # counters carried across rollout boundaries.
        self._obs: np.ndarray | None = None
        self._masks: np.ndarray | None = None
        self._episode_decisions = np.zeros(config.num_envs, dtype=np.int64)
        self._episode_shaped_return = np.zeros(config.num_envs, dtype=np.float64)

        out = Path(config.output_dir)
        self.output_dir = out
        self.train_log = CSVLogger(out / "train_log.csv", TRAIN_COLUMNS)
        self.eval_log = CSVLogger(out / "eval_log.csv", EVAL_COLUMNS)

    # ------------------------------------------------------------------ setup

    def _env_specs(self) -> list[EnvSpec]:
        """One spec per environment. Environment 0 uses the same seeds as the
        original single-environment trainer; colours are staggered."""
        c = self.config
        specs = []
        for e in range(c.num_envs):
            base = c.seed + 7919 * e
            specs.append(EnvSpec(
                opponents=tuple(c.opponents),
                opponent_seeds=tuple(base + 1 + i for i in range(len(c.opponents))),
                seed=base,
                game_max_plies=c.game_max_plies,
                max_episode_plies=c.max_episode_plies,
                color_schedule=c.color_schedule,
                material_shaping_coef=c.material_shaping_coef,
                shaping_gamma=c.gamma,
                color_offset=e,
            ))
        return specs

    def _resolve_workers(self) -> int:
        workers = self.config.env_workers
        if workers == -1:
            # One core stays free for the trainer process (batched inference,
            # PPO updates); measured: more workers than that only competed for
            # the CPU with inference.
            workers = min(self.config.num_envs, max(1, (os.cpu_count() or 2) - 1))
        return workers

    def close(self) -> None:
        """Stop environment worker processes (if any)."""
        self.envs.close()

    def _make_optimizer(self) -> torch.optim.Optimizer:
        c, params = self.config, self.model.parameters()
        if c.optimizer == "adam":
            return torch.optim.Adam(params, lr=c.learning_rate, eps=c.adam_eps,
                                    weight_decay=c.weight_decay)
        if c.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=c.learning_rate, eps=c.adam_eps,
                                     weight_decay=c.weight_decay)
        return torch.optim.SGD(params, lr=c.learning_rate, momentum=0.9,
                               weight_decay=c.weight_decay)

    # ------------------------------------------------------------------ rollout

    def collect_rollout(self) -> tuple[list[EpisodeStats], dict[str, float]]:
        """Fill the buffer with ``rollout_steps`` decisions of the *current*
        policy (no gradients) and compute GAE. Returns finished-episode stats
        and timing information.

        All ``num_envs`` games advance together: one batched forward pass per
        step chooses every game's move, then the games are stepped (in worker
        processes when ``env_workers`` > 0).
        """
        cfg = self.config
        self.buffer.reset()
        self.model.eval()
        episodes: list[EpisodeStats] = []
        env_time = inf_time = 0.0

        if self._obs is None:
            t0 = time.perf_counter()
            self._obs, self._masks = self.envs.reset()
            env_time += time.perf_counter() - t0
            self._episode_decisions[:] = 0
            self._episode_shaped_return[:] = 0.0

        for _ in range(cfg.rollout_steps // cfg.num_envs):
            t0 = time.perf_counter()
            actions, log_probs, values = act_batch(self.model, self._obs, self._masks, self.device)
            inf_time += time.perf_counter() - t0

            t0 = time.perf_counter()
            next_obs, next_masks, rewards, terminated, truncated, infos, finals = (
                self.envs.step(actions))
            env_time += time.perf_counter() - t0
            self._episode_decisions += 1
            self._episode_shaped_return += rewards

            truncation_values = np.zeros(cfg.num_envs, dtype=np.float32)
            cut = np.flatnonzero(truncated)
            if len(cut):
                # Bootstrap from the final observation, expressed from the
                # trained agent's perspective (the side to move may be the
                # opponent there, which flips the sign of the value).
                t0 = time.perf_counter()
                v = values_batch(self.model, np.stack([finals[i] for i in cut]), self.device)
                for j, i in enumerate(cut):
                    sign = 1.0 if infos[i]["side_to_move"] == infos[i]["agent_side"] else -1.0
                    truncation_values[i] = sign * v[j]
                inf_time += time.perf_counter() - t0

            self.buffer.add_batch(self._obs, self._masks, actions, log_probs, values, rewards,
                                  terminated, truncated, truncation_values)

            for i in np.flatnonzero(terminated | truncated):
                info = infos[i]
                # Outcome from the *game* reward, never from the (shaped) training reward.
                game_reward = info["game_reward"]
                outcome = "truncated" if truncated[i] else (
                    "win" if game_reward > 0 else "loss" if game_reward < 0 else "draw")
                episodes.append(EpisodeStats(
                    game_reward, float(self._episode_shaped_return[i]),
                    int(self._episode_decisions[i]), info["ply"], outcome,
                    info["agent_side"], info["opponent"]))
                self._episode_decisions[i] = 0
                self._episode_shaped_return[i] = 0.0
            self._obs, self._masks = next_obs, next_masks

        # The current observations are live decision points of the agent.
        t0 = time.perf_counter()
        last_values = values_batch(self.model, self._obs, self.device)
        inf_time += time.perf_counter() - t0
        self.buffer.compute_advantages(last_values, cfg.gamma, cfg.gae_lambda,
                                       normalize=cfg.normalize_advantages)
        self.total_decisions += cfg.rollout_steps
        self.total_episodes += len(episodes)
        return episodes, {"env_seconds": env_time, "inference_seconds": inf_time}

    # ------------------------------------------------------------------ update

    def update(self) -> dict[str, float]:
        """Run the PPO epochs over the (fresh, full) buffer.

        Raises:
            NonFiniteError: before any optimizer step with a non-finite loss
                or gradient norm.
        """
        cfg = self.config
        buf = self.buffer
        self.model.train()
        sums = {k: 0.0 for k in
                ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm")}
        n_batches = 0
        epochs_run = 0
        params = [p for p in self.model.parameters() if p.requires_grad]

        for _ in range(cfg.update_epochs):
            epoch_kl = 0.0
            epoch_batches = 0
            for mb in buf.minibatches(cfg.minibatch_size, self._np_rng):
                obs = torch.from_numpy(mb.observations).to(self.device)
                masks = torch.from_numpy(mb.masks).to(self.device)  # the rollout-time masks
                actions = torch.from_numpy(mb.actions).to(self.device)
                old_logp = torch.from_numpy(mb.old_log_probs).to(self.device)
                old_values = torch.from_numpy(mb.old_values).to(self.device)
                advantages = torch.from_numpy(mb.advantages).to(self.device)
                returns = torch.from_numpy(mb.returns).to(self.device)

                new_logp, entropy, new_values = evaluate_actions(self.model, obs, masks, actions)
                pol = ppo_policy_loss(new_logp, old_logp, advantages, cfg.clip_coef)
                v_loss = value_loss(new_values, old_values, returns, cfg.value_clip_coef)
                ent = entropy.mean()
                loss = total_loss(pol.loss, v_loss, ent, cfg.vf_coef, cfg.ent_coef)
                if not torch.isfinite(loss):
                    raise NonFiniteError(
                        f"non-finite loss at iteration {self.iteration}: total={loss.item()} "
                        f"policy={pol.loss.item()} value={v_loss.item()} entropy={ent.item()}"
                    )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    raise NonFiniteError(
                        f"non-finite gradient norm at iteration {self.iteration}: "
                        f"{grad_norm.item()} (loss {loss.item()})"
                    )
                self.optimizer.step()

                sums["policy_loss"] += pol.loss.item()
                sums["value_loss"] += v_loss.item()
                sums["entropy"] += ent.item()
                kl = pol.approx_kl.item()
                sums["approx_kl"] += kl
                sums["clip_fraction"] += pol.clip_fraction.item()
                sums["grad_norm"] += grad_norm.item()
                epoch_kl += kl
                epoch_batches += 1
                n_batches += 1
            epochs_run += 1
            if cfg.target_kl is not None and epoch_kl / epoch_batches > cfg.target_kl:
                break

        metrics = {k: v / n_batches for k, v in sums.items()}
        metrics["epochs_run"] = float(epochs_run)
        metrics["mean_advantage"] = float(buf.raw_advantages.mean())
        metrics["mean_return"] = float(buf.returns.mean())
        metrics["explained_variance"] = explained_variance(buf.values, buf.returns)
        for key, value in metrics.items():
            if not (math.isfinite(value) or key == "explained_variance"):
                raise NonFiniteError(f"metric {key} is not finite: {value}")
        return metrics

    # ------------------------------------------------------------------ evaluation

    def evaluate(self, record: bool = True) -> dict[str, EvalResult]:
        """Evaluate vs. the baselines (separate games, no data collected)."""
        cfg = self.config
        results = evaluate_against_baselines(
            self.model, cfg.eval_opponents, cfg.eval_games, cfg.eval_seed, self.device,
            greedy=cfg.eval_greedy, game_max_plies=cfg.game_max_plies,
            max_episode_plies=cfg.max_episode_plies,
        )
        if record:
            for name, r in results.items():
                row: dict[str, Any] = {
                    "iteration": self.iteration, "opponent": name, "games": r.games,
                    "wins": r.wins, "draws": r.draws, "losses": r.losses,
                    "truncated": r.truncated, "win_rate": r.win_rate, "draw_rate": r.draw_rate,
                    "score": r.score, "average_plies": r.average_plies,
                    "invalid_actions": r.invalid_actions,
                    "checkpoint": str(self.output_dir / "latest.pt"),
                }
                for color, c in r.by_color.items():
                    row.update({f"{color}_games": c.games, f"{color}_wins": c.wins,
                                f"{color}_draws": c.draws, f"{color}_losses": c.losses})
                row.update(reason_columns(r.reasons))
                self.eval_log.log(row)
        return results

    @staticmethod
    def eval_score(results: dict[str, EvalResult]) -> float:
        """Mean score over opponents (used to pick the best checkpoint)."""
        return float(np.mean([r.score for r in results.values()]))

    # ------------------------------------------------------------------ main loop

    def train(self) -> list[dict[str, float]]:
        """Run until ``config.total_iterations``; returns per-iteration metrics."""
        cfg = self.config
        history: list[dict[str, float]] = []
        if cfg.eval_interval > 0 and self.iteration == 0:
            self._evaluate_and_maybe_save_best()
        while self.iteration < cfg.total_iterations:
            start = time.perf_counter()
            episodes, timing = self.collect_rollout()
            lr_used = self.optimizer.param_groups[0]["lr"]
            t0 = time.perf_counter()
            metrics = self.update()
            timing["update_seconds"] = time.perf_counter() - t0
            if self.scheduler is not None:
                self.scheduler.step()
            self.iteration += 1
            self._wall_seconds += time.perf_counter() - start

            metrics.update(timing)
            metrics.update(self._episode_summary(episodes))
            metrics.update(
                iteration=self.iteration, decisions=self.total_decisions,
                episodes=self.total_episodes,
                learning_rate=lr_used,
                wall_seconds=self._wall_seconds,
            )
            self.train_log.log(metrics)
            history.append(metrics)
            if self.iteration % cfg.log_interval == 0:
                print(self._format(metrics), flush=True)

            if cfg.eval_interval > 0 and (
                self.iteration % cfg.eval_interval == 0 or self.iteration == cfg.total_iterations
            ):
                self._evaluate_and_maybe_save_best()
            if self.iteration % cfg.checkpoint_interval == 0 or self.iteration == cfg.total_iterations:
                self.save(self.output_dir / "latest.pt")
        return history

    def _evaluate_and_maybe_save_best(self) -> None:
        results = self.evaluate()
        score = self.eval_score(results)
        summary = ", ".join(f"{n}: W{r.wins}/D{r.draws}/L{r.losses}" for n, r in results.items())
        print(f"[eval @ iter {self.iteration}] score={score:.3f} ({summary})", flush=True)
        if score > self.best_eval_score:
            self.best_eval_score = score
            self.save(self.output_dir / "best.pt")

    @staticmethod
    def _episode_summary(episodes: list[EpisodeStats]) -> dict[str, float]:
        n = len(episodes)
        count = lambda o: float(sum(e.outcome == o for e in episodes))  # noqa: E731
        return {
            "episodes_in_rollout": float(n),
            "episode_reward": float(np.mean([e.reward for e in episodes])) if n else float("nan"),
            "episode_shaped_return": (
                float(np.mean([e.shaped_return for e in episodes])) if n else float("nan")),
            "episode_length": float(np.mean([e.decisions for e in episodes])) if n else float("nan"),
            "wins": count("win"), "draws": count("draw"),
            "losses": count("loss"), "truncated": count("truncated"),
        }

    @staticmethod
    def _format(m: dict[str, float]) -> str:
        return (
            f"iter {int(m['iteration']):4d} | decisions {int(m['decisions']):7d} | "
            f"R {m['episode_reward']:+.2f} (shaped {m['episode_shaped_return']:+.2f}) len {m['episode_length']:.0f} "
            f"W{int(m['wins'])}/D{int(m['draws'])}/L{int(m['losses'])} | "
            f"pi {m['policy_loss']:+.4f} v {m['value_loss']:.4f} H {m['entropy']:.3f} "
            f"kl {m['approx_kl']:.4f} clip {m['clip_fraction']:.2f} ev {m['explained_variance']:.2f} | "
            f"env {m['env_seconds']:.1f}s inf {m['inference_seconds']:.1f}s "
            f"upd {m['update_seconds']:.1f}s"
        )

    # ------------------------------------------------------------------ checkpoints

    def _training_state(self) -> dict[str, Any]:
        return {
            "kind": "ppo_training",
            "ppo_config": self.config.to_dict(),
            "iteration": self.iteration,
            "total_decisions": self.total_decisions,
            "total_episodes": self.total_episodes,
            "best_eval_score": None if self.best_eval_score == -math.inf else self.best_eval_score,
            "wall_seconds": self._wall_seconds,
            "envs": self.envs.get_state(),
            "scheduler_state_dict": None if self.scheduler is None else self.scheduler.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy_global": _numpy_global_state(),
                "numpy_trainer": self._np_rng.bit_generator.state,
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        }

    def save(self, path: str | Path) -> None:
        """Write a resumable checkpoint (also loadable by ``nn.load_model``)."""
        save_checkpoint(path, self.model, optimizer=self.optimizer, extra=self._training_state())

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        config: PPOConfig | None = None,
        device: str | None = None,
    ) -> PPOTrainer:
        """Rebuild a trainer and restore model, optimizer, scheduler, counters
        and RNG states. A fresh episode is started (a half-finished episode is
        not stored); rollouts are always collected fresh anyway. If
        ``num_envs`` changed, the saved environment states are applied to the
        first environments and the others start from their own seeds.

        ``config`` may change run settings (e.g. ``total_iterations``) but its
        model config must match the checkpoint.
        """
        payload = load_checkpoint(path, map_location="cpu")
        state = payload.get("extra", {})
        if state.get("kind") != "ppo_training" or "optimizer_state_dict" not in payload:
            raise ValueError(f"{path} is not a resumable PPO training checkpoint")
        cfg = config if config is not None else PPOConfig.from_dict(state["ppo_config"])
        if device is not None:
            cfg = cfg.with_overrides({"device": device})
        if cfg.model.to_dict() != payload["config"]:
            raise ValueError("config.model does not match the checkpoint's model config")

        trainer = cls(cfg, resuming=True)
        trainer.model.load_state_dict(payload["model_state_dict"])
        trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if trainer.scheduler is not None and state.get("scheduler_state_dict") is not None:
            trainer.scheduler.load_state_dict(state["scheduler_state_dict"])
        trainer.iteration = int(state["iteration"])
        if trainer.scheduler is not None:
            # The restored optimizer holds the lr of the *old* schedule; if
            # total_iterations/learning_rate changed, recompute it for the new
            # schedule at the saved iteration.
            trainer.scheduler.base_lrs = [cfg.learning_rate] * len(trainer.optimizer.param_groups)
            factor = max(0.0, 1.0 - trainer.iteration / max(cfg.total_iterations, 1))
            for group in trainer.optimizer.param_groups:
                group["lr"] = cfg.learning_rate * factor
        else:
            for group in trainer.optimizer.param_groups:
                group["lr"] = cfg.learning_rate
        trainer.total_decisions = int(state["total_decisions"])
        trainer.total_episodes = int(state["total_episodes"])
        best = state["best_eval_score"]
        trainer.best_eval_score = -math.inf if best is None else float(best)
        trainer._wall_seconds = float(state["wall_seconds"])

        rng = state["rng"]
        random.setstate(_to_tuple(rng["python"]))
        _set_numpy_global_state(rng["numpy_global"])
        trainer._np_rng.bit_generator.state = rng["numpy_trainer"]
        torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and len(rng["torch_cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([t.cpu() for t in rng["torch_cuda"]])
        if "envs" in state:
            env_states = state["envs"]
        else:  # single-environment checkpoint written before num_envs existed
            env_states = [{"rng": rng["env"], "episodes_started": state["episodes_started"],
                           "opponents": rng["opponents"]}]
        trainer.envs.set_state([
            {"rng": _to_tuple(s["rng"]), "episodes_started": int(s["episodes_started"]),
             "opponents": [_to_tuple(o) for o in s["opponents"]]}
            for s in env_states
        ])
        return trainer


def _numpy_global_state() -> list:
    """NumPy's legacy global RNG state as plain Python values (so it can be
    stored in a ``weights_only`` checkpoint)."""
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return [name, keys.tolist(), int(pos), int(has_gauss), float(cached)]


def _set_numpy_global_state(state: list) -> None:
    name, keys, pos, has_gauss, cached = state
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))


def _to_tuple(value: Any) -> Any:
    """Recursively turn lists back into tuples (RNG states need tuples)."""
    if isinstance(value, (list, tuple)):
        return tuple(_to_tuple(v) for v in value)
    return value
