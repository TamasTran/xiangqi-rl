from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xiangqi_rl.algorithms.ppo import (  # noqa: E402
    SMOKE_OVERRIDES,
    NonFiniteError,
    PPOConfig,
    PPOTrainer,
    evaluate_actions,
    load_config,
    parse_override,
)
from xiangqi_rl.nn import load_model  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def small_config(tmp_path, **extra) -> PPOConfig:
    overrides = {**SMOKE_OVERRIDES, "device": "cpu", "output_dir": str(tmp_path / "run"),
                 "eval_interval": 0, "total_iterations": 2, **extra}
    return PPOConfig().with_overrides(overrides)


def params(model) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


# ----------------------------------------------------------------------------- config


def test_baseline_yaml_loads_and_matches_dataclass_fields() -> None:
    cfg = load_config(REPO / "configs" / "ppo_baseline.yaml")
    assert cfg.rollout_steps == 2048 and cfg.model.norm == "group"
    assert set(cfg.to_dict()) == set(PPOConfig().to_dict())  # no ignored / missing keys


def test_config_overrides_and_validation(tmp_path) -> None:
    cfg = PPOConfig().with_overrides({"model.hidden_channels": 32, "model.norm_groups": 8,
                                      "rollout_steps": 128, "minibatch_size": 32})
    assert cfg.model.hidden_channels == 32 and cfg.rollout_steps == 128
    assert parse_override("opponents=[random]") == ("opponents", ["random"])
    assert parse_override("target_kl=null") == ("target_kl", None)
    for bad in (
        {"minibatch_size": 4096},       # larger than rollout
        {"gamma": 1.5}, {"gae_lambda": -0.1}, {"clip_coef": 0.0}, {"learning_rate": 0.0},
        {"update_epochs": 0}, {"opponents": ["gpt"]}, {"eval_games": 3}, {"device": "tpu"},
        {"color_schedule": "blue"}, {"model.norm": "batch"}, {"max_grad_norm": 0.0},
    ):
        with pytest.raises(ValueError):
            PPOConfig().with_overrides(bad)
    with pytest.raises(TypeError):
        PPOConfig().with_overrides({"not_a_key": 1})
    (tmp_path / "bad.yaml").write_text("bogus: 1\n")
    with pytest.raises(TypeError):
        load_config(tmp_path / "bad.yaml")


# ----------------------------------------------------------------------------- rollout


def test_rollout_is_legal_finite_and_matches_update_time_log_probs(tmp_path) -> None:
    t = PPOTrainer(small_config(tmp_path))
    before = params(t.model)
    episodes, timing = t.collect_rollout()
    buf = t.buffer
    assert buf.is_full and len(buf) == 64
    assert buf.masks[np.arange(64), buf.actions].all()          # never an illegal action
    assert np.isfinite(buf.log_probs).all() and (buf.log_probs <= 0).all()
    assert np.isfinite(buf.values).all() and np.isfinite(buf.advantages).all()
    assert all(a.grad is None for a in t.model.parameters())    # no graph kept / no grads
    assert all(torch.equal(a, b) for a, b in zip(before, t.model.parameters()))  # policy frozen
    # Update-time re-evaluation with the stored masks reproduces rollout log-probs.
    t.model.eval()
    with torch.no_grad():
        logp, _, values = evaluate_actions(
            t.model, torch.from_numpy(buf.observations), torch.from_numpy(buf.masks),
            torch.from_numpy(buf.actions))
    np.testing.assert_allclose(logp.numpy(), buf.log_probs, atol=1e-5)
    np.testing.assert_allclose(values.numpy(), buf.values, atol=1e-5)
    assert timing["env_seconds"] > 0 and timing["inference_seconds"] > 0


def test_both_colours_and_truncation_appear_in_training_data(tmp_path) -> None:
    t = PPOTrainer(small_config(tmp_path, max_episode_plies=10, rollout_steps=64,
                                minibatch_size=32, game_max_plies=None))
    episodes, _ = t.collect_rollout()
    assert {e.agent_side for e in episodes} == {0, 1}
    assert any(e.outcome == "truncated" for e in episodes)
    buf = t.buffer
    assert buf.truncated.any() and not (buf.terminated & buf.truncated).any()
    assert np.isfinite(buf.truncation_values[buf.truncated]).all()
    assert (buf.rewards[buf.truncated] == 0).all()              # no result fabricated


def test_only_requested_opponents_are_used(tmp_path) -> None:
    t = PPOTrainer(small_config(tmp_path, opponents=["random"]))
    episodes, _ = t.collect_rollout()
    assert episodes and {e.opponent for e in episodes} == {"random"}


# ----------------------------------------------------------------------------- update


def test_update_changes_parameters_keeps_rollout_data_and_metrics_finite(tmp_path) -> None:
    t = PPOTrainer(small_config(tmp_path))
    t.collect_rollout()
    old_logp = t.buffer.log_probs.copy()
    old_obs = t.buffer.observations.copy()
    before = params(t.model)
    metrics = t.update()
    assert any(not torch.equal(a, b) for a, b in zip(before, t.model.parameters()))
    np.testing.assert_array_equal(t.buffer.log_probs, old_logp)   # behaviour data untouched
    np.testing.assert_array_equal(t.buffer.observations, old_obs)
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                "grad_norm", "mean_advantage", "mean_return"):
        assert np.isfinite(metrics[key]), key
    assert metrics["entropy"] > 0 and metrics["epochs_run"] == 2
    assert all(torch.isfinite(p).all() for p in t.model.parameters())


def test_update_reuses_the_rollout_masks(tmp_path, monkeypatch) -> None:
    import xiangqi_rl.algorithms.ppo.trainer as trainer_module

    t = PPOTrainer(small_config(tmp_path))
    t.collect_rollout()
    stored = {row.tobytes() for row in t.buffer.masks}
    calls = []
    real = trainer_module.evaluate_actions

    def spy(model, observations, masks, actions):
        logp, entropy, values = real(model, observations, masks, actions)
        calls.append((masks.numpy().copy(), actions.numpy().copy(), logp.detach().numpy().copy()))
        return logp, entropy, values

    monkeypatch.setattr(trainer_module, "evaluate_actions", spy)
    old_logp = t.buffer.log_probs.copy()
    t.update()
    assert calls
    for masks, actions, _ in calls:
        assert all(row.tobytes() in stored for row in masks)          # the stored masks...
        assert masks[np.arange(len(actions)), actions].all()          # ...containing the action
        assert masks.sum(axis=1).max() < masks.shape[1]               # never an all-True mask
    # First minibatch of the first epoch: behaviour policy == current policy, so
    # new log-probs equal the stored old ones (ratio 1) because the masks match.
    first_masks, first_actions, first_logp = calls[0]
    assert np.isfinite(first_logp).all()
    assert old_logp.min() < 0 and np.all(first_logp <= 0)


def test_gradient_clipping_is_applied(tmp_path) -> None:
    t = PPOTrainer(small_config(tmp_path, max_grad_norm=1e-3))
    t.collect_rollout()
    metrics = t.update()
    grads = [p.grad for p in t.model.parameters() if p.grad is not None]
    total = torch.sqrt(sum((g ** 2).sum() for g in grads)).item()
    assert total <= 1e-3 * (1 + 1e-4)                  # clipped to max_grad_norm
    assert metrics["grad_norm"] > 1e-3                 # pre-clip norm was larger


def test_nonfinite_loss_stops_before_any_optimizer_step(tmp_path) -> None:
    t = PPOTrainer(small_config(tmp_path))
    t.collect_rollout()
    t.buffer.advantages[:] = np.nan
    before = params(t.model)
    state_before = {k: v.clone() if torch.is_tensor(v) else v
                    for k, v in t.optimizer.state_dict()["state"].items()}
    with pytest.raises(NonFiniteError, match="non-finite loss"):
        t.update()
    assert all(torch.equal(a, b) for a, b in zip(before, t.model.parameters()))
    assert t.optimizer.state_dict()["state"].keys() == state_before.keys()  # no step happened


def test_target_kl_stops_epochs_early(tmp_path) -> None:
    t = PPOTrainer(small_config(tmp_path, update_epochs=8, target_kl=1e-12, learning_rate=0.01))
    t.collect_rollout()
    assert t.update()["epochs_run"] < 8


@pytest.mark.parametrize("extra", [{"value_clip_coef": 0.2}, {"optimizer": "sgd"},
                                   {"optimizer": "adamw"}, {"lr_schedule": "none"},
                                   {"normalize_advantages": False}])
def test_optional_features_run(tmp_path, extra) -> None:
    t = PPOTrainer(small_config(tmp_path, **extra))
    t.train()
    assert t.iteration == 2


def test_training_is_reproducible(tmp_path) -> None:
    def run(sub: str) -> tuple[list[float], list[torch.Tensor]]:
        t = PPOTrainer(small_config(tmp_path / sub, seed=7))
        history = t.train()
        return [h["policy_loss"] for h in history], params(t.model)

    (l1, p1), (l2, p2) = run("a"), run("b")
    assert l1 == l2 and all(torch.equal(a, b) for a, b in zip(p1, p2))


def test_train_loop_logs_and_checkpoints(tmp_path) -> None:
    cfg = small_config(tmp_path, eval_interval=1, eval_games=2, checkpoint_interval=1)
    t = PPOTrainer(cfg)
    history = t.train()
    out = Path(cfg.output_dir)
    assert (out / "latest.pt").exists() and (out / "best.pt").exists()
    rows = (out / "train_log.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + 2 and rows[0].startswith("iteration,")
    eval_rows = (out / "eval_log.csv").read_text().strip().splitlines()
    assert len(eval_rows) == 1 + 3 * 2          # iterations 0,1,2 x 2 opponents
    for key in ("env_seconds", "inference_seconds", "update_seconds", "wall_seconds"):
        assert history[0][key] >= 0
    assert not list(out.glob("*.tmp"))


# ----------------------------------------------------------------------------- checkpoint / resume


def test_resume_restores_training_state(tmp_path) -> None:
    cfg = small_config(tmp_path)
    t = PPOTrainer(cfg)
    t.train()
    t.save(tmp_path / "ckpt.pt")
    saved_torch_rng = torch.get_rng_state().clone()
    torch.manual_seed(999)  # perturb, then verify the restore brings it back
    r = PPOTrainer.from_checkpoint(tmp_path / "ckpt.pt")
    assert (r.iteration, r.total_decisions, r.total_episodes) == (2, t.total_decisions, t.total_episodes)
    assert all(torch.equal(a, b) for a, b in zip(t.model.parameters(), r.model.parameters()))
    s1, s2 = t.optimizer.state_dict()["state"], r.optimizer.state_dict()["state"]
    assert s1.keys() == s2.keys() and len(s1) > 0
    for k in s1:
        assert torch.equal(s1[k]["exp_avg"], s2[k]["exp_avg"]) and s1[k]["step"] == s2[k]["step"]
    assert torch.equal(torch.get_rng_state(), saved_torch_rng)
    assert r.envs.get_state() == t.envs.get_state()  # env RNGs, colour counters, opponent RNGs
    assert r._np_rng.bit_generator.state == t._np_rng.bit_generator.state


def test_resumed_run_completes_more_updates(tmp_path) -> None:
    t = PPOTrainer(small_config(tmp_path))
    t.train()
    cfg = small_config(tmp_path, total_iterations=3)
    r = PPOTrainer.from_checkpoint(Path(cfg.output_dir) / "latest.pt", config=cfg)
    history = r.train()
    assert r.iteration == 3 and len(history) == 1 and history[0]["learning_rate"] > 0
    assert np.isfinite(history[0]["policy_loss"])
    # CSV continues the same file (one header).
    text = (Path(cfg.output_dir) / "train_log.csv").read_text()
    assert text.count("iteration,") == 1 and len(text.strip().splitlines()) == 1 + 3


def test_new_run_refuses_existing_output_dir(tmp_path) -> None:
    cfg = small_config(tmp_path, total_iterations=1)
    PPOTrainer(cfg).train()
    out = Path(cfg.output_dir)
    log_before = (out / "train_log.csv").read_text()
    ckpt_before = (out / "latest.pt").read_bytes()
    with pytest.raises(FileExistsError, match="already contains a run"):
        PPOTrainer(cfg)
    assert (out / "train_log.csv").read_text() == log_before      # nothing appended
    assert (out / "latest.pt").read_bytes() == ckpt_before        # nothing overwritten
    resumed = PPOTrainer.from_checkpoint(out / "latest.pt")       # resuming is allowed
    assert resumed.iteration == 1


def test_evaluation_only_loading_and_incompatible_resume(tmp_path) -> None:
    t = PPOTrainer(small_config(tmp_path))
    t.train()
    model = load_model(Path(t.config.output_dir) / "latest.pt", map_location="cpu")
    assert model.config == t.config.model
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), t.model.parameters()))
    other = small_config(tmp_path, **{"model.hidden_channels": 32, "model.norm_groups": 4})
    with pytest.raises(ValueError, match="model config"):
        PPOTrainer.from_checkpoint(Path(t.config.output_dir) / "latest.pt", config=other)
    from xiangqi_rl.nn import create_model, save_checkpoint

    save_checkpoint(tmp_path / "plain.pt", create_model(t.config.model, seed=0))
    with pytest.raises(ValueError, match="not a resumable"):
        PPOTrainer.from_checkpoint(tmp_path / "plain.pt")


def test_evaluation_does_not_touch_training_state(tmp_path) -> None:
    t = PPOTrainer(small_config(tmp_path, eval_games=2))
    t.collect_rollout()
    before = params(t.model)
    buf_logp = t.buffer.log_probs.copy()
    rng_state = torch.get_rng_state().clone()
    t.model.train()
    results = t.evaluate(record=False)
    assert t.model.training
    assert all(torch.equal(a, b) for a, b in zip(before, t.model.parameters()))
    np.testing.assert_array_equal(t.buffer.log_probs, buf_logp)
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert set(results) == {"random", "heuristic"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_training_step(tmp_path) -> None:
    t = PPOTrainer(small_config(tmp_path, device="cuda"))
    t.train()
    assert next(t.model.parameters()).device.type == "cuda"
