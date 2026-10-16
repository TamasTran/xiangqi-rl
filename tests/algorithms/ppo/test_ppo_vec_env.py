"""Parallel environments: batched rollouts, worker processes, per-env GAE."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xiangqi_rl.algorithms.ppo import SMOKE_OVERRIDES, PPOConfig, PPOTrainer, RolloutBuffer, compute_gae  # noqa: E402
from xiangqi_rl.algorithms.ppo.vec_env import (  # noqa: E402
    EnvSpec,
    InProcessVecEnv,
    SubprocVecEnv,
    WorkerError,
)


def specs(n: int, opponents=("random", "heuristic")) -> list[EnvSpec]:
    return [EnvSpec(opponents=opponents, opponent_seeds=tuple(100 * e + i for i in range(len(opponents))),
                    seed=e, game_max_plies=30, max_episode_plies=None, color_schedule="alternate",
                    material_shaping_coef=0.5, shaping_gamma=0.99, color_offset=e) for e in range(n)]


def drive(vec, steps: int, seed: int = 0):
    """Play random legal actions; return everything the env produced."""
    rng = np.random.default_rng(seed)
    obs, masks = vec.reset()
    out = [(obs.copy(), masks.copy())]
    for _ in range(steps):
        actions = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
        obs, masks, rewards, term, trunc, infos, finals = vec.step(actions)
        out.append((obs.copy(), masks.copy(), rewards.copy(), term.copy(), trunc.copy(), infos,
                    [None if f is None else f.copy() for f in finals]))
    return out


# ----------------------------------------------------------------------------- buffer


def test_buffer_gae_runs_separately_per_environment() -> None:
    E, T = 2, 5
    buf = RolloutBuffer(E * T, (1, 1, 1), 4, num_envs=E)
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=(T, E)).astype(np.float32)
    values = rng.normal(size=(T, E)).astype(np.float32)
    term = np.zeros((T, E), bool)
    term[2, 0] = True                      # env 0 ends an episode at t=2, env 1 never
    mask = np.zeros((E, 4), bool)
    mask[:, 1] = True
    for t in range(T):
        buf.add_batch(np.full((E, 1, 1, 1), t, np.float32), mask, np.ones(E, int), np.full(E, -0.1),
                      values[t], rewards[t], term[t], np.zeros(E, bool), np.zeros(E))
    last = np.array([0.3, -0.7])
    buf.compute_advantages(last, 0.9, 0.8, normalize=False)
    for e in range(E):
        adv, ret = compute_gae(rewards[:, e], values[:, e], term[:, e], np.zeros(T, bool),
                               last[e], np.zeros(T, np.float32), 0.9, 0.8)
        np.testing.assert_allclose(buf.raw_advantages[e::E], adv, atol=1e-6)
        np.testing.assert_allclose(buf.returns[e::E], ret, atol=1e-6)
    np.testing.assert_array_equal(buf.observations[::E, 0, 0, 0], np.arange(T))  # time-major


def test_buffer_rejects_bad_num_envs() -> None:
    with pytest.raises(ValueError):
        RolloutBuffer(10, (1,), 4, num_envs=3)


# ----------------------------------------------------------------------------- vec env


def test_subprocess_backend_matches_in_process_exactly() -> None:
    a = InProcessVecEnv(specs(4))
    b = SubprocVecEnv(specs(4), num_workers=2)
    try:
        ra, rb = drive(a, 70), drive(b, 70)
        assert len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for u, v in zip(x, y):
                if isinstance(u, np.ndarray):
                    np.testing.assert_array_equal(u, v)
                elif isinstance(u, list) and u and isinstance(u[0], dict):
                    assert u == v
                elif isinstance(u, list):
                    assert all((p is None and q is None) or np.array_equal(p, q) for p, q in zip(u, v))
        assert any(step[3].any() for step in ra[1:])          # some games finished (auto-reset)
        assert a.get_state() == b.get_state()
    finally:
        b.close()


def test_auto_reset_returns_final_and_new_observation() -> None:
    vec = InProcessVecEnv(specs(2))
    rng = np.random.default_rng(1)
    obs, masks = vec.reset()
    for _ in range(200):
        actions = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
        obs, masks, rewards, term, trunc, infos, finals = vec.step(actions)
        done = term | trunc
        for i in range(2):
            assert (finals[i] is not None) == bool(done[i])
            assert masks[i].any()                               # always a live decision point
        if done.any():
            i = int(np.flatnonzero(done)[0])
            assert not np.array_equal(finals[i], obs[i])        # new game, not the final position
            return
    pytest.fail("no game finished")


def test_colours_are_staggered_across_environments() -> None:
    vec = InProcessVecEnv(specs(4))
    vec.reset()
    assert [e.agent_side.name for e in vec.envs] == ["RED", "BLACK", "RED", "BLACK"]


def test_state_roundtrip_and_worker_errors() -> None:
    a = InProcessVecEnv(specs(3))
    drive(a, 25)
    state = a.get_state()
    b = SubprocVecEnv(specs(3), num_workers=3)
    try:
        b.set_state(state)
        assert b.get_state() == state
    finally:
        b.close()
    with pytest.raises(WorkerError, match="unknown opponent"):
        SubprocVecEnv(specs(2, opponents=("stockfish",)), num_workers=1)


# ----------------------------------------------------------------------------- trainer


def config(tmp_path, **extra) -> PPOConfig:
    return PPOConfig().with_overrides({**SMOKE_OVERRIDES, "device": "cpu", "eval_interval": 0,
                                       "total_iterations": 1, "output_dir": str(tmp_path / "run"),
                                       "rollout_steps": 64, "minibatch_size": 32, **extra})


def test_trainer_with_parallel_envs_collects_legal_batched_rollouts(tmp_path) -> None:
    # 128 decisions / 4 envs = 32 per env; smoke games last <= 20 decisions.
    t = PPOTrainer(config(tmp_path, num_envs=4, rollout_steps=128))
    episodes, _ = t.collect_rollout()
    buf = t.buffer
    assert buf.is_full and buf.num_envs == 4
    assert buf.masks[np.arange(128), buf.actions].all()
    assert np.isfinite(buf.advantages).all()
    assert {e.agent_side for e in episodes} == {0, 1}
    assert np.isfinite(t.update()["policy_loss"])


def test_worker_processes_give_identical_training(tmp_path) -> None:
    def run(sub: str, workers: int):
        t = PPOTrainer(config(tmp_path / sub, num_envs=4, env_workers=workers, seed=3))
        try:
            history = t.train()
            return history[0]["policy_loss"], t.buffer.actions.copy(), [p.detach().clone() for p in t.model.parameters()]
        finally:
            t.close()

    (l0, a0, p0), (l2, a2, p2) = run("inproc", 0), run("workers", 2)
    assert l0 == l2
    np.testing.assert_array_equal(a0, a2)
    assert all(torch.equal(x, y) for x, y in zip(p0, p2))


def test_resume_restores_every_environment(tmp_path) -> None:
    cfg = config(tmp_path, num_envs=4)
    t = PPOTrainer(cfg)
    t.train()
    t.save(tmp_path / "ckpt.pt")
    r = PPOTrainer.from_checkpoint(tmp_path / "ckpt.pt")
    assert r.envs.get_state() == t.envs.get_state() and len(r.envs.get_state()) == 4


def test_resume_from_single_env_checkpoint_format(tmp_path) -> None:
    """Checkpoints written before num_envs existed stored one env's state."""
    t = PPOTrainer(config(tmp_path, num_envs=1, env_workers=0))
    t.train()
    t.save(tmp_path / "new.pt")
    payload = torch.load(tmp_path / "new.pt", map_location="cpu", weights_only=True)
    extra = payload["extra"]
    env_state = extra.pop("envs")[0]
    extra["episodes_started"] = env_state["episodes_started"]
    extra["rng"]["env"] = env_state["rng"]
    extra["rng"]["opponents"] = env_state["opponents"]
    for key in ("num_envs", "env_workers"):
        extra["ppo_config"].pop(key)
    torch.save(payload, tmp_path / "old.pt")
    r = PPOTrainer.from_checkpoint(tmp_path / "old.pt")
    assert r.envs.get_state() == t.envs.get_state()


def test_config_validation_for_parallel_envs() -> None:
    with pytest.raises(ValueError, match="divide"):
        PPOConfig(rollout_steps=100, minibatch_size=50, num_envs=3)
    with pytest.raises(ValueError, match="env_workers"):
        PPOConfig(num_envs=2, env_workers=3)
    assert PPOConfig(num_envs=16, env_workers=-1).env_workers == -1
