"""Several :class:`OpponentEnv` instances stepped together.

Why: the network runs on the GPU, but the rules engine and the baseline
opponents are pure Python and run on the CPU. With one game at a time the GPU
sees batches of size 1 and mostly waits. With ``N`` games, every step gives the
GPU one batch of ``N`` positions, and the games themselves can be spread over
worker processes so several CPU cores simulate in parallel.

Two backends with the same interface and identical results for the same
specs and actions:

* :class:`InProcessVecEnv` steps all games in the calling process.
* :class:`SubprocVecEnv` splits them over worker processes (``spawn`` start
  method, so it is safe next to an initialised CUDA context).

Finished games are reset automatically. :meth:`step` returns the observation
of the *new* game for those environments and, separately, the final
observation of the finished one (needed to bootstrap truncated episodes).
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np

from xiangqi_rl.agents import HeuristicAgent, LookaheadAgent, RandomAgent
from xiangqi_rl.algorithms.ppo.opponent_env import OpponentEnv
from xiangqi_rl.engine import GameConfig

#: Keys of the per-step info dict that callers rely on.
INFO_KEYS = ("game_reward", "agent_side", "side_to_move", "opponent", "ply")


@dataclass(frozen=True)
class EnvSpec:
    """Picklable recipe for one :class:`OpponentEnv` (built inside workers)."""

    opponents: tuple[str, ...]
    opponent_seeds: tuple[int, ...]
    seed: int
    game_max_plies: int | None
    max_episode_plies: int | None
    color_schedule: str
    material_shaping_coef: float
    shaping_gamma: float
    color_offset: int = 0


def _make_agent(name: str, seed: int):
    if name == "random":
        return RandomAgent(seed=seed, name="random")
    if name == "heuristic":
        return HeuristicAgent(seed=seed, name="heuristic")
    if name == "lookahead":
        return LookaheadAgent(seed=seed, name="lookahead")
    raise ValueError(f"unknown opponent {name!r}")


def build_env(spec: EnvSpec) -> OpponentEnv:
    """Construct the environment (and its seeded opponents) described by ``spec``."""
    agents = [_make_agent(n, s) for n, s in zip(spec.opponents, spec.opponent_seeds)]
    return OpponentEnv(
        agents,
        game_config=GameConfig(max_plies=spec.game_max_plies),
        max_episode_plies=spec.max_episode_plies,
        color_schedule=spec.color_schedule,
        seed=spec.seed,
        material_shaping_coef=spec.material_shaping_coef,
        shaping_gamma=spec.shaping_gamma,
        color_offset=spec.color_offset,
    )


# ---------------------------------------------------------------------------- shared logic


def _reset_env(env: OpponentEnv) -> tuple[np.ndarray, np.ndarray]:
    obs, info = env.reset()
    return obs, info["action_mask"]


def _step_env(env: OpponentEnv, action: int) -> tuple:
    """One step with auto-reset. Returns
    ``(obs, mask, reward, terminated, truncated, info, final_obs)``."""
    obs, reward, terminated, truncated, info = env.step(action)
    small = {k: info[k] for k in INFO_KEYS}
    final_obs = None
    if terminated or truncated:
        final_obs = obs
        obs, mask = _reset_env(env)
    else:
        mask = info["action_mask"]
    return obs, mask, reward, terminated, truncated, small, final_obs


def _get_state(env: OpponentEnv) -> dict[str, Any]:
    return {
        "rng": env.rng_state,
        "episodes_started": env.episodes_started,
        "opponents": [a.rng_state for a in env._opponents],
    }


def _set_state(env: OpponentEnv, state: dict[str, Any]) -> None:
    env.rng_state = state["rng"]
    env.episodes_started = int(state["episodes_started"])
    for agent, agent_state in zip(env._opponents, state["opponents"]):
        agent.rng_state = agent_state


def _stack(results: list[tuple]) -> tuple:
    obs, masks, rewards, term, trunc, infos, finals = zip(*results)
    return (
        np.stack(obs).astype(np.float32, copy=False),
        np.stack(masks),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(term, dtype=np.bool_),
        np.asarray(trunc, dtype=np.bool_),
        list(infos),
        list(finals),
    )


class VecEnv:
    """Common interface (see module docstring)."""

    num_envs: int
    observation_shape: tuple[int, ...]
    action_dim: int

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Start a game in every environment: ``(obs [N,...], masks [N, A])``."""
        raise NotImplementedError

    def step(self, actions: np.ndarray) -> tuple:
        """Step every environment once; returns
        ``(obs, masks, rewards, terminated, truncated, infos, final_obs)``."""
        raise NotImplementedError

    def get_state(self) -> list[dict[str, Any]]:
        """RNG states and counters of every environment (for checkpoints)."""
        raise NotImplementedError

    def set_state(self, states: list[dict[str, Any]]) -> None:
        """Restore states from :meth:`get_state` (extra/missing entries ignored)."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources (worker processes)."""


class InProcessVecEnv(VecEnv):
    """All environments in the calling process."""

    def __init__(self, specs: list[EnvSpec]) -> None:
        self.envs = [build_env(s) for s in specs]
        self.num_envs = len(self.envs)
        self.observation_shape = self.envs[0].observation_shape
        self.action_dim = self.envs[0].action_dim

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        obs, masks = zip(*(_reset_env(e) for e in self.envs))
        return np.stack(obs), np.stack(masks)

    def step(self, actions: np.ndarray) -> tuple:
        return _stack([_step_env(e, int(a)) for e, a in zip(self.envs, actions)])

    def get_state(self) -> list[dict[str, Any]]:
        return [_get_state(e) for e in self.envs]

    def set_state(self, states: list[dict[str, Any]]) -> None:
        for env, state in zip(self.envs, states):
            _set_state(env, state)


# ---------------------------------------------------------------------------- workers


def _worker(conn, specs: list[EnvSpec]) -> None:  # pragma: no cover - runs in a subprocess
    try:
        envs = [build_env(s) for s in specs]
    except Exception:
        conn.send(("error", traceback.format_exc()))
        return
    conn.send(("ok", (envs[0].observation_shape, envs[0].action_dim)))
    while True:
        cmd, payload = conn.recv()
        try:
            if cmd == "reset":
                result = [_reset_env(e) for e in envs]
            elif cmd == "step":
                result = [_step_env(e, int(a)) for e, a in zip(envs, payload)]
            elif cmd == "get_state":
                result = [_get_state(e) for e in envs]
            elif cmd == "set_state":
                for env, state in zip(envs, payload):
                    _set_state(env, state)
                result = None
            elif cmd == "close":
                conn.send(("ok", None))
                return
            else:
                raise ValueError(f"unknown command {cmd!r}")
        except Exception:
            conn.send(("error", traceback.format_exc()))
            continue
        conn.send(("ok", result))


class WorkerError(RuntimeError):
    """An environment raised inside a worker process (traceback included)."""


class SubprocVecEnv(VecEnv):
    """Environments split over ``num_workers`` processes (contiguous chunks)."""

    def __init__(self, specs: list[EnvSpec], num_workers: int) -> None:
        if not 1 <= num_workers <= len(specs):
            raise ValueError(f"num_workers must be in [1, {len(specs)}], got {num_workers}")
        self.num_envs = len(specs)
        bounds = np.linspace(0, len(specs), num_workers + 1).astype(int)
        self._chunks = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]
        ctx = mp.get_context("spawn")
        self._conns, self._procs = [], []
        for a, b in self._chunks:
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, specs[a:b]), daemon=True)
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)
        shapes = [self._recv(c) for c in self._conns]
        self.observation_shape, self.action_dim = shapes[0]
        self._closed = False

    @staticmethod
    def _recv(conn) -> Any:
        try:
            status, payload = conn.recv()
        except EOFError:
            raise WorkerError(
                "an environment worker process exited unexpectedly. With the 'spawn' start "
                "method the main program must be importable: run training from a script "
                "file (python scripts/train_ppo.py ...), not from stdin; or use env_workers=0."
            ) from None
        if status == "error":
            raise WorkerError(payload)
        return payload

    def _broadcast(self, cmd: str, payloads: list[Any] | None = None) -> list[Any]:
        for i, conn in enumerate(self._conns):
            conn.send((cmd, None if payloads is None else payloads[i]))
        return [self._recv(c) for c in self._conns]

    def _split(self, items) -> list[Any]:
        return [list(items[a:b]) for a, b in self._chunks]

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        parts = self._broadcast("reset")
        obs, masks = zip(*(r for part in parts for r in part))
        return np.stack(obs), np.stack(masks)

    def step(self, actions: np.ndarray) -> tuple:
        parts = self._broadcast("step", self._split([int(a) for a in actions]))
        return _stack([r for part in parts for r in part])

    def get_state(self) -> list[dict[str, Any]]:
        return [s for part in self._broadcast("get_state") for s in part]

    def set_state(self, states: list[dict[str, Any]]) -> None:
        padded = list(states) + [None] * (self.num_envs - len(states))
        chunks = [[s for s in chunk if s is not None] for chunk in self._split(padded)]
        self._broadcast("set_state", chunks)

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        for conn in self._conns:
            try:
                conn.send(("close", None))
                conn.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()

    def __del__(self) -> None:
        self.close()


def make_vec_env(specs: list[EnvSpec], num_workers: int) -> VecEnv:
    """``num_workers == 0``: in-process; otherwise :class:`SubprocVecEnv`."""
    if num_workers == 0:
        return InProcessVecEnv(specs)
    return SubprocVecEnv(specs, num_workers)
