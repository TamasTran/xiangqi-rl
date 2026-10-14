"""Model + real Phase 2 environment: the observation -> logits -> mask -> action loop."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xiangqi_rl.engine import GameConfig  # noqa: E402
from xiangqi_rl.env import ACTION_DIM, XiangqiEnv  # noqa: E402
from xiangqi_rl.nn import MaskedCategorical, ModelConfig, create_model  # noqa: E402

CFG = ModelConfig(hidden_channels=16, num_res_blocks=1, norm_groups=4, policy_channels=4,
                  value_channels=4, value_hidden=16)


def test_untrained_model_can_drive_the_environment_for_a_whole_game() -> None:
    model = create_model(CFG, seed=0).eval()
    assert model.config.action_dim == XiangqiEnv().action_space.n == ACTION_DIM
    env = XiangqiEnv(game_config=GameConfig(max_plies=60))
    obs, info = env.reset(seed=0)
    gen = torch.Generator().manual_seed(0)
    terminated = truncated = False
    sides = []
    while not (terminated or truncated):
        x = torch.from_numpy(obs).unsqueeze(0)
        assert x.dtype == torch.float32 and tuple(x.shape) == (1, 17, 10, 9)
        with torch.no_grad():
            logits, value = model(x)
        mask = torch.from_numpy(env.action_masks()).unsqueeze(0)
        assert mask.shape == logits.shape
        dist = MaskedCategorical(logits, mask)
        action = int(dist.sample(gen).item())
        assert np.flatnonzero(info["action_mask"]).tolist().count(action) == 1
        sides.append(info["current_player"])
        obs, _, terminated, truncated, info = env.step(action)  # never IllegalActionError
    assert sides[:4] == [0, 1, 0, 1]
    assert env.state.is_terminal  # reached the engine's 60-ply draw at the latest


def test_batched_masks_from_different_positions() -> None:
    model = create_model(CFG, seed=0).eval()
    envs = [XiangqiEnv() for _ in range(3)]
    obs = []
    for i, env in enumerate(envs):
        o, _ = env.reset()
        for _ in range(i * 2):  # different depths -> different positions and masks
            o, *_ = env.step(int(np.flatnonzero(env.action_masks())[0]))
        obs.append(torch.from_numpy(o))
    masks = [torch.from_numpy(e.action_masks()) for e in envs]
    logits, value = model(torch.stack(obs))
    dist = MaskedCategorical(logits, torch.stack(masks))
    probs = dist.probs
    assert (probs[~torch.stack(masks)] == 0).all()
    assert torch.allclose(probs.sum(-1), torch.ones(3), atol=1e-5)
    actions = dist.sample()
    for env, a in zip(envs, actions.tolist()):
        env.step(a)
