from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from xiangqi_rl.algorithms.ppo import evaluate_actions, ppo_policy_loss, total_loss, value_loss  # noqa: E402
from xiangqi_rl.algorithms.ppo import act  # noqa: E402
from xiangqi_rl.env import XiangqiEnv  # noqa: E402
from xiangqi_rl.nn import ModelConfig, create_model  # noqa: E402


def policy(ratio: float, adv: float, eps: float = 0.2):
    new = torch.tensor([math.log(ratio)], requires_grad=True)
    out = ppo_policy_loss(new, torch.zeros(1), torch.tensor([adv]), eps)
    out.loss.backward()
    return out, new.grad.item()


@pytest.mark.parametrize(
    ("ratio", "adv", "loss", "grad_nonzero"),
    [
        (1.1, 2.0, -2.2, True),    # inside range, A>0: unclipped
        (1.5, 1.0, -1.2, False),   # above range, A>0: clipped, no gradient
        (0.5, 1.0, -0.5, True),    # below range, A>0: min picks unclipped
        (1.5, -1.0, 1.5, True),    # above range, A<0: min picks unclipped (pessimistic)
        (0.5, -1.0, 0.8, False),   # below range, A<0: clipped, no gradient
        (0.9, -2.0, 1.8, True),    # inside range, A<0
    ],
)
def test_clipped_objective(ratio, adv, loss, grad_nonzero) -> None:
    out, grad = policy(ratio, adv)
    assert out.loss.item() == pytest.approx(loss, abs=1e-6)
    assert (abs(grad) > 1e-9) == grad_nonzero


def test_ratio_is_exp_of_log_difference_and_diagnostics() -> None:
    new = torch.tensor([0.0, math.log(1.5), math.log(0.5), math.log(1.1)])
    out = ppo_policy_loss(new, torch.zeros(4), torch.ones(4), 0.2)
    assert torch.allclose(out.ratio, torch.tensor([1.0, 1.5, 0.5, 1.1]))
    assert out.clip_fraction.item() == pytest.approx(0.5)  # 1.5 and 0.5 are outside
    expected_kl = sum((r - 1) - math.log(r) for r in (1.0, 1.5, 0.5, 1.1)) / 4
    assert out.approx_kl.item() == pytest.approx(expected_kl, rel=1e-5)
    same = ppo_policy_loss(torch.zeros(3), torch.zeros(3), torch.ones(3), 0.2)
    assert same.approx_kl.item() == pytest.approx(0.0) and same.clip_fraction.item() == 0.0


def test_old_log_probs_receive_no_gradient() -> None:
    new = torch.zeros(2, requires_grad=True)
    old = torch.zeros(2, requires_grad=True)
    ppo_policy_loss(new, old, torch.ones(2), 0.2).loss.backward()
    assert old.grad is None


def test_value_loss_plain() -> None:
    v = torch.tensor([1.0, 3.0])
    r = torch.tensor([0.0, 1.0])
    assert value_loss(v, v.detach(), r).item() == pytest.approx(0.5 * (1 + 4) / 2)


def test_value_loss_clipping() -> None:
    # old=0, new=1, return=0.1, eps=0.2: unclipped (0.9)^2, clipped (0.2-0.1)^2 -> max = 0.81
    args = (torch.tensor([1.0]), torch.tensor([0.0]))
    assert value_loss(*args, torch.tensor([0.1]), 0.2).item() == pytest.approx(0.5 * 0.81)
    # return=1.0: unclipped 0, clipped (0.2-1)^2=0.64 -> clipping keeps the penalty.
    assert value_loss(*args, torch.tensor([1.0]), 0.2).item() == pytest.approx(0.5 * 0.64)


def test_total_loss_combination_and_gradients_finite() -> None:
    assert total_loss(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0), 0.5, 0.01).item() \
        == pytest.approx(1.0 + 1.0 - 0.03)
    new = torch.randn(8, requires_grad=True)
    v = torch.randn(8, requires_grad=True)
    pol = ppo_policy_loss(new, torch.randn(8), torch.randn(8), 0.2)
    loss = total_loss(pol.loss, value_loss(v, torch.randn(8), torch.randn(8), 0.2),
                      torch.tensor(1.0), 0.5, 0.01)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(new.grad).all() and torch.isfinite(v.grad).all()


def test_old_and_new_log_probs_use_the_same_mask() -> None:
    """At the behaviour policy, update-time evaluation reproduces rollout log-probs."""
    cfg = ModelConfig(hidden_channels=16, num_res_blocks=1, norm_groups=4, policy_channels=4,
                      value_channels=4, value_hidden=16)
    model = create_model(cfg, seed=0).eval()
    env = XiangqiEnv()
    obs, info = env.reset()
    records = []
    for _ in range(6):
        mask = info["action_mask"]
        action, logp, value = act(model, obs, mask, torch.device("cpu"))
        records.append((obs, mask, action, logp, value))
        obs, _, _, _, info = env.step(action)
    o = torch.from_numpy(__import__("numpy").stack([r[0] for r in records]))
    m = torch.from_numpy(__import__("numpy").stack([r[1] for r in records]))
    a = torch.tensor([r[2] for r in records])
    with torch.no_grad():
        logp, entropy, values = evaluate_actions(model, o, m, a)
    assert torch.allclose(logp, torch.tensor([r[3] for r in records]), atol=1e-5)
    assert torch.allclose(values, torch.tensor([r[4] for r in records]), atol=1e-5)
    assert (entropy > 0).all() and torch.isfinite(logp).all() and (logp <= 0).all()
    # The log-prob under an *unmasked* policy would differ: masking matters.
    with torch.no_grad():
        raw = torch.log_softmax(model(o)[0], dim=-1).gather(1, a[:, None]).squeeze(1)
    assert not torch.allclose(raw, logp, atol=1e-3)
