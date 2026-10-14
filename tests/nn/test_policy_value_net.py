from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from xiangqi_rl.engine import Game  # noqa: E402
from xiangqi_rl.env import ACTION_DIM, encode_observation  # noqa: E402
from xiangqi_rl.nn import ModelConfig, PolicyValueNet, create_model  # noqa: E402

SMALL = ModelConfig(hidden_channels=16, num_res_blocks=2, norm_groups=4, policy_channels=4,
                    value_channels=4, value_hidden=16)


@pytest.fixture(scope="module")
def model() -> PolicyValueNet:
    return create_model(SMALL, seed=0)


def test_single_and_batch_shapes(model) -> None:
    for batch in (1, 5):
        logits, value = model(torch.randn(batch, 17, 10, 9))
        assert logits.shape == (batch, ACTION_DIM) and value.shape == (batch, 1)
        assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_value_bounded_even_for_extreme_input(model) -> None:
    _, value = model(torch.randn(8, 17, 10, 9) * 1e3)
    assert value.abs().max() <= 1.0


def test_value_unbounded_when_activation_none() -> None:
    m = create_model(ModelConfig(**{**SMALL.to_dict(), "value_activation": "none"}), seed=0)
    last = [mod for mod in m.value_head if isinstance(mod, torch.nn.Linear)][-1]
    last.bias.data.fill_(5.0)
    _, value = m(torch.zeros(1, 17, 10, 9))
    assert value.item() == pytest.approx(5.0, abs=0.1)


def test_policy_dim_follows_config() -> None:
    m = create_model(ModelConfig(**{**SMALL.to_dict(), "action_dim": 123}))
    logits, _ = m(torch.zeros(2, 17, 10, 9))
    assert logits.shape == (2, 123)


@pytest.mark.parametrize(
    "bad",
    [torch.zeros(17, 10, 9), torch.zeros(2, 16, 10, 9), torch.zeros(2, 17, 9, 10),
     torch.zeros(2, 17, 10, 9, dtype=torch.int64)],
)
def test_wrong_input_rejected(model, bad) -> None:
    with pytest.raises((ValueError, TypeError)):
        model(bad)


def test_real_phase2_observation_is_accepted(model) -> None:
    game = Game.new_game()
    obs = torch.from_numpy(encode_observation(game)).unsqueeze(0)
    logits, value = model(obs)
    assert logits.shape == (1, ACTION_DIM) and torch.isfinite(value).all()


def test_untrained_model_is_near_uniform_and_near_zero(model) -> None:
    logits, value = model(torch.from_numpy(encode_observation(Game.new_game())).unsqueeze(0))
    assert logits.abs().max() < 1.0 and value.abs().item() < 0.1


@pytest.mark.parametrize("norm", ["group", "batch", "none"])
def test_gradients_flow_to_all_parameters(norm) -> None:
    cfg = ModelConfig(**{**SMALL.to_dict(), "norm": norm})
    m = create_model(cfg, seed=1)
    # Make the output layers non-trivial so every layer receives gradient.
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear):
            torch.nn.init.normal_(mod.weight, std=0.1)
    logits, value = m(torch.randn(4, 17, 10, 9))
    loss = logits.logsumexp(-1).mean() + value.pow(2).mean()
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    stem_conv = m.stem[0].weight.grad  # reaches the shared backbone
    assert stem_conv.abs().sum() > 0


def test_each_head_alone_updates_the_backbone() -> None:
    m = create_model(SMALL, seed=2)
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear):
            torch.nn.init.normal_(mod.weight, std=0.1)
    for head in (0, 1):
        m.zero_grad()
        out = m(torch.randn(3, 17, 10, 9))[head]
        out.sum().backward()
        assert m.stem[0].weight.grad.abs().sum() > 0


def test_optimizer_step_changes_weights_and_reduces_loss() -> None:
    m = create_model(SMALL, seed=3)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x = torch.randn(8, 17, 10, 9)
    target_action = torch.randint(0, ACTION_DIM, (8,))
    target_value = torch.full((8, 1), 0.5)
    before = [p.detach().clone() for p in m.parameters()]
    losses = []
    for _ in range(5):
        logits, value = m(x)
        loss = torch.nn.functional.cross_entropy(logits, target_action) \
            + torch.nn.functional.mse_loss(value, target_value)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert all(map(lambda v: v == v, losses)) and losses[-1] < losses[0]
    assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))


def test_batch_independence_with_group_norm(model) -> None:
    x = torch.randn(4, 17, 10, 9)
    full_logits, full_value = model(x)
    one_logits, one_value = model(x[2:3])
    assert torch.allclose(full_logits[2], one_logits[0], atol=1e-5)
    assert torch.allclose(full_value[2], one_value[0], atol=1e-5)


def test_seeded_creation_is_reproducible_and_preserves_global_rng() -> None:
    torch.manual_seed(123)
    state = torch.get_rng_state()
    a, b = create_model(SMALL, seed=7), create_model(SMALL, seed=7)
    assert torch.equal(torch.get_rng_state(), state)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_parameter_count_matches_manual_sum() -> None:
    m = create_model(SMALL, seed=0)
    assert m.num_parameters() == sum(p.numel() for p in m.parameters())
    for p in m.stem.parameters():
        p.requires_grad_(False)
    assert m.num_parameters() < m.num_parameters(trainable_only=False)


def test_cpu_and_cuda_forward() -> None:
    m = create_model(SMALL, seed=0)
    logits, value = m(torch.randn(2, 17, 10, 9))
    assert logits.device.type == "cpu"
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cuda = create_model(SMALL, seed=0).to("cuda")
    c_logits, c_value = cuda(torch.randn(2, 17, 10, 9, device="cuda"))
    assert c_logits.shape == (2, ACTION_DIM) and torch.isfinite(c_value).all()
