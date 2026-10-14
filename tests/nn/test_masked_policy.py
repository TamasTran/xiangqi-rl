from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from xiangqi_rl.engine import Game, Side  # noqa: E402
from xiangqi_rl.env import ACTION_DIM, XiangqiEnv, encode_observation, legal_action_mask  # noqa: E402
from xiangqi_rl.nn import (  # noqa: E402
    EmptyLegalMaskError,
    MaskedCategorical,
    masked_logits,
    terminal_rows,
    value_to_red_perspective,
)


def batch_mask(*rows: list[int], dim: int = 20) -> torch.Tensor:
    mask = torch.zeros(len(rows), dim, dtype=torch.bool)
    for i, legal in enumerate(rows):
        mask[i, legal] = True
    return mask


def test_illegal_zero_legal_sum_to_one_and_rows_differ() -> None:
    logits = torch.randn(3, 20) * 5
    mask = batch_mask([0, 3, 7], [5], [1, 2, 3, 4, 19])
    dist = MaskedCategorical(logits, mask)
    probs = dist.probs
    assert (probs[~mask] == 0).all()
    assert torch.allclose(probs.sum(-1), torch.ones(3), atol=1e-6)
    assert probs[1, 5] == pytest.approx(1.0)
    assert torch.isfinite(dist.logits).all()


def test_masked_probs_equal_softmax_over_legal_subset() -> None:
    logits = torch.randn(1, 20)
    mask = batch_mask([2, 9, 11])
    expected = torch.softmax(logits[0, [2, 9, 11]], dim=0)
    assert torch.allclose(MaskedCategorical(logits, mask).probs[0, [2, 9, 11]], expected, atol=1e-6)


def test_stable_with_extreme_logits() -> None:
    logits = torch.tensor([[1e30, -1e30, 50.0, 0.0]])
    mask = torch.tensor([[False, True, True, True]])
    dist = MaskedCategorical(logits, mask)
    assert torch.isfinite(dist.logits).all() and torch.isfinite(dist.entropy()).all()
    assert dist.probs[0, 0] == 0 and dist.probs.sum() == pytest.approx(1.0)


def test_sampling_stays_legal_and_is_seedable() -> None:
    logits = torch.randn(64, 20)
    mask = batch_mask(*([[3, 4, 17]] * 32 + [[0]] * 32))
    dist = MaskedCategorical(logits, mask)
    gen = torch.Generator().manual_seed(0)
    actions = dist.sample(gen)
    assert actions.shape == (64,) and mask.gather(1, actions[:, None]).all()
    assert torch.equal(actions, dist.sample(torch.Generator().manual_seed(0)))
    assert (dist.mode()[32:] == 0).all()


def test_log_prob_and_entropy() -> None:
    logits = torch.zeros(1, 20)
    mask = batch_mask([0, 1, 2, 3])
    dist = MaskedCategorical(logits, mask)
    assert dist.log_prob(torch.tensor([2])).item() == pytest.approx(-torch.log(torch.tensor(4.0)).item())
    assert dist.entropy().item() == pytest.approx(torch.log(torch.tensor(4.0)).item(), abs=1e-6)
    single = MaskedCategorical(logits, batch_mask([5]))
    assert single.entropy().item() == pytest.approx(0.0, abs=1e-6)


def test_log_prob_rejects_illegal_or_malformed_actions() -> None:
    dist = MaskedCategorical(torch.zeros(2, 20), batch_mask([0, 1], [2]))
    with pytest.raises(ValueError, match="illegal"):
        dist.log_prob(torch.tensor([0, 3]))
    with pytest.raises(ValueError, match="out-of-range"):
        dist.log_prob(torch.tensor([0, 20]))
    with pytest.raises(ValueError):
        dist.log_prob(torch.tensor([0.0, 2.0]))
    with pytest.raises(ValueError):
        dist.log_prob(torch.tensor([0]))


def test_all_false_mask_raises_not_nan() -> None:
    mask = batch_mask([1, 2], [])
    assert terminal_rows(mask).tolist() == [False, True]
    with pytest.raises(EmptyLegalMaskError, match=r"\[1\]"):
        MaskedCategorical(torch.randn(2, 20), mask)
    with pytest.raises(EmptyLegalMaskError):
        masked_logits(torch.randn(2, 20), mask)


def test_invalid_masks_and_shapes_rejected() -> None:
    logits = torch.randn(2, 20)
    with pytest.raises(TypeError):
        MaskedCategorical(logits, torch.ones(2, 20))  # float mask
    with pytest.raises(ValueError):
        MaskedCategorical(logits, torch.ones(2, 19, dtype=torch.bool))
    with pytest.raises(ValueError):
        MaskedCategorical(logits[0], torch.ones(20, dtype=torch.bool))
    with pytest.raises(TypeError):
        MaskedCategorical(torch.zeros(2, 20, dtype=torch.int64), torch.ones(2, 20, dtype=torch.bool))


def test_gradient_flows_through_masked_log_prob() -> None:
    logits = torch.randn(2, 20, requires_grad=True)
    dist = MaskedCategorical(logits, batch_mask([1, 2, 3], [4, 5]))
    (dist.log_prob(torch.tensor([2, 5])).sum() + dist.entropy().sum()).backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[:, [0, 6, 7]] == 0).all()  # illegal logits get no gradient


def test_real_env_masks_and_terminal_state(make_fen) -> None:
    env = XiangqiEnv()
    _, info = env.reset()
    mask = torch.from_numpy(info["action_mask"]).unsqueeze(0)
    assert mask.shape == (1, ACTION_DIM) and mask.dtype == torch.bool and mask.sum() == 44
    dist = MaskedCategorical(torch.randn(1, ACTION_DIM), mask)
    action = int(dist.sample()[0])
    env.step(action)  # the env accepts every sampled action

    fen = make_fen({"d0": "K", "a8": "R", "i5": "R", "e9": "k"})
    game = Game.from_fen(fen)
    game.make_move("i5i9")
    terminal = torch.from_numpy(legal_action_mask(game)).unsqueeze(0)
    assert terminal_rows(terminal).tolist() == [True]
    with pytest.raises(EmptyLegalMaskError):
        MaskedCategorical(torch.randn(1, ACTION_DIM), terminal)


def test_value_perspective_helper() -> None:
    game = Game.new_game()
    red = torch.from_numpy(encode_observation(game)).unsqueeze(0)
    game.make_move("h2e2")
    black = torch.from_numpy(encode_observation(game)).unsqueeze(0)
    assert game.side_to_move is Side.BLACK
    obs = torch.cat([red, black])
    value = torch.tensor([[0.5], [0.5]])
    assert value_to_red_perspective(value, obs).flatten().tolist() == [0.5, -0.5]
    assert value_to_red_perspective(value.flatten(), obs).tolist() == [0.5, -0.5]
