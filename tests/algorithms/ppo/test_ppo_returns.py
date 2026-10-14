from __future__ import annotations

import numpy as np
import pytest

from xiangqi_rl.algorithms.ppo import compute_gae, explained_variance, normalize_advantages

F, T = False, True


def arr(x, dtype=np.float32):
    return np.array(x, dtype=dtype)


def test_gae_hand_calculated_terminal_episode() -> None:
    # gamma=0.9, lambda=0.8; r=[0,0,1], V=[0.5,0.4,0.3], episode ends at t=2.
    # d2 = 1 - 0.3 = 0.7                      A2 = 0.7
    # d1 = 0.9*0.3 - 0.4 = -0.13              A1 = -0.13 + 0.72*0.7 = 0.374
    # d0 = 0.9*0.4 - 0.5 = -0.14              A0 = -0.14 + 0.72*0.374 = 0.12928
    adv, ret = compute_gae(
        arr([0, 0, 1]), arr([0.5, 0.4, 0.3]), arr([F, F, T], bool), arr([F, F, F], bool),
        last_value=99.0, truncation_values=arr([0, 0, 0]), gamma=0.9, gae_lambda=0.8,
    )
    assert adv.dtype == np.float32 and adv.shape == (3,)
    np.testing.assert_allclose(adv, [0.12928, 0.374, 0.7], atol=1e-6)
    np.testing.assert_allclose(ret, [0.62928, 0.774, 1.0], atol=1e-6)  # A + V


def test_terminal_does_not_bootstrap_from_last_value() -> None:
    kw = dict(rewards=arr([1.0]), values=arr([0.2]), terminated=arr([T], bool),
              truncated=arr([F], bool), truncation_values=arr([0.0]), gamma=0.99, gae_lambda=0.95)
    a1, _ = compute_gae(**kw, last_value=0.0)
    a2, _ = compute_gae(**kw, last_value=123.0)
    assert a1[0] == a2[0] == pytest.approx(0.8)


def test_unfinished_rollout_bootstraps_from_last_value() -> None:
    adv, _ = compute_gae(arr([0.0]), arr([0.1]), arr([F], bool), arr([F], bool),
                         last_value=0.5, truncation_values=arr([0.0]), gamma=0.9, gae_lambda=1.0)
    assert adv[0] == pytest.approx(0.9 * 0.5 - 0.1)


def test_truncation_bootstraps_from_final_observation_value() -> None:
    # gamma = lambda = 1; t1 truncated with V(final)=0.8 -> d1 = 0.8 - 0.5 = 0.3,
    # t0: d0 = V1 - V0 = 0.3, A0 = 0.3 + 0.3 = 0.6.
    adv, _ = compute_gae(arr([0, 0]), arr([0.2, 0.5]), arr([F, F], bool), arr([F, T], bool),
                         last_value=7.0, truncation_values=arr([0.0, 0.8]), gamma=1.0, gae_lambda=1.0)
    np.testing.assert_allclose(adv, [0.6, 0.3], atol=1e-6)


def test_truncated_and_terminated_differ_only_by_bootstrap() -> None:
    common = dict(rewards=arr([0.0]), values=arr([0.1]), last_value=0.0,
                  truncation_values=arr([0.6]), gamma=1.0, gae_lambda=1.0)
    term, _ = compute_gae(terminated=arr([T], bool), truncated=arr([F], bool), **common)
    trunc, _ = compute_gae(terminated=arr([F], bool), truncated=arr([T], bool), **common)
    assert term[0] == pytest.approx(-0.1) and trunc[0] == pytest.approx(0.5)


def test_recursion_does_not_leak_across_episodes() -> None:
    # Two episodes: [t0, t1(end)] and [t2, t3 unfinished]. gamma=lambda=1.
    adv, _ = compute_gae(
        arr([0, 1, 0, 0]), arr([0, 0, 0, 0]), arr([F, T, F, F], bool), arr([F, F, F, F], bool),
        last_value=0.7, truncation_values=arr([0, 0, 0, 0]), gamma=1.0, gae_lambda=1.0,
    )
    # Second episode's 0.7 must not reach t1 / t0.
    np.testing.assert_allclose(adv, [1.0, 1.0, 0.7, 0.7], atol=1e-6)


def test_lambda_extremes() -> None:
    kw = dict(rewards=arr([0, 0, 1]), values=arr([0.5, 0.4, 0.3]),
              terminated=arr([F, F, T], bool), truncated=arr([F, F, F], bool),
              last_value=0.0, truncation_values=arr([0, 0, 0]), gamma=1.0)
    td, _ = compute_gae(**kw, gae_lambda=0.0)  # one-step TD
    np.testing.assert_allclose(td, [0.4 - 0.5, 0.3 - 0.4, 1 - 0.3], atol=1e-6)
    mc, ret = compute_gae(**kw, gae_lambda=1.0)  # Monte-Carlo return
    np.testing.assert_allclose(ret, [1.0, 1.0, 1.0], atol=1e-6)


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="both terminated and truncated"):
        compute_gae(arr([0]), arr([0]), arr([T], bool), arr([T], bool), 0.0, arr([0]), 1.0, 1.0)
    with pytest.raises(ValueError, match="length"):
        compute_gae(arr([0, 0]), arr([0]), arr([F, F], bool), arr([F, F], bool), 0.0,
                    arr([0, 0]), 1.0, 1.0)


def test_normalize_advantages() -> None:
    out = normalize_advantages(arr([1.0, 2.0, 3.0, 10.0]))
    assert out.dtype == np.float32 and abs(out.mean()) < 1e-6 and out.std() == pytest.approx(1.0, abs=1e-4)
    np.testing.assert_array_equal(normalize_advantages(arr([5.0, 5.0, 5.0])), [0, 0, 0])


def test_explained_variance() -> None:
    target = arr([1.0, 2.0, 3.0])
    assert explained_variance(target, target) == pytest.approx(1.0)
    assert explained_variance(np.full(3, 2.0, np.float32), target) == pytest.approx(0.0)
    assert np.isnan(explained_variance(target, np.ones(3, np.float32)))
