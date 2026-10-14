from __future__ import annotations

import numpy as np
import pytest

from xiangqi_rl.algorithms.ppo import BufferStateError, RolloutBuffer

OBS, ACT = (2, 3, 3), 10


def fill(buf: RolloutBuffer, n: int, *, done_at: dict[int, str] | None = None) -> None:
    done_at = done_at or {}
    for i in range(n):
        obs = np.full(OBS, float(i), dtype=np.float32)
        mask = np.zeros(ACT, dtype=bool)
        mask[[i % ACT, (i + 3) % ACT]] = True
        kind = done_at.get(i)
        buf.add(obs, mask, i % ACT, -0.5 - i, 0.1 * i, 1.0 if kind == "T" else 0.0,
                kind == "T", kind == "X", 0.25 if kind == "X" else 0.0)


def test_shapes_dtypes_and_alignment() -> None:
    buf = RolloutBuffer(6, OBS, ACT)
    fill(buf, 6, done_at={2: "T", 4: "X"})
    assert buf.is_full and len(buf) == 6
    assert buf.observations.shape == (6, *OBS) and buf.observations.dtype == np.float32
    assert buf.masks.shape == (6, ACT) and buf.masks.dtype == np.bool_
    assert buf.actions.dtype == np.int64
    for arr in (buf.log_probs, buf.values, buf.rewards, buf.advantages, buf.returns):
        assert arr.shape == (6,) and arr.dtype == np.float32
    assert buf.terminated.tolist() == [False, False, True, False, False, False]
    assert buf.truncated.tolist() == [False, False, False, False, True, False]
    assert buf.truncation_values[4] == 0.25 and buf.rewards[2] == 1.0
    # every stored action is legal under its own stored mask
    assert buf.masks[np.arange(6), buf.actions].all()
    assert buf.log_probs[3] == pytest.approx(-3.5) and buf.values[3] == pytest.approx(0.3)


def test_minibatches_cover_everything_once_and_stay_aligned() -> None:
    buf = RolloutBuffer(10, OBS, ACT)
    fill(buf, 10)
    buf.compute_advantages(0.0, 0.99, 0.95, normalize=False)
    seen = []
    for mb in buf.minibatches(4, np.random.default_rng(0)):
        seen.extend(mb.indices.tolist())
        for j, i in enumerate(mb.indices):
            assert mb.observations[j, 0, 0, 0] == i          # observation <-> index
            assert mb.actions[j] == i % ACT
            assert mb.masks[j, mb.actions[j]]                # mask <-> action
            assert mb.old_log_probs[j] == pytest.approx(-0.5 - i)
            assert mb.old_values[j] == pytest.approx(0.1 * i)
            assert mb.returns[j] == buf.returns[i] and mb.advantages[j] == buf.advantages[i]
    assert sorted(seen) == list(range(10))
    sizes = [len(mb.indices) for mb in buf.minibatches(4, np.random.default_rng(1))]
    assert sizes == [4, 4, 2]


def test_minibatches_are_copies() -> None:
    buf = RolloutBuffer(4, OBS, ACT)
    fill(buf, 4)
    buf.compute_advantages(0.0, 0.9, 0.9)
    mb = next(iter(buf.minibatches(4, np.random.default_rng(0))))
    mb.old_log_probs[:] = 99.0
    assert (buf.log_probs < 0).all()


def test_state_errors_and_reset() -> None:
    buf = RolloutBuffer(3, OBS, ACT)
    fill(buf, 2)
    with pytest.raises(BufferStateError):
        buf.compute_advantages(0.0, 0.9, 0.9)
    with pytest.raises(BufferStateError):
        next(buf.minibatches(2, np.random.default_rng(0)))
    fill_more = RolloutBuffer(3, OBS, ACT)
    fill(fill_more, 3)
    with pytest.raises(BufferStateError, match="advantages"):
        next(fill_more.minibatches(2, np.random.default_rng(0)))
    with pytest.raises(BufferStateError, match="full"):
        fill(fill_more, 1)
    fill_more.compute_advantages(0.0, 0.9, 0.9)
    fill_more.reset()
    assert len(fill_more) == 0 and not fill_more.is_full
    fill(fill_more, 3)  # reusable after reset
    with pytest.raises(BufferStateError):  # advantages are stale after reset
        next(fill_more.minibatches(2, np.random.default_rng(0)))


def test_add_validates_inputs() -> None:
    buf = RolloutBuffer(3, OBS, ACT)
    obs, mask = np.zeros(OBS, np.float32), np.zeros(ACT, bool)
    mask[2] = True
    with pytest.raises(ValueError, match="not legal"):
        buf.add(obs, mask, 5, 0.0, 0.0, 0.0, False, False)
    with pytest.raises(ValueError, match="both"):
        buf.add(obs, mask, 2, 0.0, 0.0, 0.0, True, True)
    with pytest.raises(ValueError, match="finite"):
        buf.add(obs, mask, 2, float("nan"), 0.0, 0.0, False, False)
    with pytest.raises(ValueError, match="observation shape"):
        buf.add(np.zeros((1, 3, 3), np.float32), mask, 2, 0.0, 0.0, 0.0, False, False)
    with pytest.raises(ValueError, match="mask"):
        buf.add(obs, mask.astype(np.float32), 2, 0.0, 0.0, 0.0, False, False)
    assert len(buf) == 0  # nothing was stored


def test_normalized_and_raw_advantages() -> None:
    buf = RolloutBuffer(8, OBS, ACT)
    fill(buf, 8, done_at={7: "T"})
    buf.compute_advantages(0.0, 0.99, 0.95, normalize=True)
    assert abs(buf.advantages.mean()) < 1e-5
    assert not np.allclose(buf.advantages, buf.raw_advantages)
    np.testing.assert_allclose(buf.returns, buf.raw_advantages + buf.values, atol=1e-6)
