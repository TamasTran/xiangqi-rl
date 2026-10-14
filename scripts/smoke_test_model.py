"""Smoke test for the policy-value network (CPU; uses CUDA if available).

Checks tensor shapes, parameter count, legal-action masking, one backward
pass and one optimizer step. Synthetic observations only test plumbing; they
say nothing about playing strength (the model is untrained).

Usage:
    python scripts/smoke_test_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import xiangqi_rl  # noqa: F401
except ImportError:  # allow running from a checkout without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    sys.exit("PyTorch is required: python -m pip install -e '.[nn]'")

from xiangqi_rl.env import XiangqiEnv  # noqa: E402
from xiangqi_rl.nn import MaskedCategorical, ModelConfig, create_model  # noqa: E402


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig()
    model = create_model(cfg, seed=0).to(device)
    print(f"device: {device}")
    print(f"config: {cfg}")
    print(f"trainable parameters: {model.num_parameters():,}")

    batch = 4
    obs = torch.randn(batch, *cfg.observation_shape, device=device)
    logits, value = model(obs)
    print(f"policy logits shape: {tuple(logits.shape)}  value shape: {tuple(value.shape)}")
    assert logits.shape == (batch, cfg.action_dim) and value.shape == (batch, 1)

    mask = torch.zeros_like(logits, dtype=torch.bool)
    for row, legal in enumerate(([0, 5, 90], [7], [1, 2, 3, 4], [8099, 100])):
        mask[row, legal] = True
    dist = MaskedCategorical(logits, mask)
    print(f"masked prob sums: {dist.probs.sum(-1).tolist()}")
    print(f"illegal probability mass: {dist.probs[~mask].sum().item()}")
    print(f"sampled legal actions: {dist.sample().tolist()}")

    loss = logits.logsumexp(-1).mean() + value.pow(2).mean()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    print(f"backward pass: loss={loss.item():.4f} finite gradients={grad_ok}")
    assert grad_ok

    # Optional: a real initial position from the Phase 2 environment.
    env = XiangqiEnv()
    real_obs, info = env.reset(seed=0)
    with torch.no_grad():
        r_logits, r_value = model(torch.from_numpy(real_obs).unsqueeze(0).to(device))
    r_mask = torch.from_numpy(info["action_mask"]).unsqueeze(0).to(device)
    action = int(MaskedCategorical(r_logits, r_mask).sample().item())
    env.step(action)
    print(f"real start position: {int(r_mask.sum())} legal actions, "
          f"value={r_value.item():+.4f} (untrained), sampled action {action} accepted by env")
    print("OK")


if __name__ == "__main__":
    main()
