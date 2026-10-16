"""Train the policy-value network with PPO against the baseline agents.

Examples:
    python scripts/train_ppo.py --config configs/ppo_baseline.yaml
    python scripts/train_ppo.py --config configs/ppo_baseline.yaml --device cpu --smoke-test
    python scripts/train_ppo.py --config configs/ppo_baseline.yaml --set total_iterations=50
    python scripts/train_ppo.py --resume artifacts/ppo/latest.pt --set total_iterations=300
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import tempfile
from pathlib import Path

try:
    import xiangqi_rl  # noqa: F401
except ImportError:  # allow running from a checkout without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    sys.exit("PyTorch is required: python -m pip install -e '.[nn]'")

from xiangqi_rl.algorithms.ppo import (  # noqa: E402
    SMOKE_OVERRIDES,
    PPOConfig,
    PPOTrainer,
    load_config,
    parse_override,
)
from xiangqi_rl.engine import Game, GameConfig  # noqa: E402
from xiangqi_rl.env import encode_observation, legal_action_mask  # noqa: E402
from xiangqi_rl.nn import MaskedCategorical, load_checkpoint, load_model  # noqa: E402


def smoke_checks(trainer: PPOTrainer) -> None:
    """Reload the latest checkpoint, compare outputs, check legal sampling,
    and resume for one more iteration."""
    latest = trainer.output_dir / "latest.pt"
    reloaded = load_model(latest, map_location="cpu").eval()
    original = trainer.model.to("cpu").eval()
    rng = random.Random(0)
    game = Game.new_game(GameConfig(max_plies=60))
    checked = 0
    while not game.is_terminal() and checked < 30:
        obs = torch.from_numpy(encode_observation(game)).unsqueeze(0)
        mask = torch.from_numpy(legal_action_mask(game)).unsqueeze(0)
        with torch.no_grad():
            l1, v1 = original(obs)
            l2, v2 = reloaded(obs)
        assert torch.allclose(l1, l2, atol=1e-6) and torch.allclose(v1, v2, atol=1e-6)
        action = int(MaskedCategorical(l2, mask).sample().item())
        assert mask[0, action], "sampled an illegal action"
        game.make_move(rng.choice(game.legal_moves()))
        checked += 1
    print(f"smoke: checkpoint reload matches; {checked} sampled actions were legal")

    saved = load_checkpoint(latest)["extra"]
    cfg = PPOConfig.from_dict(saved["ppo_config"]).with_overrides(
        {"device": "cpu", "total_iterations": int(saved["iteration"]) + 1}
    )
    resumed = PPOTrainer.from_checkpoint(latest, config=cfg)
    before = resumed.iteration
    try:
        history = resumed.train()
    finally:
        resumed.close()
    assert history[0]["learning_rate"] > 0
    assert resumed.iteration == before + 1
    print(f"smoke: resumed from iteration {before} and completed one more update")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="configs/ppo_baseline.yaml", help="YAML config file")
    p.add_argument("--resume", help="resume from a training checkpoint (latest.pt)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], help="override config device")
    p.add_argument("--output-dir", help="override config output_dir")
    p.add_argument("--iterations", type=int, help="override total_iterations")
    p.add_argument("--smoke-test", action="store_true",
                   help="tiny model/rollout, 2 updates, eval, save+reload+resume checks")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config key (model fields as model.<name>); repeatable")
    args = p.parse_args()

    overrides: dict = {}
    if args.smoke_test:
        overrides.update(SMOKE_OVERRIDES)
    for text in args.overrides:
        key, value = parse_override(text)
        overrides[key] = value
    if args.device:
        overrides["device"] = args.device
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.iterations is not None:
        overrides["total_iterations"] = args.iterations
    temp_output = args.smoke_test and "output_dir" not in overrides
    if temp_output:
        overrides["output_dir"] = tempfile.mkdtemp(prefix="xiangqi_ppo_smoke_")

    if args.resume:
        base = PPOConfig.from_dict(
            torch.load(args.resume, map_location="cpu", weights_only=True)["extra"]["ppo_config"]
        )
        config = base.with_overrides(overrides) if overrides else base
        trainer = PPOTrainer.from_checkpoint(args.resume, config=config)
        print(f"resumed from {args.resume} at iteration {trainer.iteration}")
    else:
        config = load_config(args.config, overrides)
        trainer = PPOTrainer(config)

    print(f"device={trainer.device} output_dir={trainer.output_dir} "
          f"parameters={trainer.model.num_parameters():,}")
    try:
        trainer.train()
        print(f"finished at iteration {trainer.iteration}; checkpoints in {trainer.output_dir}")
        if args.smoke_test:
            smoke_checks(trainer)
            print("smoke test OK (verifies the pipeline, not playing strength)")
    finally:
        trainer.close()
    if temp_output:
        shutil.rmtree(trainer.output_dir, ignore_errors=True)  # only reached on success


if __name__ == "__main__":
    main()
