"""Evaluate a trained checkpoint against the baseline agents (no training).

Example:
    python scripts/evaluate_ppo.py --checkpoint artifacts/ppo/best.pt --games 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import xiangqi_rl  # noqa: F401
except ImportError:  # allow running from a checkout without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch  # noqa: F401
except ImportError:
    sys.exit("PyTorch is required: python -m pip install -e '.[nn]'")

from xiangqi_rl.algorithms.ppo import resolve_device  # noqa: E402
from xiangqi_rl.evaluation.ppo_evaluator import evaluate_against_baselines  # noqa: E402
from xiangqi_rl.nn import load_model  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True, help="PPO or Phase 3 model checkpoint")
    p.add_argument("--games", type=int, default=100, help="games per opponent (even)")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--opponents", nargs="+", default=["random", "heuristic", "lookahead"],
                   choices=["random", "heuristic", "lookahead"])
    p.add_argument("--sample", action="store_true", help="sample moves instead of greedy")
    p.add_argument("--max-plies", type=int, default=200, help="engine draw limit")
    args = p.parse_args()
    if args.games < 2 or args.games % 2:
        sys.exit("--games must be an even number >= 2 (both colours equally)")

    device = resolve_device(args.device)
    model = load_model(args.checkpoint, map_location=device)
    results = evaluate_against_baselines(
        model, tuple(args.opponents), args.games, args.seed, device,
        greedy=not args.sample, game_max_plies=args.max_plies,
    )
    print(f"checkpoint: {args.checkpoint}  device: {device}  seed: {args.seed}  "
          f"greedy: {not args.sample}")
    for name, r in results.items():
        print(f"\nvs {name}: games={r.games} W{r.wins} D{r.draws} L{r.losses} "
              f"truncated={r.truncated} win_rate={r.win_rate:.3f} draw_rate={r.draw_rate:.3f} "
              f"avg_plies={r.average_plies:.1f} illegal_actions={r.invalid_actions}")
        for color, c in r.by_color.items():
            print(f"  as {color:<5}: games={c.games} W{c.wins} D{c.draws} L{c.losses}")
    print("\nNote: with few games these numbers are noisy; compare checkpoints with the same seed.")


if __name__ == "__main__":
    main()
