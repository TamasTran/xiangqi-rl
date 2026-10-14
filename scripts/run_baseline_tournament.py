"""Run a small baseline tournament (CPU only).

Example:
    python scripts/run_baseline_tournament.py --games 20 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import xiangqi_rl  # noqa: F401
except ImportError:  # allow running from a checkout without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xiangqi_rl.agents import HeuristicAgent, LookaheadAgent, RandomAgent  # noqa: E402
from xiangqi_rl.engine import GameConfig, Side  # noqa: E402
from xiangqi_rl.evaluation import MatchStats, run_match  # noqa: E402


def report(title: str, stats: MatchStats) -> None:
    print(f"\n=== {title} ===")
    print(
        f"games={stats.games}  RED wins={stats.red_wins}  BLACK wins={stats.black_wins}  "
        f"draws={stats.draws}  truncated={stats.truncated}  "
        f"invalid actions={stats.invalid_actions}  avg plies={stats.average_plies:.1f}"
    )
    for name, s in sorted(stats.by_agent.items()):
        print(
            f"  {name:<14} W{s.wins} D{s.draws} L{s.losses} T{s.truncated}  "
            f"win rate={s.win_rate:.2f}  score={s.score:.2f}"
        )
        for side in Side:
            c = stats.by_agent_color.get((name, side))
            if c:
                print(
                    f"    as {side.name:<5} games={c.games} W{c.wins} D{c.draws} L{c.losses} "
                    f"T{c.truncated} score={c.score:.2f}"
                )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--games", type=int, default=20, help="games per matchup")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-plies", type=int, default=200, help="engine ply limit (draw)")
    p.add_argument("--max-steps", type=int, default=None, help="external step limit (truncation)")
    p.add_argument("--no-swap", action="store_true", help="first agent is always RED")
    args = p.parse_args()

    config = GameConfig(max_plies=args.max_plies)
    kwargs = dict(
        seed=args.seed,
        swap_colors=not args.no_swap,
        game_config=config,
        max_episode_steps=args.max_steps,
    )

    stats, _ = run_match(
        RandomAgent(args.seed, name="random_a"), RandomAgent(args.seed + 1, name="random_b"),
        args.games, **kwargs,
    )
    report("random vs random", stats)

    stats, _ = run_match(
        HeuristicAgent(seed=args.seed), RandomAgent(args.seed + 2), args.games, **kwargs
    )
    report("heuristic vs random (colours alternate unless --no-swap)", stats)

    stats, _ = run_match(
        LookaheadAgent(seed=args.seed), RandomAgent(args.seed + 3), args.games, **kwargs
    )
    report("lookahead vs random", stats)

    stats, _ = run_match(
        LookaheadAgent(seed=args.seed), HeuristicAgent(seed=args.seed + 4), args.games, **kwargs
    )
    report("lookahead vs heuristic", stats)


if __name__ == "__main__":
    main()
