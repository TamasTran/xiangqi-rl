"""Evaluate a policy-value network against the baseline agents.

Evaluation never touches PPO rollouts: it runs complete games through the
Phase 2 match runner with the model in ``eval`` mode and no gradients.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from xiangqi_rl.agents import HeuristicAgent, LookaheadAgent, RandomAgent
from xiangqi_rl.engine import Game, GameConfig, Side
from xiangqi_rl.env import encode_observation, legal_action_mask
from xiangqi_rl.evaluation.match_runner import run_match
from xiangqi_rl.evaluation.metrics import ColorStats
from xiangqi_rl.nn import MaskedCategorical

POLICY_NAME = "ppo"


class PolicyAgent:
    """Adapter exposing a policy-value network as an :class:`Agent`.

    Picks the most probable legal move (``greedy``) or samples from the masked
    policy with its own seeded generator.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str = "cpu",
        greedy: bool = True,
        seed: int | None = 0,
        name: str = POLICY_NAME,
    ) -> None:
        self.name = name
        self.model = model
        self.device = torch.device(device)
        self.greedy = greedy
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(0 if seed is None else seed)

    @torch.no_grad()
    def select_action(self, game: Game) -> int:
        obs = torch.from_numpy(encode_observation(game)).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(legal_action_mask(game)).unsqueeze(0).to(self.device)
        logits, _ = self.model(obs)
        dist = MaskedCategorical(logits, mask)
        action = dist.mode() if self.greedy else dist.sample(self._generator)
        return int(action.item())


@dataclass(frozen=True)
class EvalResult:
    """Outcome of ``games`` games of the policy against one opponent.

    ``draws`` also counts nothing else: externally truncated games are
    reported in ``truncated`` (neither win nor loss nor draw).
    """

    opponent: str
    games: int
    wins: int
    draws: int
    losses: int
    truncated: int
    invalid_actions: int
    average_plies: float
    by_color: dict[str, ColorStats] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games if self.games else 0.0

    @property
    def score(self) -> float:
        """Win = 1, draw/truncation = 0.5, loss = 0."""
        return (self.wins + 0.5 * (self.draws + self.truncated)) / self.games if self.games else 0.0


def make_baseline(name: str, seed: int):
    """Construct a seeded baseline opponent by name."""
    if name == "random":
        return RandomAgent(seed=seed, name="random")
    if name == "heuristic":
        return HeuristicAgent(seed=seed, name="heuristic")
    if name == "lookahead":
        return LookaheadAgent(seed=seed, name="lookahead")
    raise ValueError(f"unknown opponent {name!r}")


def evaluate_against_baselines(
    model: nn.Module,
    opponents: tuple[str, ...] = ("random", "heuristic"),
    games: int = 20,
    seed: int = 12345,
    device: torch.device | str = "cpu",
    greedy: bool = True,
    game_max_plies: int | None = 200,
    max_episode_plies: int | None = None,
) -> dict[str, EvalResult]:
    """Play ``games`` games per opponent, alternating colours.

    The same ``seed`` gives the same opponents' behaviour and game seeds, so
    results of different checkpoints are comparable. The model's train/eval
    mode is restored afterwards.
    """
    was_training = model.training
    model.eval()
    results: dict[str, EvalResult] = {}
    try:
        for name in opponents:
            policy = PolicyAgent(model, device, greedy=greedy, seed=seed)
            stats, _ = run_match(
                policy,
                make_baseline(name, seed),
                games,
                seed=seed,
                swap_colors=True,
                game_config=GameConfig(max_plies=game_max_plies),
                max_episode_steps=max_episode_plies,
            )
            mine = stats.by_agent.get(POLICY_NAME, ColorStats())
            results[name] = EvalResult(
                opponent=name,
                games=stats.games,
                wins=mine.wins,
                draws=mine.draws,
                losses=mine.losses,
                truncated=mine.truncated,
                invalid_actions=stats.invalid_actions,
                average_plies=stats.average_plies,
                by_color={
                    side.name.lower(): stats.by_agent_color.get((POLICY_NAME, side), ColorStats())
                    for side in Side
                },
            )
    finally:
        model.train(was_training)
    return results
