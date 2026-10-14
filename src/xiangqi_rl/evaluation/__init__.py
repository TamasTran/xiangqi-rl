"""Match running and statistics for baseline evaluation."""

from xiangqi_rl.evaluation.match_runner import play_game, run_match
from xiangqi_rl.evaluation.metrics import ColorStats, GameRecord, MatchStats, summarize

__all__ = [
    "ColorStats",
    "GameRecord",
    "MatchStats",
    "play_game",
    "run_match",
    "summarize",
]
