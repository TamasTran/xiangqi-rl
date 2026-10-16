from __future__ import annotations

import csv

import pytest

torch = pytest.importorskip("torch")

from xiangqi_rl.algorithms.ppo import SMOKE_OVERRIDES, PPOConfig, PPOTrainer  # noqa: E402
from xiangqi_rl.algorithms.ppo.logger import CSVLogger  # noqa: E402
from xiangqi_rl.algorithms.ppo.trainer import _REASON_COLUMNS, EVAL_COLUMNS, reason_columns  # noqa: E402
from xiangqi_rl.engine import GameResult, Side  # noqa: E402
from xiangqi_rl.evaluation import GameRecord  # noqa: E402
from xiangqi_rl.evaluation.ppo_evaluator import outcome_reasons  # noqa: E402


def rec(red, black, result, termination, **kw) -> GameRecord:
    return GameRecord(red=red, black=black, result=result, plies=10, termination=termination, **kw)


def test_outcome_reasons_from_the_policys_point_of_view() -> None:
    records = [
        rec("ppo", "x", GameResult.RED_WINS, "checkmate"),
        rec("x", "ppo", GameResult.RED_WINS, "perpetual_check"),      # ppo (BLACK) loses
        rec("x", "ppo", GameResult.BLACK_WINS, "perpetual_check"),    # ppo wins: x repeated checks
        rec("ppo", "x", GameResult.DRAW, "max_plies"),
        rec("ppo", "x", GameResult.BLACK_WINS, "illegal_action", invalid_actions=1, forfeit_by=Side.RED),
        rec("ppo", "x", GameResult.IN_PROGRESS, "external_step_limit", truncated=True),
    ]
    assert outcome_reasons(records) == {
        "draw_max_plies": 1, "loss_illegal_action": 1, "loss_perpetual_check": 1,
        "truncated_external_step_limit": 1, "win_checkmate": 1, "win_perpetual_check": 1,
    }


def test_reason_columns_cover_everything_and_sum_to_games() -> None:
    cols = reason_columns({"win_checkmate": 3, "win_weird": 2, "loss_illegal_action": 1,
                           "draw_repetition": 4, "truncated_external_step_limit": 5})
    assert cols["win_checkmate"] == 3 and cols["win_other"] == 2 and cols["loss_other"] == 1
    assert cols["draw_repetition"] == 4 and cols["truncated_external"] == 5
    assert sum(cols.values()) == 15
    assert set(cols) <= set(EVAL_COLUMNS)


def test_eval_log_has_reason_columns_that_sum_to_games(tmp_path) -> None:
    cfg = PPOConfig().with_overrides({**SMOKE_OVERRIDES, "device": "cpu", "eval_interval": 1,
                                      "eval_games": 4, "total_iterations": 1,
                                      "output_dir": str(tmp_path / "run")})
    PPOTrainer(cfg).train()
    rows = list(csv.DictReader(open(tmp_path / "run" / "eval_log.csv")))
    assert rows
    for r in rows:
        total = sum(int(r[c]) for c in _REASON_COLUMNS)
        assert total == int(r["games"]) == 4


def test_logger_keeps_old_file_when_columns_change(tmp_path) -> None:
    path = tmp_path / "log.csv"
    CSVLogger(path, ["a", "b"]).log({"a": 1, "b": 2})
    same = CSVLogger(path, ["a", "b"])
    assert same.migrated_from is None
    same.log({"a": 3, "b": 4})
    new = CSVLogger(path, ["a", "b", "c"])
    assert new.migrated_from == tmp_path / "log.csv.old"
    new.log({"a": 5, "b": 6, "c": 7})
    old_rows = list(csv.DictReader(open(tmp_path / "log.csv.old")))
    assert [r["a"] for r in old_rows] == ["1", "3"]               # nothing lost
    new_rows = list(csv.DictReader(open(path)))
    assert new_rows == [{"a": "5", "b": "6", "c": "7"}]
    third = CSVLogger(path, ["x"])                                # a second migration
    assert third.migrated_from == tmp_path / "log.csv.old2"
