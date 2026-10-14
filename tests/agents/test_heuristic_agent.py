from __future__ import annotations

import pytest

from conftest import RED_MATE_IN_ONE
from xiangqi_rl.agents import HeuristicAgent, HeuristicConfig, NoLegalActionError
from xiangqi_rl.engine import Game, Move, PieceType, Side


def chosen(agent: HeuristicAgent, game: Game) -> str:
    return Move.from_action(agent.select_action(game)).uci()


def test_prefers_most_valuable_capture(make_fen) -> None:
    # Red chariot a0 can take the black chariot on a5 (9) or the horse on h0 (4).
    fen = make_fen({"e1": "K", "a0": "R", "a5": "r", "h0": "n", "f9": "k"})
    game = Game.from_fen(fen)
    assert {"a0a5", "a0h0"} <= {m.uci() for m in game.legal_moves()}
    for seed in range(5):
        assert chosen(HeuristicAgent(seed=seed), game) == "a0a5"


def test_capture_value_is_configurable(make_fen) -> None:
    fen = make_fen({"e1": "K", "a0": "R", "a5": "r", "h0": "n", "f9": "k"})
    game = Game.from_fen(fen)
    cfg = HeuristicConfig(piece_values={**HeuristicConfig().piece_values, PieceType.HORSE: 50.0})
    assert chosen(HeuristicAgent(cfg, seed=0), game) == "a0h0"


def test_takes_checkmate(make_fen) -> None:
    game = Game.from_fen(make_fen(RED_MATE_IN_ONE))
    assert chosen(HeuristicAgent(seed=0), game) == "i5i9"


def test_soldier_advancement_breaks_ties_forward(make_fen) -> None:
    # Nothing to capture: the only moves with a bonus are soldier advances.
    game = Game.from_fen(make_fen({"e0": "K", "f9": "k", "a3": "P"}))
    agent = HeuristicAgent(seed=0)
    assert chosen(agent, game) == "a3a4"


def test_river_crossing_scores_higher_than_plain_advance(make_fen) -> None:
    game = Game.from_fen(make_fen({"e0": "K", "f9": "k", "a4": "P", "c3": "P"}))
    scores = {Move.from_action(a).uci(): s for s, a in HeuristicAgent().score_moves(game)}
    assert scores["a4a5"] > scores["c3c4"] > scores["e0e1"] == 0.0


def test_only_legal_actions_and_no_mutation() -> None:
    agent = HeuristicAgent(seed=3)
    game = Game.new_game()
    for _ in range(60):
        if game.is_terminal():
            break
        fen, keys, legal = game.to_fen(), game.position_keys, game.legal_moves()
        action = agent.select_action(game)
        assert game.to_fen() == fen and game.position_keys == keys
        assert game.legal_moves() == legal
        move = Move.from_action(action)
        assert game.is_legal(move)
        game.make_move(move)


def test_tie_breaking_deterministic_and_seeded() -> None:
    game = Game.new_game()
    first = [chosen(HeuristicAgent(tie_break="first"), game) for _ in range(3)]
    assert len(set(first)) == 1
    assert chosen(HeuristicAgent(seed=4), game) == chosen(HeuristicAgent(seed=4), game)
    with pytest.raises(ValueError):
        HeuristicAgent(tie_break="nope")  # type: ignore[arg-type]


def test_terminal_position_raises(make_fen) -> None:
    game = Game.from_fen(make_fen(RED_MATE_IN_ONE))
    game.make_move("i5i9")
    with pytest.raises(NoLegalActionError):
        HeuristicAgent().select_action(game)
    assert game.side_to_move is Side.BLACK
