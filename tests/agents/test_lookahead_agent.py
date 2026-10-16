from __future__ import annotations

import random

import pytest

from conftest import RED_MATE_IN_ONE
from xiangqi_rl.agents import HeuristicAgent, LookaheadAgent, NoLegalActionError
from xiangqi_rl.agents.heuristic_agent import DEFAULT_PIECE_VALUES
from xiangqi_rl.agents.lookahead_agent import MATE_SCORE
from xiangqi_rl.engine import Game, GameConfig, Move, PieceType
from xiangqi_rl.engine.rules import generate_legal_moves, has_legal_move, is_in_check

# Red chariot a0 can take the black horse on a5, but the black chariot on a9
# recaptures (gain 4, lose 9).
POISONED = {"e0": "K", "a0": "R", "a5": "n", "a9": "r", "f9": "k"}


def chosen(agent, game: Game) -> str:
    return Move.from_action(agent.select_action(game)).uci()


def naive_two_ply_scores(game: Game) -> dict[str, float]:
    """Independent, unpruned minimax over (move, reply) with material only."""
    value = lambda p: 0.0 if p is None or p.kind is PieceType.GENERAL else DEFAULT_PIECE_VALUES[p.kind]  # noqa: E731
    side, opp = game.side_to_move, game.side_to_move.opponent
    out = {}
    for move in game.legal_moves():
        gain = value(game.board.piece_at(move.to_sq))
        piece = game.board.piece_at(move.from_sq)
        if piece.kind is PieceType.SOLDIER:
            gain += LookaheadAgent()._soldier_bonus(side, move.from_sq, move.to_sq)
        after = game.board.apply_move(move)
        replies = generate_legal_moves(after, opp)
        if not replies:
            out[move.uci()] = MATE_SCORE
            continue
        worst = min(gain - value(after.piece_at(r.to_sq)) for r in replies)
        for r in replies:
            b = after.apply_move(r)
            if is_in_check(b, side) and not has_legal_move(b, side):
                worst = -MATE_SCORE
        out[move.uci()] = worst
    return out


def test_declines_a_defended_piece_that_the_heuristic_grabs(make_fen) -> None:
    game = Game.from_fen(make_fen(POISONED))
    assert chosen(HeuristicAgent(seed=0), game) == "a0a5"
    for seed in range(3):
        assert chosen(LookaheadAgent(seed=seed), game) != "a0a5"


def test_takes_mate_in_one(make_fen) -> None:
    game = Game.from_fen(make_fen(RED_MATE_IN_ONE))
    assert chosen(LookaheadAgent(seed=0), game) == "i5i9"


def test_matches_unpruned_minimax_and_never_allows_avoidable_mate() -> None:
    rng = random.Random(42)
    checked = 0
    for game_index in range(2):
        game = Game.new_game(GameConfig(max_plies=120))
        while not game.is_terminal():
            if game.ply % 7 == 3:
                naive = naive_two_ply_scores(game)
                move = chosen(LookaheadAgent(seed=game_index, avoid_repetition_loss=False), game)
                assert naive[move] == pytest.approx(max(naive.values())), game.to_fen()
                # If some move avoids an immediate mate, the chosen one does too.
                if max(naive.values()) > -MATE_SCORE:
                    assert naive[move] > -MATE_SCORE
                checked += 1
            game.make_move(rng.choice(game.legal_moves()))
    assert checked >= 10


# Positions found in random games where most moves allow the opponent an
# immediate checkmate and the one-ply heuristic plays one of them.
MATE_THREAT_FENS = [
    "2b2ab2/4Ck3/2r6/n8/2p6/2c5p/1c4r2/N1p4CB/p3A1N2/3AK4 w - - 8 74",
    "1nb2a3/4a4/4k1n2/8p/2p3b1C/2P3P2/3c4P/4p2rN/3K2C1R/2B2AB2 w - - 6 38",
    "1n2kabn1/5c3/b4a2r/p3p3P/2p1P4/P1P3BpR/1cC6/2N2A3/5K1C1/R1B2A1r1 w - - 11 45",
    "r1bak1b2/4r4/3a5/3np3P/6p2/R1P6/6n2/3A1K2C/1R7/5C1N1 w - - 2 58",
]


@pytest.mark.parametrize("fen", MATE_THREAT_FENS)
def test_avoids_moves_that_allow_mate_in_one(fen) -> None:
    game = Game.from_fen(fen)
    naive = naive_two_ply_scores(game)
    unsafe = {m for m, s in naive.items() if s <= -MATE_SCORE}
    assert unsafe and len(unsafe) < len(naive)
    assert chosen(HeuristicAgent(seed=0), game) in unsafe   # the 1-ply baseline falls for it
    for seed in range(5):
        assert chosen(LookaheadAgent(seed=seed), game) not in unsafe


# Red rook checks every move (a8-a9, a9-a8) while Black shuffles its king. Playing
# a9a8 after six plies lets Black's e8e9 repeat the start position a third time,
# and the engine ends the game: Red, the only side that checked, loses.
PERPETUAL = {"d0": "K", "a8": "R", "e9": "k"}
CYCLE = ["a8a9", "e9e8", "a9a8", "e8e9"]


def perpetual_position(make_fen) -> Game:
    game = Game.from_fen(make_fen(PERPETUAL))
    for move in CYCLE + CYCLE[:2]:
        game.make_move(move)
    assert game.ply == 6 and not game.is_terminal()
    return game


def test_avoids_completing_a_perpetual_check_loss(make_fen) -> None:
    game = perpetual_position(make_fen)
    trial = game.copy()
    trial.make_move("a9a8")
    trial.make_move("e8e9")      # the engine really ends the game here, against Red
    assert trial.termination_reason.value == "perpetual_check" and trial.winner.name == "BLACK"

    aware = {chosen(LookaheadAgent(seed=s), game) for s in range(15)}
    assert "a9a8" not in aware
    blind = {chosen(LookaheadAgent(seed=s, avoid_repetition_loss=False), game) for s in range(15)}
    assert "a9a8" in blind       # without the awareness the move is picked (all moves tie)


def test_repetition_awareness_is_a_noop_without_repetition() -> None:
    game = Game.new_game()
    for seed in range(3):
        assert (chosen(LookaheadAgent(seed=seed), game)
                == chosen(LookaheadAgent(seed=seed, avoid_repetition_loss=False), game))


def test_only_legal_moves_and_no_mutation() -> None:
    agent = LookaheadAgent(seed=1)
    game = Game.new_game(GameConfig(max_plies=60))
    while not game.is_terminal():
        fen, keys = game.to_fen(), game.position_keys
        move = Move.from_action(agent.select_action(game))
        assert game.to_fen() == fen and game.position_keys == keys
        assert game.is_legal(move)
        game.make_move(move)


def test_seeded_tie_breaking_and_rng_state() -> None:
    game = Game.new_game()
    assert chosen(LookaheadAgent(seed=4), game) == chosen(LookaheadAgent(seed=4), game)
    first = LookaheadAgent(tie_break="first")
    assert len({chosen(first, game) for _ in range(3)}) == 1
    a = LookaheadAgent(seed=9)
    state = a.rng_state
    picks = [chosen(a, game) for _ in range(5)]
    a.rng_state = state
    assert [chosen(a, game) for _ in range(5)] == picks
    with pytest.raises(ValueError):
        LookaheadAgent(tie_break="nope")  # type: ignore[arg-type]


def test_terminal_position_raises(make_fen) -> None:
    game = Game.from_fen(make_fen(RED_MATE_IN_ONE))
    game.make_move("i5i9")
    with pytest.raises(NoLegalActionError):
        LookaheadAgent().select_action(game)


def test_available_as_baseline_and_ppo_opponent() -> None:
    from xiangqi_rl.evaluation.ppo_evaluator import make_baseline

    assert isinstance(make_baseline("lookahead", 0), LookaheadAgent)
    try:
        from xiangqi_rl.algorithms.ppo import PPOConfig
    except ImportError:  # torch not installed
        return
    cfg = PPOConfig(opponents=("lookahead",), eval_opponents=("random", "lookahead"))
    assert cfg.opponents == ("lookahead",)
