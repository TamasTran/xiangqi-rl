# XiangqiRL

XiangqiRL is a reinforcement-learning project that will train an agent to play
**Xiangqi** (Chinese chess). Most of the system is written from scratch,
starting with the game engine, rather than wrapping an existing engine.

## Current status: Phases 1-4 (engine, RL environment + baselines, network, PPO)

Phase 1 delivers a complete, tested Xiangqi rules engine in pure Python.
Phase 2 adds a Gymnasium environment, observation/action encoders, a sparse
zero-sum reward, a random and a heuristic baseline agent, and a match runner
(see [docs/rl_environment.md](docs/rl_environment.md)). Runtime dependencies
are `numpy` and `gymnasium`; the engine itself needs neither.

Phase 3 adds an (untrained) PyTorch policy-value network with legal-action
masking and checkpointing; see
[docs/policy_value_network.md](docs/policy_value_network.md). PyTorch is an
optional extra: `python -m pip install -e ".[dev,nn]"`.

Phase 4 adds PPO written from scratch (GAE, clipped objective, masked
policy, rollout buffer, evaluation, resumable checkpoints); see
[docs/ppo_training.md](docs/ppo_training.md).

**Implemented**

* 10×9 board, standard initial position, FEN import/export.
* Movement for all seven piece types: palace, river, elephant eye, horse leg,
  and cannon screen rules.
* Pseudo-legal move generation, plus legal-move filtering that handles pins,
  check evasion, and the flying-general rule.
* Attack detection by reverse lookup (pinned pieces still attack).
* Check, checkmate, and no-legal-move (stalemate) losses.
* Configurable maximum-ply draw.
* Deterministic, simplified repetition policy: perpetual check loses, other
  threefold repetitions draw.
* `Game` API with history, capture records, undo, reset, clone, immutable
  state snapshots, and a deterministic position key.
* Move ↔ action encoding (`action = 90 * from_sq + to_sq`) for later RL use.
* A pytest suite that includes perft reference counts and an independent
  brute-force cross-check.

The exact rules, coordinate convention, and policies are documented in
[docs/xiangqi_rules.md](docs/xiangqi_rules.md).

**Phase 2 (implemented)**

* `XiangqiEnv`: one ply per `step`, `terminated` vs `truncated`, legal-action
  mask, `(17, 10, 9)` float32 observation, `Discrete(8100)` actions,
  rewards from the mover's perspective (or a fixed side).
* `RandomAgent` and a one-ply material `HeuristicAgent` (a yardstick, **not**
  a strong engine), plus `run_match` and a tournament script.

**Phase 3 (implemented)**

* `xiangqi_rl.nn`: residual CNN backbone, policy head (8100 logits) and
  value head (side-to-move perspective, tanh), `MaskedCategorical` for
  legal-action sampling / log-probabilities / entropy, safe checkpoints.
* `scripts/smoke_test_model.py`. The model is untrained: it has not learned
  to play Xiangqi.

**Phase 4 (implemented)**

* `xiangqi_rl.algorithms.ppo`: PPO trainer against the random/heuristic
  baselines (trained side alternates RED/BLACK), fixed-perspective wrapper,
  GAE with termination/truncation handling, CSV metrics, periodic
  evaluation, `latest.pt`/`best.pt` and resume.
* `scripts/train_ppo.py`, `scripts/evaluate_ppo.py`, `configs/ppo_baseline.yaml`.
* The tests verify the implementation; **no claim is made that the agent
  has learned to play well** (no long training run has been performed).

**Deliberately not implemented yet** (planned for later phases)

* Self-play league / advanced self-play.
* MCTS / AlphaZero-style search.
* Pygame user interface.
* Full official repetition rules (perpetual *chase*) and a no-capture
  natural move limit. See "Known limitations" in the rules document.

## Installation

Requires Python 3.10 or newer. If `python` is not found, use `python3`
(or `uv venv --seed .venv` where the system venv module lacks pip).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The only development dependency is `pytest`.

## Running the tests

```bash
pytest -q
```

The default run takes a few seconds. One long check, perft depth 4 (about
45 s), is marked `slow` and skipped by default:

```bash
pytest -m slow
```

## Quick example

```python
from xiangqi_rl.engine import Game, Move, Side

game = Game.new_game()
print(game.board)                      # ASCII diagram, Red at the bottom
print(game.side_to_move.name)          # RED
print(len(game.legal_moves()))         # 44
print([m.uci() for m in game.legal_moves()][:5])

game.make_move("h2e2")                 # central cannon, UCCI notation
game.make_move(Move.from_uci("h9g7"))  # or pass a Move object
print(game.is_in_check(Side.BLACK))    # False
print(game.get_result())               # GameResult.IN_PROGRESS
print(game.to_fen())

clone = game.copy()                    # independent copy for simulations
action = game.legal_moves()[0].to_action()   # 90 * from_sq + to_sq
```

Squares are named with a file letter `a`–`i` (left to right from Red's view)
and a rank `0`–`9` counted from Red's back rank. Internally a square is the
integer `row * 9 + col`, with row 0 at the top (Black's side).

## Baseline tournament and environment example

```bash
python scripts/run_baseline_tournament.py --games 20 --seed 0
```

```python
from xiangqi_rl.env import XiangqiEnv
from xiangqi_rl.agents import RandomAgent

env, agent = XiangqiEnv(), RandomAgent(seed=0)
obs, info = env.reset(seed=42)
terminated = truncated = False
while not (terminated or truncated):
    action = agent.select_action(env.game_copy())   # or sample from info["action_mask"]
    obs, reward, terminated, truncated, info = env.step(action)
print(info["result"], info["termination_reason"], env.steps)
```

## Playing in the terminal

A small text client lets you try the engine by hand (it is a convenience
script, not the Phase 4 UI):

```bash
python scripts/play_cli.py                  # two humans
python scripts/play_cli.py --vs-random      # you are Red vs. a random-move opponent
```

Enter moves such as `h2e2`; type `help` for `moves`, `undo`, `fen`, `history`, `reset`.

## Project layout

```
src/xiangqi_rl/engine/
    constants.py       sides, piece types, results, coordinates
    errors.py          exception hierarchy
    pieces.py          Piece model and FEN symbols
    move.py            Move, MoveRecord, action encoding
    board.py           immutable Board, FEN placement, position validation
    move_generator.py  pseudo-legal moves per piece
    rules.py           attack detection, check, legal-move filtering
    repetition.py      isolated repetition adjudication policy
    game.py            Game API, terminal-state policy, history
src/xiangqi_rl/env/         Gymnasium env, observation, action space, rewards
src/xiangqi_rl/agents/      RandomAgent, HeuristicAgent
src/xiangqi_rl/evaluation/  match runner and metrics
src/xiangqi_rl/nn/          policy-value network, masking, checkpoints (needs torch)
src/xiangqi_rl/algorithms/ppo/  PPO: config, buffer, GAE, losses, wrapper, trainer
configs/ppo_baseline.yaml   PPO experiment configuration
scripts/train_ppo.py, scripts/evaluate_ppo.py
scripts/run_baseline_tournament.py
scripts/smoke_test_model.py
scripts/play_cli.py      terminal client for trying the engine by hand
tests/                 pytest suites: engine/, env/, agents/, evaluation/, nn/, algorithms/ppo/
docs/xiangqi_rules.md  rules, conventions, policies, limitations
docs/rl_environment.md RL environment, rewards, agents, interfaces for Phases 3-4
docs/policy_value_network.md  network architecture, masking, value perspective, checkpoints
docs/ppo_training.md   PPO objective, GAE, perspective, evaluation, resume
```
