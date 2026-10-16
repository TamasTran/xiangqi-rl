# XiangqiRL reinforcement-learning environment (Phase 2)

Package layout: `xiangqi_rl.env` (environment, encoders, rewards),
`xiangqi_rl.agents` (baselines), `xiangqi_rl.evaluation` (match runner).
Everything wraps the Phase 1 engine; no rules are re-implemented.

## 1. Interface

```python
from xiangqi_rl.env import XiangqiEnv

env = XiangqiEnv()                       # see constructor arguments below
obs, info = env.reset(seed=42)           # obs: float32 (17, 10, 9)
mask = env.action_masks()                # bool (8100,)  (also info["action_mask"])
obs, reward, terminated, truncated, info = env.step(action)
```

Constructor (all keyword-only):

| Argument | Default | Meaning |
|---|---|---|
| `game_config` | `GameConfig()` | Engine policy (ply limit 300, 3-fold repetition) |
| `max_episode_steps` | `None` | **External** step limit, causes `truncated` |
| `reward_perspective` | `None` | `None` = mover; `Side.RED` / `Side.BLACK` = fixed |
| `initial_fen` | `None` | Start every episode from this FEN |
| `render_mode` | `None` | `None` or `"ansi"` (text board via `render()`) |

`reset(options={"fen": ...})` starts one episode from a custom position.
`env.state` returns an immutable `GameState`; `env.game_copy()` returns an
independent engine `Game` (what agents receive); `env.side_to_move`,
`env.steps` are read-only. Spaces: `observation_space = Box(0, 1, (17,10,9),
float32)`, `action_space = Discrete(8100)`.

## 2. Turn handling: one ply per step

Each `step(action)` plays **exactly one legal move for the side to move**,
then the turn passes. The environment never plays an opponent move itself:
the caller (self-play trainer or match runner) supplies the actions of both
sides. Consequences:

* One step = one ply; `info["current_player"]` (0 = RED, 1 = BLACK) is the
  side that must move **next**; `info["mover"]` is the side that just moved
  (`None` after `reset`).
* A policy is evaluated as RED or BLACK simply by being asked to act when
  `current_player` matches; there is no hard-wired "agent" side.
* The trainer must store `current_player` alongside each observation and
  transform values/returns by perspective (below).

## 3. Observation, shape `(17, 10, 9)`, `float32`, values in `[0, 1]`

Board axes are **absolute**: row 0 = Black's back rank (top), row 9 = Red's
(bottom), column 0 = file `a`. The board is not mirrored for the side to move.

| Channel | Content |
|---|---|
| 0–6 | RED general, advisor, elephant, horse, chariot, cannon, soldier |
| 7–13 | BLACK general, advisor, elephant, horse, chariot, cannon, soldier |
| 14 | side to move: 1.0 everywhere if RED to move, else 0.0 |
| 15 | repetition: (occurrences of the position − 1) / (threshold − 1), clipped; 0 if disabled |
| 16 | ply progress: plies / engine `max_plies`, clipped; 0 if disabled |

`piece channel = 7 * side + piece_type` with `Side.RED = 0`, `BLACK = 1` and
`PieceType` order GENERAL, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON,
SOLDIER. Names are available as `CHANNEL_NAMES`. Encoding is deterministic, a
fresh array is returned each call, and positions with equal placement, side,
and the same history-derived planes encode identically. Channels 15–16 exist
because repetition and the ply limit can end the game.

**Partial observability.** The observation is *not* a full Markov state:
perpetual-check adjudication depends on which earlier moves gave check, and
the observation does not encode that (nor the exact move history). It is
exact with respect to the board, the side to move, and the distance to the repetition /
ply-limit rules.

## 4. Action space, `Discrete(8100)`

`action = 90 * from_sq + to_sq`, with `square = row * 9 + col` (row 0 at the
top). Helpers in `xiangqi_rl.env.action_space`: `square_to_index`,
`index_to_square`, `move_to_action`, `action_to_move`. Decoding accepts Python
or NumPy integers only: floats, bools, strings, out-of-range values and the
90 "null" actions (`from == to`) raise; nothing is coerced.

## 5. Legal-action mask

`env.action_masks()` (also `info["action_mask"]`) is `bool`, shape `(8100,)`,
`True` where the engine allows the move now. It is built from the engine's
legal-move list, so it is exactly the set of actions `step()` accepts. After
the game ends the mask is **all False** (no move is invented). The mask does
not replace validation: `step()` still checks the action itself.

## 6. Illegal actions

`step()` raises `IllegalActionError` (a `ValueError`) for an action that is
out of range, not an integer, or not legal. The game, step counter and
history are unchanged and the episode continues to be usable. Rationale: the
engine is the single source of truth, a custom PPO sampler is expected to
apply the mask (so an illegal action indicates a bug, which should be loud),
and silently turning it into a penalty or a loss would pollute returns.
Penalty-based handling can be added later as an explicit wrapper.

## 7. Rewards

Sparse and zero-sum: +1 win, −1 loss, 0 draw, 0 on non-terminal plies, and 0
on external truncation (no result is fabricated). Perspective is explicit:

* **Default (`reward_perspective=None`): the player who made this step's
  move.** `reward` is +1 if that move ended the game in the mover's favour
  (e.g. checkmate), −1 if it lost (e.g. the mover completes a perpetual
  check), 0 otherwise. Consecutive rewards therefore alternate perspective.
* `reward_perspective=Side.RED` or `Side.BLACK`: always from that side.

Value targets: a terminal reward `r` earned by the mover at step `t` is worth
`r` to the mover and `−r` to the opponent; `info["mover"]` identifies whose
perspective `reward` used. Never combine fixed-perspective rewards with
side-to-move value predictions without this sign flip.

## 8. Termination vs truncation

| Situation | `terminated` | `truncated` | Reward |
|---|---|---|---|
| Checkmate / no legal move / perpetual check | True | False | ±1 |
| Repetition draw, **engine** ply limit (`max_plies`) | True | False | 0 |
| `max_episode_steps` reached, game not over | False | True | 0 |
| Natural end on the same step as the external limit | True | False | per result |

The engine's `max_plies` (default 300) is a *rule* of the game (a draw); the
environment's `max_episode_steps` is an *external* cutoff that leaves the
engine result `IN_PROGRESS`. After either flag, `step()` raises
`EpisodeOverError` until `reset()`. `info` also carries `result`,
`termination_reason`, `engine_terminal` and `externally_truncated`.
Truncated episodes should be bootstrapped from the value estimate, not
treated as terminal.

## 9. Baseline agents

Both implement `name` and `select_action(game: Game) -> int` and never mutate
the game.

* `RandomAgent(seed)`: uniform over legal moves with a private
  `random.Random`; raises `NoLegalActionError` in a finished game.
* `HeuristicAgent(config, seed, tie_break)`: one-ply greedy. Score = value of
  the captured piece (default: general 100, chariot 9, cannon 4.5, horse 4,
  advisor 2, elephant 2, soldier 1) + small soldier-advance/river-crossing
  bonus + a large bonus for a move after which the opponent has no legal
  move. Ties: seeded random or smallest action. **It is not a strong
  engine**: it sees no recaptures, ignores threats against its own pieces,
  and is purely a yardstick.

* `LookaheadAgent(config, seed, tie_break)`: two-ply minimax with the same
  material values: for each move it assumes the opponent's best reply
  (largest capture or a mate) and plays the move whose worst case is best;
  a move after which the opponent has no legal move wins at once. It
  declines defended pieces, sees mate in one and avoids allowing mate in one.
  It also asks the engine whether its move, or the opponent's reply, would
  end the game by repetition and avoids losing that way
  (`avoid_repetition_loss=True`; set it to `False` for the blind version).
  Limits: no deeper tactics, stalemating replies are not detected, only the
  immediate effect of repetition is seen, ~6x slower than the heuristic
  (about 25 ms per move on the development CPU). In 40-game matches it beat
  both `random` and `heuristic` 40-0.

## 10. Evaluation

`run_match(agent_a, agent_b, n, seed=, swap_colors=True, game_config=,
max_episode_steps=)` returns `(MatchStats, [GameRecord])`. Game `i` is reset
with `seed + i`; `agent_a` is RED in even games when swapping. The runner
asks `env.side_to_move`'s agent for one action per step, so no agent moves
twice in a row. An illegal action counts in `invalid_actions` and forfeits
that game. Reported: games, RED/BLACK wins, draws, truncated (sum = games),
invalid actions, average plies, and per-agent and per-agent-per-colour
wins/losses/draws, win rate and score (draw/truncation = 0.5).

```bash
python scripts/run_baseline_tournament.py --games 20 --seed 0
```

A handful of games proves nothing about relative strength; use many games
and report the count with any claim.

## 11. Reproducibility

`reset(seed=...)` seeds `env.np_random` per Gymnasium convention; the engine
itself is deterministic and uses no randomness, so the same action sequence
always gives the same trajectory. Agents own their RNGs (seeded at
construction) and never touch the global `random`/`numpy` state.

## 12. Notes for Phases 3 and 4

* **Network input**: the `(17, 10, 9)` tensor; flip rows (and swap the Red /
  Black plane groups) if you prefer a side-to-move-relative view. Output
  heads: policy logits over 8100 actions (mask illegal ones before softmax)
  and a scalar value from the side-to-move's perspective.
* **PPO**: store `(obs, action, logp, mask, current_player)` each step; use
  the mover-perspective reward and negate when propagating returns across
  the opponent's plies. Treat `truncated` with bootstrapping.

## 13. Known limitations

* Partially observable w.r.t. perpetual-check history (section 3).
* Repetition rules are the simplified Phase 1 policy (no perpetual chase).
* Pure-Python engine: roughly 80k legal-move nodes/s, fine for baselines but
  a bottleneck for large-scale self-play.
* One environment instance, no vectorisation; run several instances for
  parallel rollouts.
