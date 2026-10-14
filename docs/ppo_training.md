# PPO training (Phase 4)

Code: `src/xiangqi_rl/algorithms/ppo/`, evaluation in
`src/xiangqi_rl/evaluation/ppo_evaluator.py`, entry points
`scripts/train_ppo.py` and `scripts/evaluate_ppo.py`, config
`configs/ppo_baseline.yaml`. Requires `pip install -e ".[dev,nn]"`
(PyTorch + PyYAML).

> **What the tests prove.** Passing tests show that the implementation
> behaves as specified: GAE matches hand calculations, masks are consistent,
> losses/gradients are finite, checkpoints restore state. They do **not**
> show that the agent has learned to play Xiangqi. Whether training improves
> play can only be judged from evaluation against the baselines over many
> games, and no long training run was performed while developing this phase.

## 1. Data flow

```
OpponentEnv (XiangqiEnv + baseline opponent)
   -> observation [17,10,9] + legal-action mask [8100]
   -> PolicyValueNet -> logits, value
   -> MaskedCategorical(logits, mask) -> sample action, log pi_old(a|s), V(s)
   -> OpponentEnv.step(action)  (agent move, then opponent replies)
   -> RolloutBuffer (obs, mask, action, logp, value, reward, terminated, truncated, ...)
   -> GAE -> advantages, returns
   -> PPO epochs over shuffled minibatches (same stored masks)
   -> updated policy -> buffer.reset() -> next rollout
```

## 2. The PPO objective

With `r_t = exp(log pi_theta(a_t|s_t) - log pi_old(a_t|s_t))` and advantage
estimate `A_t`:

    L_clip = E[ min( r_t A_t,  clip(r_t, 1-eps, 1+eps) A_t ) ]
    L      = -L_clip + c_v * L_value - c_e * H(pi_theta)

* The `min` makes the objective pessimistic: moving the ratio outside
  `[1-eps, 1+eps]` in the direction that the advantage favours earns no
  further gain (and no gradient), while moving it the wrong way is still
  penalised.
* `L_value = 0.5 * mean((V_theta - R)^2)`; with `value_clip_coef` set, the
  PPO value clipping `max((V-R)^2, (V_old + clip(V-V_old, +-e) - R)^2)` is
  used instead.
* `H` is the entropy of the *masked* policy over legal moves, in nats.
* Diagnostics: `approx_kl = mean((r-1) - log r)` (k3 estimator),
  `clip_fraction = mean(|r-1| > eps)`.

Implemented in `losses.py`; every formula has a hand-computed unit test.

## 3. GAE and returns (`returns.py`)

    next_value_t = 0                      natural termination
                 = V(final obs)           external truncation (bootstrapped)
                 = V(s_{t+1})             otherwise; last_value at the rollout edge
    delta_t = r_t + gamma * next_value_t - V(s_t)
    A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    R_t     = A_t + V(s_t)

`(1 - done_t)` stops the recursion at episode boundaries. Advantages can be
normalised over the whole rollout (`normalize_advantages`); the un-normalised
values are kept for logging. `gamma` is per *decision* of the trained agent.

## 4. Legal-action masking

Rollout (`distributions.act`) and update (`distributions.evaluate_actions`)
both construct `MaskedCategorical(logits, mask)`; the buffer stores the mask
used at rollout time and the update re-uses exactly that mask. Illegal
actions get probability exactly 0, raw logits are never used as
log-probabilities, `log_prob` raises if an action is illegal under the mask,
and an all-False mask (terminal state) raises `EmptyLegalMaskError`. Terminal
observations are never passed to the policy: the wrapper resets
immediately, and the final observation is only used for truncation
bootstrapping (a value, no policy).

## 5. Perspective, turns and the training wrapper

The base environment alternates players every ply, so consecutive rewards
and values have alternating perspectives and plain GAE would be wrong.
`OpponentEnv` solves this for training without changing `XiangqiEnv`:

* The trained agent plays one side; one baseline (random, heuristic or
  lookahead, drawn per episode from `opponents`) plays the other.
* `step(action)` plays the agent's move, then the opponent's replies, until
  it is the agent's turn again or the game ends. Every stored transition is
  therefore *agent decision -> next agent decision*, with a single, fixed
  perspective: the trained agent's. The network's side-to-move value equals
  the agent's value at every decision point.
* Reward: `+1` win, `-1` loss, `0` draw, `0` everywhere else, given at the
  end of the game, from the agent's side. If the *opponent's* move ends the
  game, the episode ends on that step with the corresponding reward.
* Optional **material shaping** (`material_shaping_coef`, see section 5a) adds
  a dense term to that reward. `info["game_reward"]` always keeps the unshaped
  result, and wins/draws/losses in the logs are computed from it.
* Colours: `color_schedule: alternate` plays RED in even episodes and BLACK
  in odd ones (also `red`, `black`, `random`). When the agent is BLACK the
  opponent makes the first move during `reset()`.

## 5a. Potential-based material shaping

With only the final result as reward, games against the random baseline
almost always end in a draw by the ply limit (reward 0) and games against the
heuristic in a loss many moves after the mistake, so there is little learning
signal. A 400-iteration run with the sparse reward kept the mean episode
reward flat at about -0.5 (lose to heuristic, draw with random).

`material_shaping_coef > 0` adds, for every agent transition `s -> s'`
(decision to next decision, so including the opponent's reply):

    F = gamma * Phi(s') - Phi(s)
    Phi(s) = coef * (material(agent) - material(opponent)) / 48,   Phi(terminal) = 0

with chariot 9, cannon 4.5, horse 4, advisor 2, elephant 2, soldier 1 (48 =
one side's starting material without the general). Winning material gives an
immediate positive reward, losing it a negative one.

Why potential-based: the discounted shaped return of an episode equals the
original one minus `Phi(s_0)` (Ng, Harada & Russell, 1999), so the optimal
policy is unchanged; the shaping cannot make "grab material and never
checkmate" better than winning. The tests check this telescoping identity
for terminated episodes with gamma 1 and 0.9 and both colours. `gamma` in
`F` is the PPO `gamma` (per decision). A truncated episode is not terminal:
`Phi` of the final observation is used and its value is bootstrapped.

Shaped values lie in `[-1 - coef, 1 + coef]`, outside tanh's range, so the
config requires `model.value_activation: none` when shaping is on. The
material values are a heuristic: the policy is still optimal for the true
reward, but learning is biased toward material in the short run.

`train_log.csv` reports both `episode_reward` (game result only) and
`episode_shaped_return` (undiscounted sum of the training rewards).

## 6. Termination vs truncation

| Event | `terminated` | `truncated` | Reward | GAE bootstrap |
|---|---|---|---|---|
| checkmate / no legal move / perpetual check | yes | no | +-1 | none |
| engine draw rule (`game_max_plies`, repetition) | yes | no | 0 | none |
| external limit (`max_episode_plies`) | no | yes | 0 | `V(final obs)` |

The engine draw is a *rule* and its progress is visible to the network (ply
and repetition planes), so it is treated as a terminal state. The external
limit is not a property of the game, so the value of the final position is
bootstrapped. Because the opponent may not have replied yet when the limit
hits, the final observation can have the *opponent* to move; the stored
bootstrap value is sign-flipped in that case so it stays in the agent's
perspective. By default `max_episode_plies` is `null` (truncation is
implemented and tested but unused).

## 7. Training loop

Per iteration: (1) collect `rollout_steps` decisions with the frozen
behaviour policy under `torch.no_grad()` (episodes may span rollouts);
(2) compute GAE with `last_value = V(current obs)`; (3) for `update_epochs`
epochs, iterate shuffled minibatches, recompute log-probs / entropy / values
with the stored masks, form the loss, `zero_grad` -> `backward` ->
`clip_grad_norm_` -> `step`; optional early stop on `target_kl`;
(4) scheduler step, logging, evaluation, checkpoint. The buffer is cleared
and refilled every iteration: data is never reused across policies.

A non-finite loss or gradient norm raises `NonFiniteError` *before* the
optimizer step. `norm: batch` is rejected for PPO (rollout and update would
use different statistics); the default GroupNorm is batch-independent.

## 8. Evaluation

`evaluation/ppo_evaluator.py` plays complete games between the current
network (greedy by default; `eval_greedy: false` samples) and each baseline,
alternating colours, through the Phase 2 match runner. It never touches
rollout data or the training RNGs, restores the model's train/eval mode, and
uses fixed seeds (`eval_seed`) so checkpoints are comparable. Reported per
opponent: games, wins/draws/losses/truncated, win rate, draw rate, average
plies, illegal-action count (an illegal action forfeits the game) and the
per-colour breakdown. Evaluation runs before training (iteration 0) and every
`eval_interval` iterations. The "best" checkpoint is the one with the highest
mean score (win = 1, draw = 0.5) over the evaluation opponents. With the
default 20 games per opponent the numbers are noisy; use `evaluate_ppo.py`
with many more games (hundreds) before drawing conclusions.

### Why the lookahead opponent exists

A 1000-iteration run (shaping 0.5) reached ~90% wins against the one-ply
heuristic, but games were only ~17 plies long, Black won 20/20 and half of the
games repeated one opening trap: the policy exploited the heuristic's
blindness to recaptures and mate threats. Re-evaluated against the two-ply
`lookahead` baseline, the same checkpoints won only 2-4 of 40 games. The
baseline config therefore trains and evaluates against all three opponents.
`lookahead` makes each opponent move ~6x slower than `heuristic`, so
iterations take longer.

## 9. Metrics and files

`<output_dir>/train_log.csv` (one row per iteration): policy/value loss,
entropy, approx KL, clip fraction, mean advantage, mean return, explained
variance, gradient norm (pre-clip), learning rate used, epochs run, mean
episode reward (game result) and shaped return, episode length, wins/draws/losses/truncated in the rollout,
decisions/episodes so far, and `env_seconds`, `inference_seconds`,
`update_seconds`, `wall_seconds`. `eval_log.csv` holds evaluation rows.

Things to watch: entropy collapsing early, approx KL much larger than ~0.02-0.05
(lower the learning rate or set `target_kl`), explained variance staying
negative, clip fraction near 1, one colour's results far below the other's.
Measured on this development machine (CPU-only PyTorch, one iteration of
`ppo_baseline.yaml`): about 4.5 s environment, 6.8 s inference and 11.9 s
update for 2048 decisions, i.e. the CPU-bound game simulation and per-step
inference are comparable to the optimisation; a GPU mainly helps the update
and does not remove the simulation cost.

## 10. Checkpoints and resuming

`latest.pt` (every `checkpoint_interval` iterations and at the end) and
`best.pt` (best evaluation) are written atomically (temp file + rename). A
checkpoint is a Phase 3 model checkpoint (loadable with
`xiangqi_rl.nn.load_model` for evaluation-only use) plus the optimizer,
scheduler, iteration, decision/episode counters, best score, the PPO config
and the RNG states (Python, NumPy, PyTorch CPU/CUDA, trainer, wrapper and
opponent agents). Loading uses `weights_only=True`.

```bash
python scripts/train_ppo.py --resume artifacts/ppo/latest.pt --set total_iterations=300
python scripts/evaluate_ppo.py --checkpoint artifacts/ppo/best.pt --games 200
```

Resuming starts a fresh episode (an in-progress episode is not stored) and
re-derives the learning rate for the (possibly extended) schedule. Model
settings cannot change on resume. **On Kaggle**, set `output_dir` to a path
under `/kaggle/working` (e.g. `--output-dir /kaggle/working/ppo`), download
`latest.pt`/`best.pt` and the CSV logs at the end of the session (or save a
notebook version), upload them as a dataset or into the next session, and
pass the file with `--resume`. No Kaggle path is hard-coded.

## 11. Reproducibility and devices

`seed_everything` seeds Python, NumPy and PyTorch (CPU and CUDA);
baseline agents have their own seeded RNGs and the trainer's minibatch
shuffling uses a seeded NumPy generator. Two CPU runs with the same config
give identical metrics and weights (tested). `deterministic: true` enables
`torch.use_deterministic_algorithms(warn_only=True)`; CUDA runs can still
differ from CPU runs and are slower in this mode. `device: auto|cpu|cuda`;
`cuda` fails with a clear error when unavailable. Game simulation stays on
the CPU.

## 12. Running

```bash
python -m pip install -e ".[dev,nn]"
pytest -q
python scripts/train_ppo.py --config configs/ppo_baseline.yaml --device cpu --smoke-test
python scripts/train_ppo.py --config configs/ppo_baseline.yaml            # longer run
python scripts/train_ppo.py --config configs/ppo_baseline.yaml --set total_iterations=50 \
       --set model.hidden_channels=96 --set model.norm_groups=8
```

`--smoke-test` shrinks model and rollout, runs 2 updates with evaluation,
saves, reloads the checkpoint (outputs must match), samples actions that must
be legal, and resumes for one more update, all in a few seconds on CPU.

## 13. Known limitations

* Opponents are fixed baselines; there is no self-play, league, or search
  (Phase 5). A policy that beats the weak baselines is not necessarily good
  at Xiangqi, and it may overfit to their weaknesses.
* Without shaping the reward is sparse and learning signal is weak (see 5a).
  With shaping, the agent may still prefer material over mating attacks in
  the short run; potential-based shaping keeps the optimum but not the
  learning dynamics. Whether shaping actually improves play has not yet been
  measured.
* The observation does not encode perpetual-check history (Phase 2).
* Single environment, no vectorisation; pure-Python engine speed bounds
  throughput.
* Default hyper-parameters are starting values, not tuned. A run with
  `learning_rate: 3e-4` showed approx KL of 0.05-0.09 per update, so the
  baseline config now uses `1e-4` with `target_kl: 0.02`; this change is
  based on that one observation.
* CUDA code paths are exercised by tests only when a CUDA build of PyTorch
  and a GPU are present; they have not been run on the development machine.
