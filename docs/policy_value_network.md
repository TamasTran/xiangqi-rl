# Policy-value network (Phase 3)

Code: `src/xiangqi_rl/nn/` (`config.py`, `model.py`, `policy.py`,
`checkpoint.py`). Requires PyTorch (`pip install -e ".[dev,nn]"`).

> **Status.** This phase produces an *untrained* network. Nothing here shows
> that it plays Xiangqi well; the tests only check shapes, numerical
> behaviour, masking, gradients and checkpoints. A network that produces
> outputs, a network that has been trained, and an agent that plays well are
> three different things, and only the first exists.

## 1. What the network predicts

| Output | Shape | Meaning |
|---|---|---|
| `policy_logits` | `[B, 8100]` | raw, **unmasked** score for every `90*from_sq + to_sq` action |
| `value` | `[B, 1]` | expected outcome in `[-1, 1]` for the **side to move** |

```python
logits, value = model(observations)       # observations: float32 [B, 17, 10, 9]
```

The forward pass never sees legal moves. Illegal actions get logits too;
masking is applied afterwards by `xiangqi_rl.nn.policy`.

## 2. Architecture

```
obs [B,17,10,9]
 -> Conv3x3(17->H) -> GroupNorm -> ReLU                    (stem, H = hidden_channels)
 -> N x ResidualBlock: Conv3x3-Norm-ReLU-Conv3x3-Norm, +skip, ReLU
 -> features [B,H,10,9]                                    (shared backbone)
     |- policy head: Conv1x1(H->8)-Norm-ReLU-Flatten-Linear(720 -> 8100)   -> logits
     `- value head : Conv1x1(H->8)-Norm-ReLU-Flatten-Linear(720 -> 64)-ReLU
                                                    -Linear(64 -> 1)-Tanh -> value
```

Why a **shared backbone**: both heads need the same "understanding" of the
position; sharing it halves the compute per position and lets the value
signal (every game) shape features that help the policy and vice versa.

Design choices (all configurable in `ModelConfig`):

* **GroupNorm by default** (8 groups), not BatchNorm. In PPO the same
  position is evaluated alone at rollout time and inside a large minibatch
  at update time; BatchNorm would make those outputs differ and add a
  train/eval discrepancy. GroupNorm is batch-independent (tested). `batch`
  and `none` are available for experiments.
* **Padding 1, 3x3 convs**: the 10x9 board size is preserved everywhere.
* **Small initialisation of the final policy/value layers** (std 0.01), so
  an untrained model is near-uniform with value near 0 rather than
  producing large arbitrary outputs.
* **Tanh value** matches the +1/-1/0 reward range; `value_activation="none"`
  returns the raw scalar if Phase 4 prefers a different value loss.
* Default size: `hidden_channels=64`, `num_res_blocks=4`, about **6.2 M**
  parameters, of which ~5.8 M are the policy head's linear layer. It is a
  small starting point for experiments, not a tuned or optimal setting; it
  runs on CPU and fits easily on a Kaggle T4.

Defaults for `in_channels`, board size and `action_dim` are read from the
Phase 2 modules (`NUM_CHANNELS`, `ACTION_DIM`), so they cannot silently
disagree with the environment. Invalid configurations (non-positive sizes,
`hidden_channels` not divisible by `norm_groups`, unknown norm) fail at
construction. Wrong input shapes/dtypes fail with a clear `ValueError`.

## 3. Observation and action interfaces (from Phase 2)

* Observation `[17, 10, 9]` float32: channels 0-6 RED pieces, 7-13 BLACK
  pieces (general, advisor, elephant, horse, chariot, cannon, soldier),
  14 side to move (1 = RED), 15 repetition progress, 16 ply progress. Board
  axes are absolute (row 0 = Black's back rank); the board is *not* flipped
  for the side to move. See `docs/rl_environment.md`.
* Action `a = 90 * from_sq + to_sq`, `square = row*9 + col`. Logit index `a`
  is the score of moving from square `a // 90` to square `a % 90`. The
  model's `action_dim` must equal the environment's action space (tested).

## 4. Legal-action masking (`xiangqi_rl.nn.policy`)

```python
from xiangqi_rl.nn import MaskedCategorical, terminal_rows

mask = torch.from_numpy(env.action_masks()).unsqueeze(0)   # bool [1, 8100]
dist = MaskedCategorical(logits, mask)
action = dist.sample()          # [B] legal actions
logp   = dist.log_prob(action)  # [B]
ent    = dist.entropy()         # [B]
probs  = dist.probs             # [B, 8100], exactly 0 where illegal
```

* **Raw logits vs probabilities.** Logits are unnormalised scores over all
  8100 actions. Probabilities are the softmax over the *legal* actions only;
  they depend on the mask, so `log_prob` is only meaningful together with it.
* Illegal logits are replaced by the most negative finite float of the
  dtype (not `-inf`), keeping `log_softmax`, entropy and gradients NaN-free.
  Illegal logits receive zero gradient.
* A row with **no** legal action (a finished game, `terminal_rows(mask)`)
  raises `EmptyLegalMaskError`. Terminal positions must be filtered out
  before building a policy; their value target comes from the game result.
* Mask must be `bool` and the same shape as the logits, otherwise a
  `TypeError`/`ValueError` is raised.
* `log_prob` raises if an action is illegal under the mask. This is an
  intentional guard against old/new mask mismatches.

### Requirement for the Phase 4 PPO trainer

The **same legal-action mask must be used for the old and new
log-probabilities of a given state-action pair.** Store the mask (or
recompute it deterministically from the stored state) with each transition;
using a different mask at update time silently changes the policy
distribution and corrupts the PPO ratio.

## 5. Value perspective

`value` is from the **side to move** in the input position: +1 means the
player about to move is expected to win, -1 lose, 0 draw. This is consistent
with Phase 2 (`reward` is from the mover's perspective):

* A reward `r` received by the mover at step `t` corresponds to a value of
  `-r` for the side to move in the next observation (the opponent). E.g.
  after a checkmate the side to move has lost: the terminal position's value
  is -1.
* Bootstrapping and returns across plies therefore flip sign each ply:
  `G_t = r_t - gamma * V(s_{t+1})` in terms of the mover's reward and the
  next side-to-move value.
* `value_to_red_perspective(value, observations)` converts to a fixed RED
  view (reads channel 14) for logging/evaluation only. Never mix
  side-to-move values with fixed-RED values in the same target.

## 6. Checkpoints

```python
from xiangqi_rl.nn import create_model, save_checkpoint, load_model

model = create_model(seed=0)                    # reproducible init
save_checkpoint("runs/model.pt", model)         # optionally optimizer=..., extra={...}
model2 = load_model("runs/model.pt", map_location="cpu")
```

A checkpoint stores the weights (always on CPU), the model config, a format
version and an architecture identifier (`xiangqi-policy-value-resnet-v1`),
plus optional optimizer state. Writes are atomic and the destination is
always explicit. Loading uses `torch.load(weights_only=True)` and an explicit
`map_location`, so CUDA checkpoints load on CPU and arbitrary pickled code is
refused. Still, load only files you trust.

## 7. Phase 4 integration notes

* Rollout: `obs -> model -> MaskedCategorical(logits, mask) -> sample`;
  store `obs, mask, action, logp, value, current_player`.
* Update: recompute `logits, value` for stored observations, build
  `MaskedCategorical` with the **stored masks**, compare `log_prob` and use
  `entropy()` for the bonus. Gradients flow from both heads into the shared
  backbone (tested); any `torch.optim` optimizer works.
* Exclude terminal positions from policy computations (`terminal_rows`).

## 8. Running

```bash
python -m pip install -e ".[dev,nn]"
pytest -q                                  # nn tests skip if torch is missing
python scripts/smoke_test_model.py
```

CUDA tests (forward on GPU, CUDA-checkpoint-on-CPU) run only when a CUDA
build of PyTorch with a device is available and are skipped otherwise.

## 9. Known limitations

* Untrained: outputs carry no Xiangqi knowledge.
* The policy head is one dense layer `720 -> 8100`; most of the parameters
  live there. A factorised from/to head would be much smaller and is a
  possible later change (it would bump the architecture identifier).
* No board mirroring/augmentation: the network sees absolute coordinates and
  must learn both colours' orientations from the side-to-move plane.
* Perpetual-check history is not in the observation (Phase 2 limitation), so
  the value of such positions is ambiguous.
* Performance on GPU has not been measured in this environment (the dev
  machine used a CPU-only PyTorch build).
