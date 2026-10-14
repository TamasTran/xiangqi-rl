# Xiangqi rules as implemented by the XiangqiRL engine

This document describes the exact rules and policies of the Phase 1 engine
(`src/xiangqi_rl/engine`). Where the engine deliberately simplifies the
official rules, this is stated explicitly.

## 1. Board and coordinates

The board has **10 rows × 9 columns = 90 squares** (pieces stand on
intersections; the engine treats them as squares).

| Concept        | Convention |
|----------------|------------|
| Row 0          | Top of the board, Black's back rank |
| Row 9          | Bottom of the board, Red's back rank |
| Column 0       | Leftmost file as seen from Red |
| Square index   | `square = row * 9 + col`, an `int` in `0..89` |
| Square name    | file letter `a`–`i` (column 0–8) + rank digit `0`–`9`, with `rank = 9 - row` |
| Forward        | Red moves toward **decreasing** rows; Black toward **increasing** rows |
| Black's half   | rows 0–4 |
| Red's half     | rows 5–9 (the river lies between rows 4 and 5) |
| Black palace   | rows 0–2, columns 3–5 (`d7`–`f9`) |
| Red palace     | rows 7–9, columns 3–5 (`d0`–`f2`) |

Square names follow the common UCCI convention, so moves read like `h2e2`
(Red's "central cannon" opening). Red's general starts on `e0` (square 85),
Black's on `e9` (square 4).

```
9  r n b a k a b n r      row 0   (Black)
8  . . . . . . . . .      row 1
7  . c . . . . . c .      row 2
6  p . p . p . p . p      row 3
5  . . . . . . . . .      row 4
   ~~~~~ river ~~~~~
4  . . . . . . . . .      row 5
3  P . P . P . P . P      row 6
2  . C . . . . . C .      row 7
1  . . . . . . . . .      row 8
0  R N B A K A B N R      row 9   (Red)
   a b c d e f g h i
```

## 2. Initial setup

Each side has 16 pieces: 1 general (K), 2 advisors (A), 2 elephants (B),
2 horses (N), 2 chariots (R), 2 cannons (C), 5 soldiers (P). Red is upper
case, Black lower case. The FEN of the starting position is

```
rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1
```

**Red moves first.** (`w` in the FEN means Red, following common Xiangqi FEN
practice; `r` is also accepted when parsing.)

## 3. Piece movement (pseudo-legal moves)

No piece may capture a piece of its own side.

| Piece | Movement |
|-------|----------|
| **General** | One step orthogonally, staying inside its own palace. |
| **Advisor** | One step diagonally, staying inside its own palace. |
| **Elephant** | Exactly two steps diagonally. The intermediate square (the *elephant's eye*) must be empty. It may never cross the river. |
| **Horse** | One step orthogonally then one step diagonally outward (the 2+1 "L"). The orthogonally adjacent square in the direction of the two-step component (the *horse leg*) must be empty. All 8 patterns are implemented; a piece diagonally adjacent to the horse never blocks it. |
| **Chariot** | Any number of squares orthogonally; cannot jump; captures the first enemy piece met. |
| **Cannon** | Moves like a chariot to an **empty** square (no pieces in between). To **capture** it must jump exactly one piece of either colour (the *screen*), and the target is the first piece beyond the screen, which must be an enemy. No capture without a screen or over two screens. |
| **Soldier** | One step forward. After crossing the river it may also move one step sideways. Never backward, never promotes; a soldier on the last rank can only move sideways. |

## 4. Legal versus pseudo-legal moves

* A **pseudo-legal** move satisfies the piece's movement pattern above.
* A **legal** move is a pseudo-legal move after which the moving side is
  **not in check** (section 5).

Legality is always decided on the **board after the move**. This one rule
covers pinned pieces, discovered attacks, a general stepping back along the
line it is attacked on, a moving piece becoming (or ceasing to be) a cannon
screen, and exposing the two generals to each other.

## 5. Attacks, the flying general, and check

**Attack.** A piece attacks a square if its movement pattern would let it
capture an enemy piece standing there on the current board. Pins are
ignored: a piece pinned against its own general still attacks, since a
general may never be left capturable. The engine computes attacks by
looking outward from the target square (reverse lookup), with the horse leg
measured next to the *horse* and the cannon needing exactly one screen. It
never uses "is the square among the opponent's legal moves", which would
wrongly ignore pinned attackers.

**Flying general.** The two generals may never stand on the same file with
no piece between them. The engine models this as its own predicate
(`generals_facing`) rather than as a long-range "attack" by the general.
Because no legal move can create this situation, a general never actually
captures the other by flying across the board, and the general's
pseudo-legal moves are only its one-step palace moves.

**Check.** A side is in check if its general is attacked **or** the generals
face each other.

## 6. Terminal states

After every move the engine evaluates, **in this order**:

| # | Condition | Result | `TerminationReason` |
|---|-----------|--------|---------------------|
| 1 | Side to move has no legal move and is in check | opponent wins | `CHECKMATE` |
| 1 | Side to move has no legal move and is **not** in check | opponent wins | `STALEMATE` |
| 2 | Repetition policy fires (section 7) | per policy | `PERPETUAL_CHECK` / `REPETITION` |
| 3 | `max_plies` plies have been played | draw | `MAX_PLIES` |

* **Stalemate loses.** Under the Asian/Chinese rules a player with no
  legal move loses whether or not they are in check. The engine still
  distinguishes the two cases through `TerminationReason`, `is_checkmate()`
  and `is_stalemate()`.
* A mate on the last allowed ply is still a win, because rule 1 is checked
  first.
* A position given to `Game(...)` is evaluated immediately, so a starting
  position with no legal moves is terminal at once.
* Terminal games reject further moves with `GameOverError`, and
  `legal_moves()` returns an empty tuple.

## 7. Repetition policy (simplified)

Official Xiangqi repetition rules (perpetual check, perpetual chase, and
the classification of "idle" and "forcing" moves) are intricate. **The engine
does not implement the full official rules.** It uses a deterministic
simplification suitable for an initial RL environment, isolated in
`engine/repetition.py` so it can be replaced:

1. A **position** is the piece placement plus the side to move.
2. When the current position has occurred `repetition_threshold` times
   (default **3**, counting the game's starting position), the game is
   adjudicated.
3. The adjudication window covers every move from the earliest of those
   occurrences to now.
4. If `perpetual_check_loses` is true (the default) and **exactly one** side
   gave check with *every* one of its moves in the window, that side
   **loses** (`PERPETUAL_CHECK`).
5. Otherwise the game is a **draw** (`REPETITION`). This includes mutual
   perpetual check.

Known gaps: perpetual **chase** is not detected (it is scored as a draw),
and there is no special handling of moves that alternate check and chase.
Setting `repetition_threshold=None` disables repetition adjudication.

## 8. Maximum-ply policy

`GameConfig.max_plies` (default **300**) bounds game length, so self-play
cannot loop forever. Reaching the limit is a **draw** (`MAX_PLIES`), unless
the final move delivered mate or stalemate. `max_plies=None` disables the
limit. A ply is one move by one side.

There is currently **no** "natural move limit" (no-capture rule). The
configurable ply limit is the only length-based draw.

## 9. Position validation

`Game(board, side_to_move)` rejects positions that cannot occur in a legal
game:

* not exactly one general per side, or a general outside its palace;
* more pieces of a type than a side starts with;
* advisors or elephants on squares they can never reach;
* soldiers behind their starting rank, or off their starting file before
  crossing the river;
* the side **not** to move being in check (including facing generals).

`Board` itself accepts any placement, so individual movement rules can be
tested in isolation.

## 10. Serialization and determinism

* `Game.position_key()` returns `"<FEN placement> <w|b>"`. It is a
  **collision-free serialization**, not a hash: two positions share a key if
  and only if they have the same placement and side to move. Repetition
  detection uses it.
* `Game.to_fen()` and `Game.from_fen()` round-trip the placement and side to
  move. When parsing, the move-counter fields are ignored.
* Legal moves are always returned sorted by `(from_sq, to_sq)`.
* Nothing in the engine uses randomness, time, or global mutable state.

## 11. Move and action encoding

`Move(from_sq, to_sq)` has value equality and ordering. For future RL use,
`Move.to_action()` returns `90 * from_sq + to_sq` (`0 <= action < 8100`), and
`Move.from_action()` inverts it. Actions with `from_sq == to_sq` are not moves
and are rejected.

## 12. Verification

* Hand-enumerated list of all 44 legal opening moves.
* Perft (leaf counts of the legal-move tree) from the start position matches
  the published reference values 44 / 1,920 / 79,666 (default test run) and
  3,290,240 at depth 4 (`pytest -m slow`).
* On reproducible random games, legal moves are compared with an
  independently written brute-force reference (`tests/engine/engine_helpers.py`),
  and reverse-lookup attacks with the move generators.
* Unit tests for every piece, palace, river, eye, leg and screen rule, check,
  evasions, pins, mate, stalemate, repetition and the ply limit.

## 13. Known limitations and planned extensions

* Simplified repetition: no perpetual-chase detection, no full official
  adjudication tables.
* No natural move limit (e.g. 60 moves without capture); only `max_plies`.
* Pure-Python immutable boards prioritise clarity over speed (about 80k
  legal-move leaf nodes per second for perft). A faster representation
  (bitboards or incremental make/unmake) may come later, validated against
  the current engine and the perft numbers.
* Planned in later phases (**not** in Phase 1): a Gymnasium-style
  environment, observation encoding, PPO and self-play, a policy-value
  network, MCTS, and a Pygame UI.
