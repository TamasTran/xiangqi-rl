"""Play Xiangqi in the terminal against a friend or a random-move opponent.

Usage:
    python scripts/play_cli.py                 # two humans
    python scripts/play_cli.py --vs-random     # you are Red, Black plays randomly
    python scripts/play_cli.py --vs-random --side black --seed 1

Enter moves in UCCI notation, e.g. ``h2e2`` (file a-i, rank 0-9 from Red's
back rank). Type ``help`` for the other commands.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

try:
    import xiangqi_rl  # noqa: F401
except ImportError:  # allow running from a checkout without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xiangqi_rl.engine import (  # noqa: E402
    Game,
    GameConfig,
    InvalidMoveFormatError,
    InvalidSquareError,
    IllegalMoveError,
    Side,
    XiangqiError,
    parse_square,
)

HELP = """Commands:
  <move>        play a move, e.g. h2e2
  moves         list all legal moves
  moves <sq>    list legal moves from a square, e.g. moves h2
  undo          take back the last move (two moves when playing the computer)
  fen           print the current FEN
  board         redraw the board
  history       list moves played so far
  reset         start a new game
  help          show this help
  quit          leave"""


def show(game: Game) -> None:
    print()
    print(game.board)
    if game.history:
        last = game.history[-1]
        capture = f" (captured {last.captured.symbol})" if last.captured else ""
        print(f"Last move: {last.move.uci()}{capture}")
    if game.is_terminal():
        winner = game.winner.name if game.winner else "nobody"
        print(f"*** Game over: {game.get_result().value} "
              f"({game.termination_reason.value}); winner: {winner} ***")
    else:
        check = "  CHECK!" if game.is_in_check() else ""
        print(f"{game.side_to_move.name} to move (ply {game.ply}){check}")


def handle_command(game: Game, line: str, computer: Side | None) -> bool:
    """Run a non-move command. Returns False if ``line`` was not a command."""
    words = line.split()
    cmd = words[0]
    if cmd == "help":
        print(HELP)
    elif cmd == "moves":
        moves = game.legal_moves()
        if len(words) > 1:
            try:
                origin = parse_square(words[1])
            except InvalidSquareError as exc:
                print(exc)
                return True
            moves = tuple(m for m in moves if m.from_sq == origin)
        print(" ".join(m.uci() for m in moves) or "(no legal moves)")
    elif cmd == "undo":
        count = 2 if computer is not None and game.ply >= 2 else 1
        try:
            for _ in range(count):
                game.undo_move()
        except XiangqiError as exc:
            print(exc)
        show(game)
    elif cmd == "fen":
        print(game.to_fen())
    elif cmd == "board":
        show(game)
    elif cmd == "history":
        for r in game.history:
            cap = f" x{r.captured.symbol}" if r.captured else ""
            chk = " +" if r.gave_check else ""
            print(f"{r.ply + 1:3d}. {r.side.name:5s} {r.move.uci()}{cap}{chk}")
    elif cmd == "reset":
        game.reset()
        show(game)
    else:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vs-random", action="store_true", help="play against random moves")
    parser.add_argument("--side", choices=["red", "black"], default="red",
                        help="your side when playing the computer (default: red)")
    parser.add_argument("--seed", type=int, default=None, help="seed for the random opponent")
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--fen", help="start from this FEN instead of the standard position")
    args = parser.parse_args()

    config = GameConfig(max_plies=args.max_plies)
    game = Game.from_fen(args.fen, config) if args.fen else Game.new_game(config)
    rng = random.Random(args.seed)
    human = Side.RED if args.side == "red" else Side.BLACK
    computer = human.opponent if args.vs_random else None

    print("XiangqiRL terminal game. Type 'help' for commands. Red = upper case, Black = lower case.")
    show(game)
    while True:
        if computer is not None and not game.is_terminal() and game.side_to_move is computer:
            move = rng.choice(game.legal_moves())
            game.make_move(move)
            print(f"\nComputer plays {move.uci()}")
            show(game)
            continue

        prompt = "game over> " if game.is_terminal() else f"{game.side_to_move.name.lower()}> "
        try:
            line = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("quit", "exit", "q"):
            return
        if handle_command(game, line, computer):
            continue
        try:
            game.make_move(line)
        except (IllegalMoveError, InvalidMoveFormatError, InvalidSquareError, XiangqiError) as exc:
            print(f"! {exc}")
            continue
        show(game)


if __name__ == "__main__":
    main()
