"""Terminal UX: play Connect 4 against the solver.

    python -m harness play-solver --depth 8 --first human

The solver's per-move evaluation is printed each turn, so you can SEE what
depth buys: at depth 2 the scores are shallow heuristics; at depth 10 forced
wins/losses start appearing as exact values many plies ahead.

In-game commands: a column number 0-6 to move, ``d N`` to change the solver
depth mid-game, ``q`` to quit.
"""

from __future__ import annotations

from typing import Optional

from .tasks.connect4 import Connect4Env
from .tasks.connect4_solver import Connect4Solver, HEURISTIC_CAP, MoveEval, WIN_BASE


def _describe(e: MoveEval) -> str:
    if e.exact and e.score > HEURISTIC_CAP:
        return f"win in {WIN_BASE - e.score} plies"
    if e.exact and e.score < -HEURISTIC_CAP:
        return f"loss in {e.score + WIN_BASE} plies"
    if e.exact and e.score == 0:
        return "draw (proven)"
    return f"eval {e.score:+d}"


def _print_board(env: Connect4Env) -> None:
    print()
    print(env.render())
    print()


def _solver_move(env: Connect4Env, depth: int) -> int:
    solver = Connect4Solver(max_depth=depth)
    evals = solver.evaluate_moves(env.board, env.current_player)
    evals.sort(key=lambda e: -e.score)
    print(f"solver (depth {depth}) sees, best first:")
    for e in evals:
        print(f"  col {e.col}: {_describe(e)}")
    choice = evals[0].col
    print(f"solver plays column {choice}  ({solver.nodes} nodes searched)")
    return choice


def play_solver(depth: int = 8, human_first: bool = True) -> None:
    env = Connect4Env()
    env.reset()
    human_player = 1 if human_first else -1
    print(f"You are {'X (first)' if human_first else 'O (second)'}. "
          f"Solver depth {depth}. Columns 0-6, 'd N' = set depth, 'q' = quit.")
    _print_board(env)

    while not env.done:
        if env.current_player == human_player:
            move = _read_human_move(env, depth)
            if move is None:
                print("goodbye")
                return
            if isinstance(move, tuple):  # depth change
                depth = move[1]
                print(f"solver depth set to {depth}")
                continue
            try:
                env.step(move)
            except ValueError as err:
                print(f"  {err}")
                continue
        else:
            env.step(_solver_move(env, depth))
        _print_board(env)

    if env.winner == 0:
        print("Draw.")
    elif env.winner == human_player:
        print("You win!")
    else:
        print(f"Solver wins (depth {depth}).")


def _read_human_move(env: Connect4Env, depth: int):
    """Returns int column, ('d', N) for depth change, or None to quit."""
    while True:
        try:
            raw = input(f"your move (legal: {env.legal_actions()}): ").strip().lower()
        except EOFError:
            return None
        if raw in ("q", "quit", "exit"):
            return None
        if raw.startswith("d ") or raw.startswith("depth "):
            try:
                return ("d", max(1, int(raw.split()[1])))
            except (IndexError, ValueError):
                print("  usage: d N   (e.g. 'd 10')")
                continue
        try:
            return int(raw)
        except ValueError:
            print("  enter a column 0-6, 'd N', or 'q'")
