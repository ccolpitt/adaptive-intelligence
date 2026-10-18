"""Terminal UX: play Connect 4 against the solver or an archived policy.

    python -m harness play-solver --depth 8 --first human
    python -m harness play                      # vs the current champion
    python -m harness play --policy connect4-pol-00042

The opponent's per-move evaluation is printed each turn. For the solver you
SEE what depth buys: at depth 2 the scores are shallow heuristics; at depth
10 forced wins/losses appear as exact values many plies ahead. For a policy
you see its Q value per column — its learned belief about each move, which
should live in [-1, +1].

In-game commands: a column number 0-6 to move, ``d N`` to change the solver
depth mid-game (solver mode only), ``q`` to quit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

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


def _policy_move(module: torch.nn.Module, obs: np.ndarray, env: Connect4Env) -> int:
    """Greedy move from the policy's Q values, with its beliefs printed."""
    legal = env.legal_actions()
    with torch.no_grad():
        q = module(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).numpy()
    ranked = sorted(legal, key=lambda c: -q[c])
    print("policy Q values (its belief per column, best first):")
    for c in ranked:
        print(f"  col {c}: {q[c]:+.3f}")
    choice = int(ranked[0])
    print(f"policy plays column {choice}")
    return choice


def play_policy(store, policy_id: Optional[str] = None, human_first: bool = True) -> None:
    """Play against an archived policy (default: the current champion)."""
    if policy_id is None:
        policy_id = store.archive.current_champion("connect4")
        if policy_id is None:
            print("no champion in the archive yet — run the loop first")
            return
    module, meta = store.archive.load_policy(policy_id)
    entry = store.archive.get_entry(policy_id)
    bench = next(
        (f"{r['score']:.3f} on {r['benchmark_ref']}"
         for r in reversed(meta.get("eval_records", []))
         if "benchmark" in str(r.get("benchmark_ref", ""))),
        "n/a",
    )
    print(f"opponent: {policy_id} (gen {entry.get('generation', '?')}, benchmark {bench})")

    env = Connect4Env()
    obs = env.reset()
    human_player = 1 if human_first else -1
    print(f"You are {'X (first)' if human_first else 'O (second)'}. "
          f"Columns 0-6, 'q' = quit.")
    _print_board(env)

    while not env.done:
        if env.current_player == human_player:
            move = _read_human_move(env, depth=0)
            if move is None:
                print("goodbye")
                return
            if isinstance(move, tuple):
                print("  (depth commands only apply in play-solver mode)")
                continue
            try:
                result = env.step(move)
            except ValueError as err:
                print(f"  {err}")
                continue
        else:
            result = env.step(_policy_move(module, obs, env))
        obs = result.next_obs
        _print_board(env)

    if env.winner == 0:
        print("Draw.")
    elif env.winner == human_player:
        print("You win!")
    else:
        print(f"{policy_id} wins.")


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
