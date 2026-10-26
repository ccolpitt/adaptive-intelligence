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


def has_two_heads(module: torch.nn.Module) -> bool:
    """Does this saved policy propose moves and judge positions separately?

    Lineages 1 to 4 produced a single set of seven numbers doing both jobs;
    lineage 5 onward returns move scores and a position value. Everything that
    reads a policy has to cope with both, because the archive keeps every
    generation forever and they must all stay playable.
    """
    with torch.no_grad():
        return isinstance(module(torch.zeros(1, 2, 6, 7)), tuple)


def _policy_move(module: torch.nn.Module, obs: np.ndarray, env: Connect4Env,
                 lookahead: int = 0) -> int:
    """The policy's move, with its reasoning printed.

    With ``lookahead`` set, the policy may examine that many distinct
    positions before committing — the same budgeted search used in training.
    The printed numbers are then what the search concluded rather than the
    network's first impression, and the two can differ sharply: that gap is
    the most interesting thing to watch while playing.
    """
    legal = env.legal_actions()
    two_heads = has_two_heads(module)

    def move_scores(o):
        with torch.no_grad():
            out = module(torch.from_numpy(o).float().unsqueeze(0))
            if isinstance(out, tuple):
                out = out[0]
            return out.squeeze(0).numpy()

    def policy_value(o):
        with torch.no_grad():
            logits, value = module(torch.from_numpy(o).float().unsqueeze(0))
            return logits.squeeze(0).numpy(), float(value.item())

    if two_heads:
        _, position_value = policy_value(obs)
        print(f"policy's view of this position: {position_value:+.3f} "
              f"(+1 = it is winning, -1 = it is losing)")

    if lookahead > 0:
        if two_heads:
            # Both heads guide the search: the move head says where to look,
            # the value head says what was found. Using the move scores as if
            # they were position values -- which is what the single-output
            # path does -- would make the search markedly worse.
            from .budgeted_search import (
                BudgetedSearch, PerfectConnect4Model, PolicyValueEvaluator, SearchConfig,
            )
            search = BudgetedSearch(
                PolicyValueEvaluator(policy_value, env.cols),
                PerfectConnect4Model(env.rows, env.cols),
                SearchConfig(node_budget=lookahead, dirichlet_noise=0.0,
                             select_by="visits"))
        else:
            from .budgeted_search import make_search

            search = make_search(move_scores, budget=lookahead, spend_rule="puct",
                                 rows=env.rows, cols=env.cols,
                                 dirichlet_noise=0.0, select_by="visits")
        result = search.run(obs)
        values, visits = result.action_values, result.visit_counts
        # The move is the most-attended one, as AlphaZero selects.
        ranked = sorted(legal, key=lambda c: -visits[c])
        print(f"policy thinking with a budget of {lookahead} positions "
              f"(examined {result.nodes_evaluated} distinct, looked "
              f"{result.max_depth_reached} moves ahead):")
        for c in ranked:
            print(f"  col {c}: {visits[c]:>5} visits   value {values[c]:+.3f}")
        choice = int(ranked[0])
    else:
        q = move_scores(obs)
        ranked = sorted(legal, key=lambda c: -q[c])
        label = ("move head's preference, one look, no thinking ahead — "
                 "this is how the policy is scored" if two_heads
                 else "policy values, no thinking ahead")
        print(f"{label} (best first):")
        for c in ranked:
            print(f"  col {c}: {q[c]:+.3f}")
        choice = int(ranked[0])
    print(f"policy plays column {choice}")
    return choice


def find_champion(store, lineage: Optional[str] = None) -> Optional[str]:
    """The champion to play by default.

    Champions are recorded per lineage, and the task id alone is not a
    lineage id — asking for "connect4" finds nothing once lineages are named
    connect4-L5. So: the named lineage if given, otherwise the champion of
    whichever lineage was promoted most recently, which is the one a person
    almost always means.
    """
    from .jsonl_store import read_jsonl

    if lineage is not None:
        return store.archive.current_champion(lineage)
    champion = None
    for rec in read_jsonl(store.archive.events_path):
        if rec.get("kind") == "promotion":
            champion = rec["policy_id"]
    return champion


def list_policies(store, lineage: Optional[str] = None,
                  champions_only: bool = True, limit: int = 40) -> None:
    """Print the archived policies a person could play against."""
    from .jsonl_store import read_jsonl

    promotions = [r for r in read_jsonl(store.archive.events_path)
                  if r.get("kind") == "promotion"]
    promoted_ids = {r["policy_id"] for r in promotions}
    current = {}
    for r in promotions:
        current[r.get("task_id")] = r["policy_id"]

    entries = store.archive.all_entries()
    if lineage:
        entries = [e for e in entries if e.get("lineage_id") == lineage]
    if champions_only:
        entries = [e for e in entries if e.get("policy_id") in promoted_ids]

    if not entries:
        print("no matching policies in the archive")
        return

    heading = ("champions" if champions_only else "all policies")
    print(f"{heading}{' in ' + lineage if lineage else ''} "
          f"({len(entries)} total, showing the last {min(limit, len(entries))}):")
    print(f"  {'policy':26s} {'lineage':16s} {'gen':>4} {'benchmark':>10}  note")
    for entry in entries[-limit:]:
        pid = entry.get("policy_id", "?")
        bench = next((r["score"] for r in reversed(entry.get("eval_records", []))
                      if "benchmark" in str(r.get("benchmark_ref", ""))
                      and "lookahead" not in str(r.get("benchmark_ref", ""))), None)
        is_current = pid in current.values()
        note = "<- current champion" if is_current else ""
        print(f"  {pid:26s} {str(entry.get('lineage_id', '-')):16s} "
              f"{entry.get('generation', '?'):>4} "
              f"{(f'{bench:.3f}' if bench is not None else '-'):>10}  {note}")
    print(f"\nplay one with:  python -m harness play --policy <policy>")
    print(f"and give it time to think:  "
          f"python -m harness play --policy <policy> --lookahead 200")


def play_policy(store, policy_id: Optional[str] = None, human_first: bool = True,
                lookahead: int = 0, lineage: Optional[str] = None) -> None:
    """Play against an archived policy (default: the latest champion).

    ``lookahead`` is how many distinct positions the policy may examine per
    move. Zero is how it is scored; a few hundred makes it markedly stronger,
    and playing both is the quickest way to feel what search adds.
    """
    if policy_id is None:
        policy_id = find_champion(store, lineage)
        if policy_id is None:
            print("no champion in the archive yet — run an experiment first, or "
                  "list what is there with: python -m harness policies --all")
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
          f"Opponent lookahead: {lookahead if lookahead else 'none (as scored)'}. "
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
            result = env.step(_policy_move(module, obs, env, lookahead))
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
