"""exp-023: is a budget of states explored better than a budget of depth?

No training. One network — the strongest policy in the archive — judged on the
fixed 168-game ladder under four ways of choosing a move:

  1. bare: one look at the board, play the best-rated column.
  2. depth-limited: examine every line to exactly k moves ahead.
  3. budgeted: examine at most N states per move, choosing which states to
     examine by what the network thinks is promising.

The comparison that matters is cost-matched: a depth-limited search of a
7-wide tree needs about 7^k states to see k moves ahead, and it spends them
uniformly whether or not a line matters. A budgeted search can abandon a line
after one look and follow another twenty moves deep.

Run:  python3 experiments/exp_023_budget_vs_depth.py
"""

import json
import time

import torch

from _common import DECISION_CRITERIA, Store

from harness.budgeted_search import budgeted_q
from harness.lookahead import LookaheadQ
from harness.tasks.connect4_ladder import Connect4LadderTask

EXPERIMENT_ID = "exp-023-budget-vs-depth"
DEPTHS = (2, 3)
BUDGETS = (50, 200, 400)


def strongest_archived(store, benchmark_ref="connect4-benchmark@v1"):
    best_id, best_score = None, -1.0
    for entry in store.archive.all_entries():
        try:
            _, meta = store.archive.load_policy(entry["policy_id"])
        except Exception:
            continue
        for rec in meta.get("eval_records", []):
            if benchmark_ref in str(rec.get("benchmark_ref")) and rec["score"] > best_score:
                best_score, best_id = rec["score"], entry["policy_id"]
    return best_id, best_score


def main() -> None:
    store = Store("store")
    task = Connect4LadderTask()
    policy_id, bare_recorded = strongest_archived(store)
    module, _ = store.archive.load_policy(policy_id)

    def q(obs):
        with torch.no_grad():
            return module(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).numpy()

    hypothesis = f"""\
CONTEXT: The task is to win Connect 4. A policy is a small convolutional
network that reads the board as two 6x7 planes (the mover's pieces and the
opponent's, flipped to the mover's view each turn) and outputs one estimated
value per column. It is judged on a fixed ladder: 168 games against
perfect-play-to-depth-k opponents for k = 1..6, scored as the average win
rate across the six. The strongest policy ever archived ({policy_id}) scores
{bare_recorded:.3f} when it plays by taking the best-rated column with no
lookahead at all. Every opponent on the ladder searches; this policy does
not. Earlier measurement (exp-020) showed that letting a network look a
fixed number of moves ahead raised its score several-fold, which says the
network's position judgment is better than its instant move choice.

HYPOTHESIS: if the lookahead budget is specified as a number of states the
policy may examine per move, rather than as a fixed number of moves to look
ahead, then the same network will score higher at a far lower cost, because
a fixed depth spends the budget uniformly across lines that do not matter,
whereas a state budget can be spent where the network's value estimates say
the game will be decided. Changes, each from -> to, with no training and no
change to any weight: move selection from "best-rated column" -> "value of
each column after examining at most N states", for N in {BUDGETS}; and, as
the comparison baseline, "best-rated column" -> "exhaustive search to depth
k" for k in {DEPTHS}.

PREDICTION: at a matched number of states examined, the state-budgeted search
scores at least as high as the depth-limited search, and reaches a greater
maximum depth, because it prunes. Specifically: 50 states budgeted should
match or beat depth 2 (which costs roughly 7^2 = 400+ states per move).

{DECISION_CRITERIA}"""

    store.experiments.register(
        experiment_id=EXPERIMENT_ID,
        hypothesis=hypothesis,
        changes=f"inference only, no training: move selection from bare argmax -> "
        f"depth-limited search (k in {DEPTHS}) and -> state-budgeted best-first "
        f"search (N in {BUDGETS}); weights unchanged",
        conditions={
            "task_ref": task.ref,
            "benchmark_ref": task.benchmark.ref,
            "config": {"policy": policy_id, "depths": list(DEPTHS),
                       "budgets": list(BUDGETS), "gamma": 0.99, "c_puct": 1.4},
            "seed": 0,
        },
        prior_art=["exp-020-lookahead-inference", "exp-022-B-search-targets-seed27"],
        input_policies=[policy_id],
        tags=["lookahead", "search-budget", "inference-only", "ladder-v1"],
    )

    rows = []
    t0 = time.time()
    res = task.benchmark.score(q)
    rows.append({"method": "bare", "states_per_move": 1, "ladder": res.score,
                 "max_depth": 0, "seconds": round(time.time() - t0, 1),
                 "per_rung": res.detail["per_rung"]})
    print(f"bare: {res.score:.3f}")

    for depth in DEPTHS:
        la = LookaheadQ(q, depth=depth)
        t0 = time.time()
        res = task.benchmark.score(la)
        rows.append({"method": f"depth-{depth}", "states_per_move": la.leaf_evals // 168,
                     "ladder": res.score, "max_depth": depth,
                     "seconds": round(time.time() - t0, 1),
                     "per_rung": res.detail["per_rung"]})
        print(f"depth {depth}: {res.score:.3f} ({la.leaf_evals // 168} states/move)")

    for budget in BUDGETS:
        search = budgeted_q(q, node_budget=budget)
        t0 = time.time()
        res = task.benchmark.score(search)
        rows.append({"method": f"budget-{budget}",
                     "states_per_move": search.total_nodes_evaluated // max(search.total_moves, 1),
                     "ladder": res.score, "max_depth": search.deepest_reached,
                     "seconds": round(time.time() - t0, 1),
                     "per_rung": res.detail["per_rung"]})
        print(f"budget {budget}: {res.score:.3f} (depth reached {search.deepest_reached})")

    path = store.telemetry_dir / f"{EXPERIMENT_ID}.json"
    path.write_text(json.dumps({"experiment_id": EXPERIMENT_ID, "task_ref": task.ref,
                                "policy": policy_id, "comparisons": rows,
                                "iterations": []}, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
