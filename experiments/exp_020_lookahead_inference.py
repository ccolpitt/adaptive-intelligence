"""exp-020: lookahead at move time — search over the champion's own Q.

Plain language: no training at all. We take the current champion network
exactly as it is and change only HOW it picks a move. Today: one glance at
the board, pick the largest of seven numbers. With lookahead: actually try
each legal move on a copy of the board (and the opponent's replies, k plies
deep), and let the network judge only the resulting positions. Every solver
on the ladder does this; our policy never has. If the ladder score jumps
with zero training, the binding constraint was "can't look one move ahead",
not "doesn't know Connect 4".

Deterministic evaluation (fixed openings, greedy play) — no seeds needed.

Run:  caffeinate -is python3 experiments/exp_020_lookahead_inference.py
"""

import json
import time

import torch

from _common import DECISION_CRITERIA, Store, context_paragraph, measure_incumbent

from harness.lookahead import LookaheadQ
from harness.tasks.connect4_ladder import Connect4LadderTask

DEPTHS = (1, 2, 3)
EXPERIMENT_ID = "exp-020-lookahead-inference"


def main() -> None:
    store = Store("store")
    task = Connect4LadderTask()
    incumbent_id, incumbent = measure_incumbent(store, task)
    module, _ = store.archive.load_policy(incumbent_id)

    def base_q(obs):
        with torch.no_grad():
            return module(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).numpy()

    hypothesis = f"""\
{context_paragraph(incumbent_id, incumbent)}
Evidence chain: capacity is sufficient (exp-010), endgame values are
calibrated in sign and magnitude (solver-proven wins +0.66, losses -0.67 on
the champion card), yet the champion still misses ~9% of immediate wins and
~11% of forced blocks (card: take_the_win 0.91, block_the_loss 0.89) and
cannot beat the depth-1 solver reliably (rung-1 = 0.34). Data levers are
exhausted (exp-002..009), deeper nets don't help (exp-018), and MC credit
propagation hurt (exp-019). The one structural difference between the
policy and every opponent that beats it: the policy never looks ahead.

HYPOTHESIS: reflex-only inference is the binding constraint. Wrapping the
UNCHANGED champion network in a k-ply negamax over real rules (leaves valued
by the network) raises its ladder score substantially with ZERO additional
training. Exact change: move selection argmax(Q(s,.)) -> argmax over
search-backed values at depth k in {DEPTHS}. Network weights untouched.

PREDICTION: depth-2 ladder >= 0.20 (vs bare {incumbent.score:.3f}; the
2-ply search mechanically eliminates both card-flagged blunder classes, and
rung-1 alone should approach 1.0 since a depth-1 solver can be matched by a
depth-2 search with any sane leaf values); depth-3 >= depth-2. If depth-2
stays < 0.12, the leaf values are too poor to search over, and the
bottleneck is value quality after all. Follow-on if supported: search during
TRAINING (self-play with lookahead = AlphaZero-lite), and the policy-card
'horizon' question becomes moot — the search supplies the horizon.

{DECISION_CRITERIA}"""

    store.experiments.register(
        experiment_id=EXPERIMENT_ID,
        hypothesis=hypothesis,
        changes="inference-only: wrap champion Q in k-ply negamax lookahead "
        f"(k in {DEPTHS}); no training, no weight change",
        conditions={
            "task_ref": task.ref,
            "benchmark_ref": task.benchmark.ref,
            "config": {"depths": list(DEPTHS), "gamma": 0.99, "policy": incumbent_id},
            "seed": 0,
        },
        prior_art=[
            "exp-019-mc-targets-seed21",
            "exp-018-conv3-seed18",
            "exp-010-capacity-probe",
            "exp-009-big-swing",
        ],
        input_policies=[incumbent_id],
        tags=["lookahead", "search", "inference-only", "ladder-v1"],
    )

    results = {"bare": {"score": incumbent.score, "per_rung": incumbent.detail["per_rung"],
                        "frontier_depth": incumbent.detail["frontier_depth"]}}
    print(f"bare: {incumbent.score:.3f} {incumbent.detail['per_rung']}")
    for depth in DEPTHS:
        la = LookaheadQ(base_q, depth=depth)
        t0 = time.time()
        res = task.benchmark.score(la)
        results[f"depth{depth}"] = {
            "score": res.score,
            "per_rung": res.detail["per_rung"],
            "frontier_depth": res.detail["frontier_depth"],
            "leaf_evals": la.leaf_evals,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"depth {depth}: {res.score:.3f} frontier={res.detail['frontier_depth']} "
              f"{res.detail['per_rung']}  ({la.leaf_evals} leaf evals, {time.time()-t0:.0f}s)")

    path = store.telemetry_dir / f"{EXPERIMENT_ID}.json"
    path.write_text(json.dumps({"experiment_id": EXPERIMENT_ID, "task_ref": task.ref,
                                "policy": incumbent_id, "evaluations": results,
                                "iterations": []}, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
