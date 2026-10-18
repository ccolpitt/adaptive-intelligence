"""exp-014: package decomposition, step 2 — shaping + cold start + BOTH-PLAYERS DATA.

Plain language: exp-012 bundled six changes and learned fastest of anything
we've run (ladder 0.116 from scratch). exp-013 kept only two of them —
tactical penalty rewards ("shaping") and starting from random weights
("cold start") — and stalled at 0.015, proving those two alone are not the
recipe. This run adds back ONE more ingredient: also storing the opponent's
moves as training data (which doubles data per game and, per exp-004, was
the best single lever). If this run learns fast again, we know the recipe is
shaping + cold start + both-players data, and the fancy opponent rotation
was unnecessary. If it still stalls, the diverse-opponent rotation is the
missing ingredient.

Run:  caffeinate -is python3 experiments/exp_014_shaping_cold_start_both_players.py
"""

from _common import DECISION_CRITERIA, Store, context_paragraph, measure_incumbent

from harness.agents.dqn import TrainerConfig
from harness.improvement_loop import LoopConfig, run_loop
from harness.tasks.connect4_ladder import Connect4LadderTask


def main() -> None:
    store = Store("store")
    task = Connect4LadderTask()
    incumbent_id, incumbent = measure_incumbent(store, task)

    hypothesis = f"""\
{context_paragraph(incumbent_id, incumbent)}
Evidence chain: exp-012's six-change package hit ladder 0.116 from cold start
in ~3600 episodes (program record). exp-013 (shaping + cold start ONLY, all
data levers off) stalled at 0.015 in 4500 episodes — REFUTED the claim that
reward density alone is the active ingredient. The remaining candidates are
the two data levers: both-players transitions (exp-004: +0.026 solo, best
single lever) and diverse opponents (exp-006: strongest h2h growth).

HYPOTHESIS: shaping + cold start + both-players data is the package's
minimal active recipe — dense tactical reward signal needs a dense, two-
sided view of the game to learn from, not merely time. Exact changes vs
exp-013 (which is the controlled reference at the same budget):
  1. store_opponent_transitions: False -> True (both seats' moves stored;
     Bellman targets computed by our net either way)
  2. buffer_capacity: 20000 -> 50000 (paired with #1 exactly as in exp-012,
     since both-players doubles the data rate; kept as one composite lever
     "both-players data as configured in the package")
Unchanged from exp-013: tactical_shaping True (0.5), cold_start True,
15 x 300 episodes, eps_decay 0.999, diverse_opponents False.

PREDICTION: if the recipe holds, best challenger ladder >= 0.08 (most of the
package's 0.116, from half its levers) with rung-1 >= 0.40. If it stalls
<= 0.04, diverse opponents are necessary and exp-015 adds them instead.
Gate note: FIRST RUN under gate v2 (ADR-005, either-path promotion) — an
absolutely-superior challenger at h2h parity can now promote. Plateau stop
is armed (patience 6, min_delta 0.02): if six straight iterations buy
nothing, the run ends itself and records the savings.

{DECISION_CRITERIA}"""

    run_loop(
        task,
        store,
        LoopConfig(
            experiment_id="exp-014-shaping-cold-start-both-players",
            hypothesis=hypothesis,
            changes="vs exp-013, each from->to: store_opponent_transitions F->T, "
            "buffer_capacity 20000->50000 (composite 'both-players data' lever as "
            "configured in exp-012); everything else held at exp-013 values",
            prior_art=[
                "exp-013-shaping-cold-start",
                "exp-012-signal-package",
                "exp-004-both-players-transitions",
            ],
            tags=["decomposition", "reward-density", "cold-start", "ladder-v1", "gate-v2"],
            seed=14,
            iterations=15,
            episodes_per_iteration=300,
            gate_games=200,
            cold_start=True,
            plateau_action="stop",
            trainer=TrainerConfig(
                seed=14,
                tactical_shaping=True,
                store_opponent_transitions=True,
                buffer_capacity=50000,
            ),
        ),
    )


if __name__ == "__main__":
    main()
