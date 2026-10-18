"""exp-017: pure replication — same recipe, new random seed.

Plain language: the identical training setup produced 0.116 at seed 12
(exp-012) and 0.054 at seed 16 (exp-016). This run changes nothing except
the random seed (now 17), to measure how much of our results is luck.
After it we have three seeds of the same recipe and can quote a mean and a
spread instead of trusting any single number.

Run:  caffeinate -is python3 experiments/exp_017_recipe_seed_replication.py
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
Evidence chain: the identical recipe (tactical shaping 0.5 + cold start +
both-players data + diverse opponents, 300 episodes/iteration) has now run
at two seeds with wildly different results: 0.116 at seed 12 (exp-012) and
0.054 at seed 16 (exp-016, plateau-stopped at iter 7). One of these is not
like the other, and nothing changed but the random seed. Either exp-012 was
a lucky draw, exp-016 an unlucky one, or run-to-run variance in this regime
is simply comparable to every effect we have been measuring.

HYPOTHESIS (replication, no new lever): the recipe's expected best-ladder
value across seeds is materially below exp-012's 0.116 — i.e. seed variance,
not the recipe, explains the gap between 0.116 and 0.054. Exact change vs
exp-016: seed 16 -> 17. Nothing else.

PREDICTION: seed 17's best challenger ladder lands in [0.03, 0.09] — nearer
exp-016 than exp-012. If instead it reaches >= 0.10, exp-016 was the
outlier and the recipe's expected value is high with occasional failures;
either way, after this run we have three seeds and can state a mean and a
spread instead of a single number. DECISION CONSEQUENCE: all future lever
comparisons in this regime must either exceed the observed seed spread or
run multiple seeds — single-seed deltas smaller than that spread are noise.

{DECISION_CRITERIA}"""

    run_loop(
        task,
        store,
        LoopConfig(
            experiment_id="exp-017-recipe-seed-replication",
            hypothesis=hypothesis,
            changes="vs exp-016, one from->to: seed 16->17; recipe, budget, gate, "
            "and plateau settings all identical (pure replication)",
            prior_art=[
                "exp-016-recipe-ceiling",
                "exp-012-signal-package",
            ],
            tags=["recipe", "replication", "seed-variance", "ladder-v1", "gate-v2"],
            seed=17,
            iterations=30,
            episodes_per_iteration=300,
            gate_games=200,
            cold_start=True,
            diverse_opponents=True,
            plateau_action="stop",
            trainer=TrainerConfig(
                seed=17,
                tactical_shaping=True,
                store_opponent_transitions=True,
                buffer_capacity=50000,
            ),
        ),
    )


if __name__ == "__main__":
    main()
