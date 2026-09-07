"""exp-009: the big swing - combine the evidence-backed changes.

Run:  python3 experiments/exp_009_big_swing.py
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

HYPOTHESIS: The two positive-direction single changes combine with honest
training volume to produce a real absolute gain. Package under test:
(1) both-players transitions (exp-004: +0.026 alone, best single result),
(2) diverse opponents (exp-006: +0.003 ladder but the strongest h2h growth),
(3) 15 iterations x 300 episodes at the ORIGINAL eps decay 0.999 - retesting
volume cleanly, since exp-008 confounded volume with a slower decay that kept
the challenger half-random,
(4) replay buffer 20k -> 50k, because both-players doubles the data rate and
exp-005's failure suggested buffer capacity interacts with data volume.
EXCLUDED as refuted alone: mirror (exp-005), terminal quota (exp-007),
frontier-solver training (exp-003).

PREDICTION: If true, we expect the final champion's ladder score to be at
least 0.09 above the incumbent's ({incumbent.score:.3f}), because the two
data changes address the diagnosed data-poverty mechanism and volume gives
them room to compound.

ATTRIBUTION NOTE: this is a PACKAGE experiment - a win credits the package,
not any single ingredient; single-variable follow-ups would apportion credit.
A loss with these four together would say the bottleneck lies elsewhere
(reward density, architecture, or search at inference).

{DECISION_CRITERIA}"""

    run_loop(
        task,
        store,
        LoopConfig(
            experiment_id="exp-009-big-swing",
            hypothesis=hypothesis,
            changes="package: store_opponent_transitions=True + diverse_opponents=True "
            "+ 15x300 episodes at eps_decay 0.999 + buffer 50k",
            prior_art=[
                "exp-004-both-players-transitions",
                "exp-006-diverse-opponents",
                "exp-008-training-volume-3x",
                "exp-005-mirror-augmentation",
                "exp-007-terminal-quota",
            ],
            tags=["package", "big-swing", "ladder-v1"],
            seed=9,
            iterations=15,
            episodes_per_iteration=300,
            gate_games=200,
            diverse_opponents=True,
            trainer=TrainerConfig(
                seed=9, store_opponent_transitions=True, buffer_capacity=50000
            ),
        ),
    )


if __name__ == "__main__":
    main()
