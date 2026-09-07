"""exp-005: left-right mirror augmentation.

Run:  python3 experiments/exp_005_mirror.py
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

HYPOTHESIS: Adding every transition's left-right mirror image to the replay
buffer will raise absolute strength. Connect 4 is exactly symmetric under
reflection, so each game legitimately teaches two games; the network stops
having to learn the same tactic separately on the left and right sides.

PREDICTION: If true, we expect the final champion's ladder score to be at
least 0.09 above the incumbent's ({incumbent.score:.3f}), because doubling
effective data at this early training stage should convert directly into
strength (the precursor used symmetric adds in every successful run).

{DECISION_CRITERIA}"""

    run_loop(
        task,
        store,
        LoopConfig(
            experiment_id="exp-005-mirror-augmentation",
            hypothesis=hypothesis,
            changes="replay data only: mirror_augmentation=True; nothing else",
            prior_art=["exp-004-both-players-transitions"],
            tags=["replay-data", "augmentation", "symmetry", "ladder-v1"],
            seed=5,
            iterations=5,
            episodes_per_iteration=300,
            gate_games=200,
            trainer=TrainerConfig(seed=5, mirror_augmentation=True),
        ),
    )


if __name__ == "__main__":
    main()
