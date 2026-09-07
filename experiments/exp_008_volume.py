"""exp-008: 3x training volume with a slower exploration decay.

Run:  python3 experiments/exp_008_volume.py
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

HYPOTHESIS: Tripling training volume (15 iterations x 300 = 4500 episodes,
vs 1500 in every experiment so far) with a proportionally slower epsilon
decay (0.9995 per episode instead of 0.999) will raise absolute strength.
The precursor's clearest lesson was that training volume beat every clever
component change; our runs so far are 4-60x shorter than its successful ones.

PREDICTION: If true, we expect the final champion's ladder score to be at
least 0.09 above the incumbent's ({incumbent.score:.3f}), because the policy
is nowhere near converged and volume is the variable with the strongest prior
evidence behind it.

{DECISION_CRITERIA}"""

    run_loop(
        task,
        store,
        LoopConfig(
            experiment_id="exp-008-training-volume-3x",
            hypothesis=hypothesis,
            changes="training volume + exploration schedule only: 15x300 episodes, "
            "eps_decay 0.999 -> 0.9995; nothing else",
            prior_art=["exp-002-more-self-play"],
            tags=["training-volume", "exploration", "ladder-v1"],
            seed=8,
            iterations=15,
            episodes_per_iteration=300,
            gate_games=200,
            trainer=TrainerConfig(seed=8, eps_decay=0.9995),
        ),
    )


if __name__ == "__main__":
    main()
