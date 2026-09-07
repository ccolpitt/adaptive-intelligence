"""exp-007: guarantee 30% of every training batch is terminal transitions.

Run:  python3 experiments/exp_007_terminal_quota.py
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

HYPOTHESIS: Guaranteeing 30% of every training batch comes from terminal
transitions will raise absolute strength. All reward signal (+1/-1) lives at
game ends, but terminals are only ~10% of a uniform buffer, so most batches
carry almost no reward gradient; the value function starves. The precursor
diagnosed exactly this and shipped a 30% terminal quota in every successful
configuration.

PREDICTION: If true, we expect the final champion's ladder score to be at
least 0.09 above the incumbent's ({incumbent.score:.3f}), because tripling
the frequency of reward-bearing samples should speed credit propagation from
game ends back through the gamma-chain.

{DECISION_CRITERIA}"""

    run_loop(
        task,
        store,
        LoopConfig(
            experiment_id="exp-007-terminal-quota",
            hypothesis=hypothesis,
            changes="replay sampling only: terminal_fraction=0.3; nothing else",
            prior_art=["exp-004-both-players-transitions"],
            tags=["replay-sampling", "credit-assignment", "ladder-v1"],
            seed=7,
            iterations=5,
            episodes_per_iteration=300,
            gate_games=200,
            trainer=TrainerConfig(seed=7, terminal_fraction=0.3),
        ),
    )


if __name__ == "__main__":
    main()
