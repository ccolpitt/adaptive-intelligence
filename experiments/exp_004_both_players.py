"""exp-004: store BOTH players' transitions, not just the challenger's.

Run:  python3 experiments/exp_004_both_players.py
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

HYPOTHESIS: Storing BOTH players' transitions in the replay buffer (instead
of only the challenger's own moves) will raise absolute strength. Q-learning
is off-policy - a transition is a fact about the game no matter who chose the
move - so the champion's moves are valid training data, and they double the
data per game while injecting positions the challenger's own habits never
produce.

PREDICTION: If true, we expect the final champion's ladder score to be at
least 0.09 above the incumbent's ({incumbent.score:.3f}), because twice the
data with broader position coverage should at minimum match the precursor
project, where both-player storage was part of every configuration that
learned successfully.

{DECISION_CRITERIA}"""

    run_loop(
        task,
        store,
        LoopConfig(
            experiment_id="exp-004-both-players-transitions",
            hypothesis=hypothesis,
            changes="replay data only: store_opponent_transitions=True; nothing else",
            prior_art=["exp-002-more-self-play", "exp-003-frontier-training-mix"],
            tags=["replay-data", "off-policy", "ladder-v1"],
            seed=4,
            iterations=5,
            episodes_per_iteration=300,
            gate_games=200,
            trainer=TrainerConfig(seed=4, store_opponent_transitions=True),
        ),
    )


if __name__ == "__main__":
    main()
