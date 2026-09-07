"""exp-006: diverse training opponents at learnable difficulty.

Run:  python3 experiments/exp_006_diverse_opponents.py
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

HYPOTHESIS: Cycling training opponents - champion, champion, tactical fixture
(takes wins / blocks losses), random - will raise absolute strength where
champion-only self-play (exp-002) plateaued. exp-002 showed self-play data
lacks competent threats; exp-003 showed the solver is too strong to learn
from (all losses). The fixture sits between: it punishes every unblocked
3-line without being unbeatable, and random keeps coverage broad.

PREDICTION: If true, we expect the final champion's ladder score to be at
least 0.09 above the incumbent's ({incumbent.score:.3f}), concentrated in
rung 1, because the fixture supplies exactly the immediate-threat lessons the
depth-1 solver grades, at a difficulty the policy can win against ~half the
time (exp-003's zone-of-proximal-development lesson).

{DECISION_CRITERIA}"""

    run_loop(
        task,
        store,
        LoopConfig(
            experiment_id="exp-006-diverse-opponents",
            hypothesis=hypothesis,
            changes="mode of learning only: diverse_opponents=True "
            "(cycle champion/champion/tactic-fixture/random); nothing else",
            prior_art=["exp-002-more-self-play", "exp-003-frontier-training-mix"],
            tags=["mode-of-learning", "opponent-diversity", "ladder-v1"],
            seed=6,
            iterations=5,
            episodes_per_iteration=300,
            gate_games=200,
            diverse_opponents=True,
            trainer=TrainerConfig(seed=6),
        ),
    )


if __name__ == "__main__":
    main()
