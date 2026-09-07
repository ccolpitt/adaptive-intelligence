"""exp-011: tactical shaping - immediate penalties for the two
decision-critical blunders (foregone win, failed block).

Run:  python3 experiments/exp_011_tactical_shaping.py
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
exp-010 (capacity probe) established that the current network can represent
tactics perfectly (100% train fit on solver labels); the bottleneck must be
the training signal. Today the only signals are +1 on the winning move and
-1 on the loser's final move; a FOREGONE WIN produces no signal at all, and
a failed block is punished only at game end.

HYPOTHESIS: Adding an immediate training-time penalty of -0.5 to exactly two
blunders - (a) mover had an immediate winning column and played something
else, (b) opponent had an immediate winning column and mover did not block
it - will convert training into absolute strength. Change is training-signal
only: tactical_shaping False -> True, shaping_penalty 0.5; the task reward,
benchmark, and all evaluation remain pure win/lose (the ladder guards
against reward hacking). Everything else identical to exp-002's baseline:
5 iterations x 300 episodes, seed 11, eps 0.5->0.1 decay 0.999, buffer
20000, uniform sampling, champion-only self-play opponent.

PREDICTION: If true, we expect the final champion's ladder aggregate to gain
at least +0.09 over the incumbent's {incumbent.score:.3f}, concentrated in
rung 1 (predict rung-1 >= 0.55 from 0.42), because the depth-1 solver wins
almost exclusively by punishing exactly these two blunders, and they now
carry immediate, dense, unmissable signal instead of sparse delayed signal.
This also addresses the babies-teach-babies bootstrap: the shaping signal
comes from the ENVIRONMENT's tactical facts, not from the (weak) opponent,
so signal quality no longer depends on champion quality.

{DECISION_CRITERIA}"""

    run_loop(
        task,
        store,
        LoopConfig(
            experiment_id="exp-011-tactical-shaping",
            hypothesis=hypothesis,
            changes="training signal only: tactical_shaping False -> True "
            "(shaping_penalty 0.5); all else = exp-002 baseline",
            prior_art=["exp-010-capacity-probe", "exp-007-terminal-quota"],
            tags=["reward-density", "shaping", "signal", "ladder-v1"],
            seed=11,
            iterations=5,
            episodes_per_iteration=300,
            gate_games=200,
            trainer=TrainerConfig(seed=11, tactical_shaping=True),
        ),
    )


if __name__ == "__main__":
    main()
