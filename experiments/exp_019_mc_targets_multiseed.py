"""exp-019: Monte-Carlo credit propagation (human-proposed), THREE seeds.

Plain language: today, only the final move of a game carries a reward; every
earlier move is stored with reward 0, and the outcome seeps backward one
step per replay through bootstrapped targets — slow, and the cards show it
(value horizon ~ 0 plies). This lever labels EVERY move at game end with the
final outcome discounted back to it: winner's last move +1, the move before
it -0.99, the one before that +0.98, and so on. Credit reaches move 1
immediately. The cost: early moves get credited/blamed for outcomes they
barely influenced (higher variance). Three seeds, judged on the
distribution.

Run:  caffeinate -is python3 experiments/exp_019_mc_targets_multiseed.py
"""

from _common import DECISION_CRITERIA, Store, context_paragraph, measure_incumbent

from harness.agents.dqn import TrainerConfig
from harness.improvement_loop import LoopConfig, run_loop
from harness.tasks.connect4_ladder import Connect4LadderTask

SEEDS = (21, 22, 23)
BASELINE_3SEED = (0.054, 0.060, 0.116)  # recipe, TD targets (exp-016/017/012)


def main() -> None:
    store = Store("store")
    task = Connect4LadderTask()

    for seed in SEEDS:
        incumbent_id, incumbent = measure_incumbent(store, task)
        hypothesis = f"""\
{context_paragraph(incumbent_id, incumbent)}
Evidence chain: performance cards show value_horizon ~ 0 plies — the game
outcome is NOT propagating backward through the value function under
one-step TD targets, despite calibrated endgame Q (solver-proven wins read
+0.66, proven losses -0.67 on the champion card). The recipe's 3-seed TD
baseline is {BASELINE_3SEED} (mean 0.077, spread 0.062).

HYPOTHESIS (human-proposed): the propagation LAG is a binding constraint.
Labeling every stored transition with the discounted final outcome
(mc_targets: reward_i = outcome_for_mover * gamma^(plies_from_end) + its
shaping delta, done=True — supervised targets, no bootstrapping) delivers
credit to early moves immediately and lifts the ladder. Exact change vs the
recipe (exp-016/017 config): trainer.mc_targets False -> True. This arm:
seed {seed} of {SEEDS}.

PREDICTION (scored over the three seeds jointly): mc-target 3-seed mean
>= 0.14 (baseline mean + spread), OR at least 2/3 seeds > 0.116.
MECHANISTIC CHECK regardless of ladder outcome: value_horizon_plies on
exp-019 policy cards should be MUCH greater than 0 — if the horizon does
not lengthen, the lever failed mechanically, not just statistically.
Known risk, stated up front: MC labels have higher variance (early moves
credited for barely-related outcomes), so a NEGATIVE result with a
lengthened horizon implicates variance, suggesting n-step middle ground.

{DECISION_CRITERIA}"""

        run_loop(
            task,
            store,
            LoopConfig(
                experiment_id=f"exp-019-mc-targets-seed{seed}",
                hypothesis=hypothesis,
                changes=f"vs recipe (exp-016/017 config), one from->to: "
                f"trainer.mc_targets False->True; this arm is seed {seed} of {SEEDS}",
                prior_art=[
                    "exp-018-conv3-seed18",
                    "exp-017-recipe-seed-replication",
                    "exp-016-recipe-ceiling",
                    "exp-012-signal-package",
                ],
                tags=["mc-targets", "credit-propagation", "multi-seed", "ladder-v1", "gate-v2"],
                seed=seed,
                iterations=30,
                episodes_per_iteration=300,
                gate_games=200,
                cold_start=True,
                diverse_opponents=True,
                plateau_action="stop",
                trainer=TrainerConfig(
                    seed=seed,
                    tactical_shaping=True,
                    store_opponent_transitions=True,
                    buffer_capacity=50000,
                    mc_targets=True,
                ),
            ),
        )


if __name__ == "__main__":
    main()
