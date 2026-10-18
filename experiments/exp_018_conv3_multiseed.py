"""exp-018: third conv layer (board view 5x5 -> 7x7), THREE seeds.

Plain language: the current network's pattern detectors each see at most a
5x5 window of the 6x7 board. Adding a third conv layer widens that to 7x7 —
wide enough to see a full 4-in-a-row plus one empty cell on EACH side, the
shape of an open-ended threat (_XXXX_). Hypothesis (human, 2026-09-19): that
wider local view helps the policy learn threats it currently can't represent
locally. First experiment run under the new multi-seed rule (exp-017): three
seeds of the recipe + conv_layers=3, compared against the recipe's known
three-seed baseline (0.054 / 0.060 / 0.116, mean 0.077).

Run:  caffeinate -is python3 experiments/exp_018_conv3_multiseed.py
"""

from _common import DECISION_CRITERIA, Store, context_paragraph, measure_incumbent

from harness.agents.dqn import TrainerConfig
from harness.improvement_loop import LoopConfig, run_loop
from harness.tasks.connect4_ladder import Connect4LadderTask

SEEDS = (18, 19, 20)
BASELINE_3SEED = (0.054, 0.060, 0.116)  # recipe with conv_layers=2 (exp-016/017/012)


def main() -> None:
    store = Store("store")
    task = Connect4LadderTask()

    for seed in SEEDS:
        incumbent_id, incumbent = measure_incumbent(store, task)
        hypothesis = f"""\
{context_paragraph(incumbent_id, incumbent)}
Evidence chain: the recipe's three-seed baseline is 0.054/0.060/0.116
(mean 0.077, spread 0.062 — exp-016/017/012); per the exp-017 standing rule
this experiment runs three seeds ({SEEDS}) and compares distributions, not
single numbers. exp-010 nuance, stated up front: a 3-conv net showed NO
advantage when FITTING solver labels, so the claim here is specifically that
the wider receptive field helps the RL loop learn from its own experience —
optimization, not capacity.

HYPOTHESIS: widening each detector's view from 5x5 to 7x7 (third conv layer)
lets the net locally represent open-ended threats (a 4-window plus one cell
on each side), improving learning from tactical-shaped self-play. Exact
change vs the recipe (exp-016/017 configuration): conv_layers 2 -> 3.
This arm: seed {seed} of {SEEDS}.

PREDICTION (scored over all three seeds jointly): the conv3 three-seed mean
exceeds the baseline mean 0.077 by at least the seed spread (i.e. mean >=
0.14), OR at least two of three seeds exceed 0.116. Anything less is
inconclusive-or-refuted: a shift smaller than the spread is noise by the
exp-017 rule.

{DECISION_CRITERIA}"""

        run_loop(
            task,
            store,
            LoopConfig(
                experiment_id=f"exp-018-conv3-seed{seed}",
                hypothesis=hypothesis,
                changes="vs recipe (exp-016/017 config), one from->to: conv_layers "
                f"2->3; this arm is seed {seed} of {SEEDS} (multi-seed rule)",
                prior_art=[
                    "exp-017-recipe-seed-replication",
                    "exp-016-recipe-ceiling",
                    "exp-012-signal-package",
                    "exp-010-capacity-probe",
                ],
                tags=["conv3", "receptive-field", "multi-seed", "ladder-v1", "gate-v2"],
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
                    conv_layers=3,
                ),
            ),
        )


if __name__ == "__main__":
    main()
