"""exp-012: the signal package - dense tactical feedback, learned from
scratch, with rich data and time to converge.

Run:  python3 experiments/exp_012_signal_package.py
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
Evidence chain: exp-010 proved capacity is not the bottleneck (current net
fits solver labels to 100% train accuracy); exp-011 showed shaping alone
fails when a warm-started value function calibrated on pure win/lose must
suddenly absorb a re-scaled reward, with too few episodes to recalibrate.

HYPOTHESIS (necessary-conditions package): for the policy to improve, it
needs BOTH dense decision-critical feedback AND enough clean data/time for
that signal to shape the value function from the start. Exact changes, each
from -> to:
  1. tactical_shaping: False -> True, shaping_penalty 0.5 (immediate -0.5
     for a foregone win; -0.5 for a failed block; training-signal only)
  2. cold_start: False -> True (challenger initialized from random weights,
     so its value function is BORN on the shaped scale; the champion still
     serves as opponent and gate incumbent)
  3. store_opponent_transitions: False -> True (both seats' moves as
     off-policy data; +0.026 alone in exp-004, best single result)
  4. diverse_opponents: False -> True (cycle champion/champion/tactical-
     fixture/random; strongest h2h growth in exp-006)
  5. episodes: 1500 -> 4500 (15 iterations x 300; eps_decay stays 0.999,
     avoiding exp-008's slow-decay confound)
  6. buffer_capacity: 20000 -> 50000 (both-players doubles data rate)
EXCLUDED, refuted alone: mirror (exp-005), terminal quota (exp-007),
frontier-solver training (exp-003), bigger network (exp-010: no benefit).

PREDICTION: If true, we expect the final champion's ladder aggregate to gain
at least +0.09 over the incumbent's {incumbent.score:.3f}, with rung-1
>= 0.60 (from 0.34), because the depth-1 solver punishes exactly the two
blunders that now carry immediate signal, and a cold-started value function
can absorb that signal without fighting a stale calibration. A FAILURE of
this package - signal + data + time all present, capacity proven sufficient -
would point the investigation at the remaining suspect: reflex-only
inference (no lookahead), i.e. the ADR-003 search direction.

ATTRIBUTION NOTE: package verdict credits/blames the package only;
single-variable follow-ups (especially shaping+cold_start alone) apportion
credit if it wins.

{DECISION_CRITERIA}"""

    run_loop(
        task,
        store,
        LoopConfig(
            experiment_id="exp-012-signal-package",
            hypothesis=hypothesis,
            changes="package, each from->to: tactical_shaping F->T (0.5), cold_start "
            "F->T, store_opponent_transitions F->T, diverse_opponents F->T, episodes "
            "1500->4500, buffer 20000->50000; eps_decay unchanged 0.999",
            prior_art=[
                "exp-010-capacity-probe",
                "exp-011-tactical-shaping",
                "exp-004-both-players-transitions",
                "exp-006-diverse-opponents",
            ],
            tags=["package", "reward-density", "cold-start", "ladder-v1"],
            seed=12,
            iterations=15,
            episodes_per_iteration=300,
            gate_games=200,
            diverse_opponents=True,
            cold_start=True,
            trainer=TrainerConfig(
                seed=12,
                tactical_shaping=True,
                store_opponent_transitions=True,
                buffer_capacity=50000,
            ),
        ),
    )


if __name__ == "__main__":
    main()
