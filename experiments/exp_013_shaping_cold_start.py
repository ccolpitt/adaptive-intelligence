"""exp-013: decompose the exp-012 package — shaping + cold start ALONE.

exp-012's package (shaping + cold_start + both-players data + diverse
opponents + 3x episodes + big buffer) produced the fastest absolute learning
curve of the program: a cold-started policy hit ladder 0.116 in ~3600
episodes. A package verdict credits the package only; this run keeps ONLY the
reward-density pair and returns every data lever to baseline, to learn which
ingredient carries the gain.

Run:  caffeinate -is python3 experiments/exp_013_shaping_cold_start.py
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
Evidence chain: exp-011 showed shaping FAILS when injected into a warm-started
value function calibrated on the pure win/lose scale (h2h collapsed 0.27-0.34;
suspected re-scaling shock). exp-012's package — where the value function was
BORN on the shaped scale via cold start — produced the program's fastest
absolute learning curve (ladder 0.116 in ~3600 episodes vs the incumbent's
0.083 in ~10500). The package verdict is deliberately unattributed; the named
first decomposition is shaping+cold_start alone.

HYPOTHESIS: dense decision-critical reward (tactical shaping) is the ACTIVE
ingredient of the exp-012 package, and cold start is its necessary enabler —
the data levers (both-players transitions, diverse opponents, 50k buffer)
were secondary. Exact changes vs the exp-001 baseline, each from -> to:
  1. tactical_shaping: False -> True, shaping_penalty 0.5 (unchanged from
     exp-012)
  2. cold_start: False -> True (unchanged from exp-012)
  3. episodes: 1500 -> 4500 (15 x 300 — matches exp-012's time budget, so a
     difference in outcome is attributable to the removed data levers, not
     to training time)
EXPLICITLY RETURNED TO BASELINE (present in exp-012, absent here):
store_opponent_transitions (False), diverse_opponents (False),
buffer_capacity (20000).

PREDICTION: if reward density is the active ingredient, the best challenger
reaches ladder >= 0.10 with rung-1 >= 0.50 within 15 iterations,
reproducing most of the package's 0.116. If it stalls at <= 0.06, the data
levers are jointly necessary and the next decomposition adds them back one
at a time. Either result apportions the package's credit.

MEASUREMENT NOTE: the promotion gate may block cold-started challengers on
h2h style mismatch (the exp-012 iter-11 finding, gate-design question still
open with the human); per-iteration benchmark telemetry captures the
learning curve REGARDLESS of promotions, so the hypothesis is scored from
telemetry, not from the gate. This is also the first run with per-episode
telemetry (loss/grad-norm/result/unique-states/epsilon) and gradient norms.

{DECISION_CRITERIA}"""

    run_loop(
        task,
        store,
        LoopConfig(
            experiment_id="exp-013-shaping-cold-start",
            hypothesis=hypothesis,
            changes="vs baseline, each from->to: tactical_shaping F->T (penalty 0.5), "
            "cold_start F->T, episodes 1500->4500; all exp-012 data levers returned "
            "to baseline (store_opponent_transitions F, diverse_opponents F, "
            "buffer 20000)",
            prior_art=[
                "exp-011-tactical-shaping",
                "exp-012-signal-package",
                "exp-010-capacity-probe",
            ],
            tags=["decomposition", "reward-density", "cold-start", "ladder-v1"],
            seed=13,
            iterations=15,
            episodes_per_iteration=300,
            gate_games=200,
            cold_start=True,
            trainer=TrainerConfig(
                seed=13,
                tactical_shaping=True,
            ),
        ),
    )


if __name__ == "__main__":
    main()


