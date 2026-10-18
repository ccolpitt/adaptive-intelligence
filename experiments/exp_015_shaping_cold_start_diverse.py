"""exp-015: package decomposition, step 3 — shaping + cold start + DIVERSE OPPONENTS.

Plain language: the exp-012 package (six changes at once) learned fastest of
anything we've run (0.116 from scratch). Decomposition so far, all at the
same episode budget:
  - shaping + cold start alone:                     0.015  (exp-013, refuted)
  - shaping + cold start + both-players data:       0.060  (exp-014, inconclusive)
  - the full package (adds diverse opponents too):  0.116  (exp-012)
This run tests the OTHER data lever: keep shaping + cold start, add the
varied training-opponent rotation (champion, champion, a scripted
take-the-win/block fixture, random) — but do NOT store the opponent's moves.
Together with exp-014 this completes a clean 2x2: each data lever alone vs
both together, letting us say whether the two levers add up or multiply.

Run:  caffeinate -is python3 experiments/exp_015_shaping_cold_start_diverse.py
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
Evidence chain (matched 15x300-episode budgets): shaping+cold_start alone
stalled at 0.015 (exp-013, refuted); adding both-players data reached 0.060
(exp-014, inconclusive — large contributor, not the whole recipe); the full
package with BOTH data levers hit 0.116 (exp-012). Remaining unmeasured cell
of the 2x2: diverse opponents WITHOUT both-players data.

HYPOTHESIS: opponent diversity is the second necessary ingredient — varied
opponents create the decision-critical situations (wins to take, threats to
block) that make dense tactical reward learnable, which champion-only
self-play under-produces. Exact changes vs exp-013 (the controlled
reference):
  1. diverse_opponents: False -> True (rotation: champion, champion,
     one-step-win fixture, random — exactly as configured in exp-012)
Unchanged from exp-013: tactical_shaping True (0.5), cold_start True,
store_opponent_transitions False, buffer 20000, 15 x 300 episodes,
eps_decay 0.999. Gate v2; plateau-stop armed (patience 6, min_delta 0.02).

PREDICTION: interpreting the 2x2 against exp-013's 0.015 floor and
exp-012's 0.116 ceiling —
  (a) if diverse opponents alone reach >= 0.08: diversity is the primary
      lever and both-players data was the helper;
  (b) if 0.03-0.08 (like exp-014's 0.060): the levers contribute
      comparably and roughly additively; the package's edge is their sum;
  (c) if <= 0.03: diversity NEEDS both-players data (the fixture/random
      games only pay off when the opponent's winning moves also enter the
      buffer) — a superadditive interaction, and the package is minimal.

{DECISION_CRITERIA}"""

    run_loop(
        task,
        store,
        LoopConfig(
            experiment_id="exp-015-shaping-cold-start-diverse",
            hypothesis=hypothesis,
            changes="vs exp-013, one from->to: diverse_opponents F->T; everything "
            "else held at exp-013 values (notably store_opponent_transitions F, "
            "buffer 20000)",
            prior_art=[
                "exp-014-shaping-cold-start-both-players",
                "exp-013-shaping-cold-start",
                "exp-012-signal-package",
                "exp-006-diverse-opponents",
            ],
            tags=["decomposition", "reward-density", "cold-start", "ladder-v1", "gate-v2"],
            seed=15,
            iterations=15,
            episodes_per_iteration=300,
            gate_games=200,
            cold_start=True,
            diverse_opponents=True,
            plateau_action="stop",
            trainer=TrainerConfig(
                seed=15,
                tactical_shaping=True,
            ),
        ),
    )


if __name__ == "__main__":
    main()
