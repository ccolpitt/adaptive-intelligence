"""exp-016: run the proven recipe to its ceiling, under the new gate.

Plain language: the 2x2 decomposition (exp-013/014/015) proved the exp-012
package is the minimal recipe — tactical penalty rewards + fresh random
weights + storing both players' moves + varied opponents. All levers are
needed; the two data levers multiply rather than add. But exp-012 ran only
15 iterations under the OLD gate, which couldn't promote its best policy
(benchmark-stronger, head-to-head-different). This run asks: how far does
the recipe actually go? Same recipe, doubled iteration budget, the new
either-path gate (ADR-005), and the plateau detector deciding when to stop
instead of a guessed episode count. Success = a promoted successor champion
and a measured ceiling.

Run:  caffeinate -is python3 experiments/exp_016_recipe_ceiling.py
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
Evidence chain: the 2x2 decomposition at matched budgets — shaping+cold_start
0.015 (exp-013), +both-players 0.060 (exp-014), +diverse-only 0.018
(exp-015), full package 0.116 (exp-012) — proves the package is minimal and
its data levers are superadditive: diversity manufactures tactical lessons,
both-players data records them. exp-012 was budget-truncated (15 iters) and
gate-blocked (old double gate); its learning curve was still rising at the
end.

HYPOTHESIS: the recipe's ceiling is materially higher than 0.116, and under
gate v2 a recipe-trained policy promotes to champion. Exact changes vs
exp-012 (the controlled reference):
  1. iterations: 15 -> 30 (plateau-stop decides the real endpoint;
     patience 6, min_delta 0.02)
  2. gate_rule: double -> either (ADR-005 — first recipe run where an
     absolutely-superior challenger at h2h parity CAN promote)
Everything else exactly as exp-012: tactical_shaping (0.5), cold_start,
store_opponent_transitions, diverse_opponents, buffer 50000, 300
episodes/iteration, eps_decay 0.999. Seed 16.

PREDICTION: (a) best challenger ladder >= 0.15 (clears exp-012's 0.116 by
more than one detectable effect at n=168), and (b) at least one promotion
occurs under gate v2. If the curve plateau-stops at ~0.116 instead, the
recipe is signal-saturated and the next bottleneck is the one exp-009 named
and no data lever can fix: reflex-only inference — the policy answers in
one forward pass with no lookahead — pointing the program at search
(ADR-003 direction).

{DECISION_CRITERIA}"""

    run_loop(
        task,
        store,
        LoopConfig(
            experiment_id="exp-016-recipe-ceiling",
            hypothesis=hypothesis,
            changes="vs exp-012, each from->to: iterations 15->30 (plateau-stop "
            "armed), gate_rule double->either (ADR-005); recipe itself unchanged",
            prior_art=[
                "exp-015-shaping-cold-start-diverse",
                "exp-014-shaping-cold-start-both-players",
                "exp-013-shaping-cold-start",
                "exp-012-signal-package",
            ],
            tags=["recipe", "ceiling-probe", "ladder-v1", "gate-v2", "plateau-stop"],
            seed=16,
            iterations=30,
            episodes_per_iteration=300,
            gate_games=200,
            cold_start=True,
            diverse_opponents=True,
            plateau_action="stop",
            trainer=TrainerConfig(
                seed=16,
                tactical_shaping=True,
                store_opponent_transitions=True,
                buffer_capacity=50000,
            ),
        ),
    )


if __name__ == "__main__":
    main()
