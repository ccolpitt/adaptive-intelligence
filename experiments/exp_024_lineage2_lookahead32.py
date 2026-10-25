"""exp-024: a new line of policies, trained with 32 states of lookahead.

Starts over from a randomly initialised network, and this time the policy is
allowed to look ahead while it practises: before each training move it may
examine up to 32 states, choosing which ones using its own value estimates,
and it is then trained to rate the move it played as the lookahead valued it
— with targets clamped into the range the rewards actually span.

It is measured, as always, playing on its own with no lookahead.

Every cycle writes a record of what was measured, and the record names the
renderer that draws it, so progress can be inspected cycle by cycle and the
whole line can be viewed as one arc.

Run:  caffeinate -is python3 experiments/exp_024_lineage2_lookahead32.py
"""

from _common import DECISION_CRITERIA, Store

from harness.cycle_loop import CycleLoopConfig, run_cycles
from harness.lineage import Lineage, LineageStore, RubricStore, rubric_r002_budgeted_lookahead
from harness.task_repository import TaskRepository, seed_repository
from harness.tasks.connect4_ladder import Connect4LadderTask

LINEAGE_ID = "connect4-L2"
EXPERIMENT_ID = "exp-024-lineage2-lookahead32"
BUDGET = 32
GAMES_PER_CYCLE = 400
CYCLES = 12


def main() -> None:
    store = Store("store")
    root = store.root
    task = Connect4LadderTask()

    tasks = TaskRepository(root / "tasks")
    seed_repository(tasks)
    evaluation = tasks.get_evaluation("connect4")

    rubrics = RubricStore(root / "rubrics")
    rubric = rubrics.publish(rubric_r002_budgeted_lookahead(BUDGET, "puct"))

    lineages = LineageStore(root / "lineages")
    lineages.create(Lineage(
        lineage_id=LINEAGE_ID,
        task_id="connect4",
        rubric_id=rubric.rubric_id,
        description=(
            "Second line of Connect 4 policies. Starts from a randomly "
            "initialised network and is allowed to look ahead while training, "
            "which the first line never was. Begun because the first line "
            "reached 0.116 out of 1.0 across 36 experiments and could not "
            "climb further: it had no mechanism that made each generation's "
            "training data better than the network producing it."
        ),
        task_evaluation_version=evaluation.version,
        parent_lineage="connect4",
        restarted_from_scratch=True,
    ))

    hypothesis = f"""\
CONTEXT: The task is to win Connect 4, scored by {evaluation.ref}: 168 games
against perfect-play opponents that search 1 to 6 moves deep, 28 games
against each, averaged. The policy is a convolutional network of about
183,000 weights that reads the board as two 6x7 planes and rates each of the
seven columns. The first line of policies trained for 36 experiments without
ever looking ahead and reached 0.116 out of a possible 1.0; its best result
did not reproduce when the random seed changed. A measurement without any
training (exp-023) showed that the same network scores 0.116 when it plays
immediately and 0.339 when allowed to examine 400 states before moving, so
its judgement of positions is considerably better than its instant choice of
move. An earlier attempt to train with lookahead (exp-022) failed because the
training targets were unbounded: they were built from the network's own
estimates, which inflated past the range the rewards span until the network
preferred a merely good move to a winning one.

HYPOTHESIS: if the policy is allowed to examine up to {BUDGET} states before
each training move, choosing which ones by its own value estimates, and is
trained to rate the move it played as that lookahead valued it, with targets
clamped to the range the rewards span, then a new line of policies starting
from random weights will exceed the first line's best score of 0.116 within
{CYCLES * GAMES_PER_CYCLE:,} training games, because the move it learns from
is better than the move it would have chosen unaided, and the clamp prevents
the failure recorded in exp-022. Changes from the first line's recipe, each
from -> to: lookahead budget 0 -> {BUDGET} states per move; spending rule
none -> follow the moves the network rates highly; training target from
"reward plus own estimate of the next position" -> "the value the lookahead
reached for the move played"; target clamping off -> on; starting weights
inherited -> random. Measurement is unchanged: the policy alone, no
lookahead.

PREDICTION: the champion's score exceeds 0.116 by cycle {CYCLES}, and the
win rate against a random opponent exceeds 90% within the first four cycles
(a floor the first line took thousands of games to approach). Mechanistic
checks that must also hold, or the result is luck rather than the mechanism:
the average magnitude of the value estimates stays below 1 throughout, which
is what exp-022 lost; and the rate at which the policy takes an available
immediate win rises toward 1, since a 32-state lookahead sees every immediate
win and the network is being trained on what it saw.

{DECISION_CRITERIA}"""

    store.tasks.register(task.definition())
    store.experiments.register(
        experiment_id=EXPERIMENT_ID,
        hypothesis=hypothesis,
        changes=(f"new lineage {LINEAGE_ID} from random weights under rubric "
                 f"{rubric.rubric_id}; each from->to: search_budget 0->{BUDGET}, "
                 f"spend_rule none->puct, target 'td'->'search-value', "
                 f"bounded_targets off->on, warm start->random init; "
                 f"{CYCLES} cycles x {GAMES_PER_CYCLE} games; evaluation unchanged"),
        conditions={
            "task_ref": task.ref,
            "benchmark_ref": task.benchmark.ref,
            "config": {"lineage_id": LINEAGE_ID, "rubric_id": rubric.rubric_id,
                       "task_evaluation": evaluation.ref,
                       "games_per_cycle": GAMES_PER_CYCLE, "cycles": CYCLES,
                       "search_budget": BUDGET, "spend_rule": "puct",
                       "gate_rule": "strict", "renderer": "connect4_v2"},
            "seed": 24,
        },
        prior_art=["exp-023-budget-vs-depth", "exp-022-B-search-targets-seed27",
                   "exp-020-lookahead-inference", "exp-017-recipe-seed-replication"],
        tags=["lineage-2", "train-lookahead", "budget-32", "cycle-records", "gate-v3"],
    )

    records = run_cycles(task, root, CycleLoopConfig(
        lineage_id=LINEAGE_ID,
        rubric_id=rubric.rubric_id,
        games_per_cycle=GAMES_PER_CYCLE,
        cycles=CYCLES,
        gate_games=200,
        random_eval_games=100,
        gate_rule="strict",
        probe_games=30,
        renderer="connect4_v2",
        experiment_id=EXPERIMENT_ID,
        seed=24,
    ))

    print("\nrendering cycles...")
    for record in records:
        print(f"  {record.render()}")

    from harness.learning_arc import render_plot, render_text, save_summary, summarize

    summary = summarize(root, LINEAGE_ID)
    print()
    print(render_text(summary))
    print(f"\narc data:  {save_summary(root, LINEAGE_ID)}")
    print(f"arc plot:  {render_plot(summary)}")


if __name__ == "__main__":
    main()
