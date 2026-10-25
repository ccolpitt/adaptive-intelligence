"""exp-025: bound the value estimates by construction, and look further ahead.

exp-024 trained a new line of policies with 32 states of lookahead per move
and clamped its training targets. It reached 0.068 of a possible 1.0 in 4,800
games, and two things went wrong in a way the cards made visible:

  - the estimates drifted past the range the rewards span (reaching 1.93)
    even though the targets were clamped, because clamping the target does
    not stop the network extrapolating beyond it;
  - it still missed about half of all forced blocks, which a 32-state
    lookahead would always see — so what the lookahead found was not making
    it into the weights.

This run changes two things. The value range becomes a property of the
architecture rather than a hope, by squashing the output; and the lookahead
budget rises from 32 to 128 states, which an earlier measurement (exp-023)
showed is where the search starts finding things a smaller budget misses.

Long-running and resumable: cycle records and the champion live on disk, so
an interrupted run continues from the last completed cycle.

Run:      caffeinate -is python3 experiments/exp_025_bounded_values.py
Resume:   the same command — it picks up where it stopped.
"""

from _common import DECISION_CRITERIA, Store

from harness.cycle_loop import CycleLoopConfig, run_cycles
from harness.lineage import (
    Lineage,
    LineageStore,
    RubricStore,
    TrainingRubric,
    rubric_r002_budgeted_lookahead,
)
from harness.task_repository import TaskRepository, seed_repository
from harness.tasks.connect4_ladder import Connect4LadderTask

LINEAGE_ID = "connect4-L3"
EXPERIMENT_ID = "exp-025-bounded-values"
BUDGET = 128
EVAL_BUDGET = 128          # the policy may also think when it is scored
GAMES_PER_CYCLE = 400
CYCLES = 40                # plateau-free budget; resumable, so it can run for days


def rubric_r003() -> TrainingRubric:
    base = rubric_r002_budgeted_lookahead(BUDGET, "puct")
    return TrainingRubric(
        rubric_id=f"r003-bounded-lookahead-{BUDGET}",
        name=f"Bounded value estimates, lookahead of {BUDGET} states per move",
        description=(
            f"As r002 — before each training move the policy may examine up to "
            f"{BUDGET} states, choosing them with its own value estimates, then "
            f"plays what the lookahead favours and is trained to rate that move "
            f"as the lookahead valued it — with one architectural change: the "
            f"network's output is squashed so that no estimate can leave the "
            f"range the rewards span. Rewards are +1 and -1 and future value is "
            f"discounted, so no position can truly be worth more than 1; "
            f"clamping the targets alone did not prevent the network "
            f"extrapolating past that range in exp-024, and once estimates "
            f"exceed 1, a merely good move can outrank a winning one, whose "
            f"target is exactly 1."
        ),
        learner="dqn",
        search_budget=BUDGET,
        spend_rule="puct",
        target_kind="search-value",
        bounded_targets=True,
        hyperparameters={**base.hyperparameters, "bounded_output": True},
        data_rules={**base.data_rules, "note": "as r002; the change is the "
                                               "bounded output, not the data"},
        notes=(
            "Second attempt at putting lookahead into training. The first "
            "(exp-024, rubric r002) produced a value function that read "
            "proven-won positions at +0.93 and proven-lost at -0.83, so the "
            "mechanism was partly working; what failed was move preference, "
            "which is what an unbounded value scale destroys."
        ),
    )


def main() -> None:
    store = Store("store")
    root = store.root
    task = Connect4LadderTask()

    tasks = TaskRepository(root / "tasks")
    seed_repository(tasks)
    evaluation = tasks.get_evaluation("connect4")

    rubric = RubricStore(root / "rubrics").publish(rubric_r003())

    lineages = LineageStore(root / "lineages")
    lineages.create(Lineage(
        lineage_id=LINEAGE_ID,
        task_id="connect4",
        rubric_id=rubric.rubric_id,
        description=(
            "Third line of Connect 4 policies. Lookahead during training at "
            f"{BUDGET} states per move, and value estimates bounded by the "
            "architecture so they cannot leave the range the rewards span. "
            "Started fresh because the change is architectural: a bounded "
            "network and an unbounded one are not the same network, so "
            "continuing from the second line's weights would confound the two."
        ),
        task_evaluation_version=evaluation.version,
        parent_lineage="connect4-L2",
        restarted_from_scratch=True,
    ))

    hypothesis = f"""\
CONTEXT: The task is to win Connect 4, scored by {evaluation.ref}: 168 games
against perfect-play opponents searching 1 to 6 moves deep, averaged. The
policy is a convolutional network of about 183,000 weights that rates each of
the seven columns. The second line of policies (exp-024) was the first to look
ahead while training — 32 states per move, choosing where to look by its own
estimates — and reached 0.068 of a possible 1.0 in 4,800 games, against the
first line's 0.116 in roughly 10,500. Its cards showed the value function
learning correctly: positions a perfect player proves are won were rated
+0.93, and proven-lost positions -0.83, up from +0.27 at the start. What
failed was move preference: it missed about half of all forced blocks, which
a lookahead of that size always sees. In the same run the largest estimate
drifted from 1.36 to 1.93, outside the range any position can truly be worth,
even though the training targets were clamped.

HYPOTHESIS: if value estimates are bounded by the architecture rather than by
clamping the targets, then move preference will stop degrading and the policy
will exceed the first line's 0.116, because a winning move's target is exactly
1 and no other move's estimate can exceed it — whereas with an unbounded
output a merely good move can be rated above a winning one, which is precisely
the failure the blocking rate showed. Raising the lookahead budget should
compound this, since a larger budget finds consequences a smaller one misses.
Changes from the second line, each from -> to: network output unbounded ->
squashed into the range the rewards span; lookahead budget while training
32 -> {BUDGET} states per move; lookahead while being scored 0 -> {EVAL_BUDGET}
states per move, recorded alongside the unassisted score rather than replacing
it; cycles 12 -> {CYCLES}. Unchanged: how the budget is spent, the training
target, the data rules, and the task's definition of success.

PREDICTION: the largest value estimate stays at or below 1 for the whole run,
by construction; the rate at which the policy blocks a forced loss rises above
0.80, against 0.51 in the second line; and the unassisted score exceeds 0.116
within {CYCLES * GAMES_PER_CYCLE:,} games. If the bound holds and the blocking
rate still does not rise, then the fault is not the value range but the
attempt to make one set of outputs serve as both the judgement of a position
and the ranking of moves, and the next step is to separate them into two
heads.

{DECISION_CRITERIA}"""

    store.tasks.register(task.definition())
    store.experiments.register(
        experiment_id=EXPERIMENT_ID,
        hypothesis=hypothesis,
        changes=(f"new lineage {LINEAGE_ID} from random weights under rubric "
                 f"{rubric.rubric_id}; each from->to: bounded_output off->on "
                 f"(tanh), search_budget 32->{BUDGET}, eval lookahead "
                 f"0->{EVAL_BUDGET} states (recorded, not deciding), cycles "
                 f"12->{CYCLES}; health conditions now applied at promotion"),
        conditions={
            "task_ref": task.ref,
            "benchmark_ref": task.benchmark.ref,
            "config": {"lineage_id": LINEAGE_ID, "rubric_id": rubric.rubric_id,
                       "task_evaluation": evaluation.ref,
                       "games_per_cycle": GAMES_PER_CYCLE, "cycles": CYCLES,
                       "search_budget": BUDGET, "eval_budget": EVAL_BUDGET,
                       "eval_budget_decides": False, "spend_rule": "puct",
                       "bounded_output": True, "gate_rule": "strict",
                       "health_gate": True, "renderer": "connect4_v2"},
            "seed": 25,
        },
        prior_art=["exp-024-lineage2-lookahead32", "exp-023-budget-vs-depth",
                   "exp-022-B-search-targets-seed27", "exp-020-lookahead-inference"],
        tags=["lineage-3", "bounded-values", "train-lookahead", "budget-128",
              "eval-lookahead", "gate-v3", "health-gate"],
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
        seed=25,
        eval_budget=EVAL_BUDGET,
        eval_budget_decides=False,
        resume=True,
    ))

    for record in records:
        record.render()

    from harness.learning_arc import render_plot, render_text, save_summary, summarize

    summary = summarize(root, LINEAGE_ID)
    print()
    print(render_text(summary))
    print(f"\narc data:  {save_summary(root, LINEAGE_ID)}")
    print(f"arc plot:  {render_plot(summary)}")


if __name__ == "__main__":
    main()
