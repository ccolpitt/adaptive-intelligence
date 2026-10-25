"""exp-027: split the policy into two heads — one proposes moves, one judges.

Three lineages of a single-headed policy all showed the same pattern: the
judgement of positions improved steadily while the choice of move did not.
In the third lineage, measured with 128 positions of lookahead, the score
climbed from 0.226 to 0.551 over 16,000 games — real competence — while the
same policies scoring themselves with no lookahead went only 0.018 to 0.101,
and the rate at which they blocked a forced loss never rose above 0.58. The
competence was being built and was not reaching the instant answer.

The suspected reason is that one set of seven numbers was being asked to do
two different jobs: rank the columns, and by its magnitudes also say how good
the position is. An error in the magnitude then corrupts the ranking.

This run separates them. A move head outputs a probability per column; a
value head outputs one number for the position, bounded to the range the
rewards span. Neither is the value of a particular move — that comes from the
search. The search, guided by both heads, is stronger than either alone, so
its conclusions become the training targets: the move head learns where the
search spent its attention, and the value head learns how the game actually
ended.

Run:      caffeinate -is python3 -u experiments/exp_028_two_heads.py
Resume:   the same command continues from the last completed cycle.
"""

from _common import DECISION_CRITERIA, Store, register_once

from harness.cycle_loop import CycleLoopConfig, run_cycles
from harness.lineage import Lineage, LineageStore, RubricStore, TrainingRubric
from harness.task_repository import TaskRepository, seed_repository
from harness.tasks.connect4_ladder import Connect4LadderTask

LINEAGE_ID = "connect4-L5"
EXPERIMENT_ID = "exp-029-two-heads-extended"
BUDGET = 200          # positions the search may examine per training move
EVAL_BUDGET = 200     # and when the policy is scored with thinking allowed
GAMES_PER_CYCLE = 200
CYCLES = 160


def rubric_r005() -> TrainingRubric:
    return TrainingRubric(
        rubric_id=f"r005-two-heads-{BUDGET}",
        name=f"Two heads, search of {BUDGET} positions per move",
        description=(
            f"The policy has two outputs. A move head gives a probability for "
            f"each column, saying which moves are worth considering. A value "
            f"head gives one number for the position, from the point of view of "
            f"whoever moves next, squashed into the range the rewards span so "
            f"it cannot drift outside it. Neither is the value of a particular "
            f"move: that comes from the search, which tries moves and backs up "
            f"what it finds. Before each training move the search examines up "
            f"to {BUDGET} distinct positions, using the move head to decide "
            f"where to look and the value head to judge what it finds. The "
            f"move actually played is sampled from where the search spent its "
            f"attention for the first twelve moves of a game and taken "
            f"greedily afterwards. The move head is then trained toward that "
            f"same distribution of attention, and the value head toward how "
            f"the game actually finished. Scoring the policy uses the move "
            f"head alone, with no search."
        ),
        learner="policy-value",
        search_budget=BUDGET,
        spend_rule="puct",
        target_kind="search-visits",
        bounded_targets=True,
        hyperparameters={"lr": 0.001, "weight_decay": 0.0001, "gamma": 0.99,
                         "batch_size": 128, "train_steps_per_game": 4,
                         "buffer_capacity": 100000, "channels": 64,
                         "hidden": 128, "conv_layers": 3, "c_puct": 1.4,
                         "dirichlet_noise": 0.25, "sampling_moves": 12,
                         "value_loss_weight": 1.0, "mirror_augmentation": True},
        data_rules={"self_play": True,
                    "note": "the policy plays itself; both sides' positions are "
                            "recorded, since in self-play both sides are the "
                            "same policy and both searches are informative. No "
                            "hand-written opponent and no hand-written reward "
                            "shaping are used, so nothing in this recipe "
                            "depends on knowing what a good Connect 4 move is."},
        notes=(
            "The structure AlphaZero used (Silver et al., 2017). Adopted here "
            "because three lineages of a single-headed policy separated value "
            "judgement from move choice in measurement but could not join them "
            "in training."
        ),
    )


def main() -> None:
    store = Store("store")
    root = store.root
    task = Connect4LadderTask()

    tasks = TaskRepository(root / "tasks")
    seed_repository(tasks)
    evaluation = tasks.get_evaluation("connect4")
    rubric = RubricStore(root / "rubrics").publish(rubric_r005())

    LineageStore(root / "lineages").create(Lineage(
        lineage_id=LINEAGE_ID,
        task_id="connect4",
        rubric_id=rubric.rubric_id,
        description=(
            "Fifth line of Connect 4 policies, and the first with two heads: "
            "one proposing moves, one judging positions. Started fresh because "
            "the network has a different shape, so continuing from an earlier "
            "line's weights is not possible."
        ),
        task_evaluation_version=evaluation.version,
        parent_lineage="connect4-L4",
        restarted_from_scratch=True,
    ))

    hypothesis = f"""\
CONTEXT: The task is to win Connect 4, scored by {evaluation.ref}: 168 games
against perfect-play opponents searching 1 to 6 moves deep, averaged, with the
policy choosing in a single look and no search. The two-headed policy
(exp-028) took this score from 0.027 to 0.622 over 12,000 self-play games in
the lineage {LINEAGE_ID}, against a best of 0.113 for every earlier lineage,
and its per-rung line at the end was 0.71 / 0.66 / 0.70 / 0.57 / 0.54 / 0.55 —
winning against every depth from 1 to 6. The trend was still rising when the
game budget ran out, so the run ended for want of compute rather than for want
of progress. Mastery requires 0.90 at every rung.

HYPOTHESIS: if the same policy keeps training under the same recipe, with no
change of any kind, then its score will continue to rise rather than level
off, because nothing in the previous run indicated a limit being reached: the
score was climbing, value estimates were stable inside their bounds, no
expectation check fired across all sixty cycles, and the tactical rates were
still improving. The only change is the number of cycles, from 60 to
{CYCLES}, continuing the SAME line of policies rather than starting a new one
— the first time this project has extended a lineage instead of restarting.

PREDICTION: the score exceeds 0.75 within {CYCLES * GAMES_PER_CYCLE:,}
cumulative games. Taking an available win and blocking a forced loss both
exceed 0.90, since both were rising and both are a precondition for beating
deeper opponents. If instead the score settles between 0.60 and 0.70 while
the tactical rates stay below 0.90, the limit is not compute and the next
move is to attack the tactical gap directly. A third outcome is worth naming
in advance: if the score passes roughly 0.75, the ladder itself becomes the
constraint, because its hardest opponent searches only six moves — and the
response is a new evaluation version with deeper rungs, recorded alongside
the current one so no earlier score loses its meaning.

{DECISION_CRITERIA}"""

    store.tasks.register(task.definition())
    register_once(
        store,
        experiment_id=EXPERIMENT_ID,
        hypothesis=hypothesis,
        changes=(f"new lineage {LINEAGE_ID} from random weights under rubric "
                 f"{rubric.rubric_id}; each from->to: learner single-output "
                 f"'dqn'->two-headed 'policy-value'; target 'search-value'->"
                 f"'search-visits' plus actual game outcome; move choice "
                 f"'best search value'->'sampled from search attention for 12 "
                 f"moves then most attended'; trunk 2x32->3x64 convolutions; "
                 f"search budget 128->{BUDGET} distinct positions; "
                 f"games/cycle 400->{GAMES_PER_CYCLE}; cycles->{CYCLES}; "
                 f"scripted opponents and reward shaping->none"),
        conditions={
            "task_ref": task.ref,
            "benchmark_ref": task.benchmark.ref,
            "config": {"lineage_id": LINEAGE_ID, "rubric_id": rubric.rubric_id,
                       "task_evaluation": evaluation.ref,
                       "games_per_cycle": GAMES_PER_CYCLE, "cycles": CYCLES,
                       "search_budget": BUDGET, "eval_budget": EVAL_BUDGET,
                       "eval_budget_decides": False, "gate_rule": "strict",
                       "health_gate": True, "renderer": "connect4_v2",
                       "self_play_only": True},
            "seed": 27,
        },
        prior_art=["exp-027-two-heads", "exp-025-bounded-values",
                   "exp-024-lineage2-lookahead32", "exp-023-budget-vs-depth"],
        tags=["lineage-5", "two-heads", "policy-value", "search-visits",
              "self-play-only", "gate-v3"],
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
        seed=27,
        eval_budget=EVAL_BUDGET,
        eval_budget_decides=False,
        diverse_opponents=False,   # self-play only: no borrowed knowledge
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
