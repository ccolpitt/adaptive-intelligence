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

Run:      caffeinate -is python3 -u experiments/exp_027_two_heads.py
Resume:   the same command continues from the last completed cycle.
"""

from _common import DECISION_CRITERIA, Store

from harness.cycle_loop import CycleLoopConfig, run_cycles
from harness.lineage import Lineage, LineageStore, RubricStore, TrainingRubric
from harness.task_repository import TaskRepository, seed_repository
from harness.tasks.connect4_ladder import Connect4LadderTask

LINEAGE_ID = "connect4-L5"
EXPERIMENT_ID = "exp-027-two-heads"
BUDGET = 200          # positions the search may examine per training move
EVAL_BUDGET = 200     # and when the policy is scored with thinking allowed
GAMES_PER_CYCLE = 200
CYCLES = 60


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
against perfect-play opponents that search 1 to 6 moves deep, averaged, with
the policy choosing its moves in a single look and no search. Four earlier
lineages used one set of seven outputs both to rank the columns and, through
their magnitudes, to judge the position. All four showed the same split
result. The most recent (16,000 games, search of 128 positions per training
move) improved steadily when measured with search allowed, from 0.226 to
0.551, which is the largest number the project has produced; measured in a
single look the same policies went from 0.018 to only 0.101, and the rate at
which they blocked a forced loss stayed between 0.31 and 0.58 throughout,
even though a search of that size always sees a forced loss one move away.

HYPOTHESIS: if the policy is given two separate outputs — a probability per
column saying which moves are worth considering, and one bounded number
judging the position — and each is trained on what the search concluded,
the move head toward where the search spent its attention and the value head
toward how the game actually ended, then the policy scored in a single look
will improve far more than it did with one shared output, because ranking the
columns no longer depends on the magnitudes used to judge positions, and
because the move head's target is the search's own distribution of attention
rather than a number derived from the network's estimates. Changes from the
fourth lineage, each from -> to: one output set -> two heads; what the network
is trained toward, from "the value the search reached for the move played" ->
"the attention the search gave each move, and the actual result of the game";
how the move is chosen while training, from "the best value the search found"
-> "sampled from the search's attention for the first twelve moves, then the
most attended"; trunk 2 convolutional layers of 32 channels -> 3 of 64;
training games per cycle 400 -> {GAMES_PER_CYCLE}; search budget 128 -> {BUDGET}
distinct positions; hand-written opponents and hand-written reward shaping ->
none at all, so the recipe no longer contains any knowledge of what a good
Connect 4 move is.

PREDICTION: within {CYCLES * GAMES_PER_CYCLE:,} training games the policy
scored in a single look exceeds 0.20, which is roughly double the best any
previous lineage reached that way (0.113) and above the best single-look score
in the project's history (0.116 from the first lineage, after 10,500 games).
The rate at which it blocks a forced loss exceeds 0.80, against a ceiling of
0.58 across four lineages. Value estimates stay inside the reward range
throughout, by construction. If the single-look score again fails to move
while the score with search allowed climbs, then the obstacle is not the
shared output and the next suspect is the amount of search per move: a budget
of {BUDGET} positions over seven columns gives roughly thirty visits each, and
the distribution of attention may simply be too coarse a target to learn from.

{DECISION_CRITERIA}"""

    store.tasks.register(task.definition())
    store.experiments.register(
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
        prior_art=["exp-025-bounded-values", "exp-024-lineage2-lookahead32",
                   "exp-023-budget-vs-depth", "exp-022-B-search-targets-seed27"],
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
