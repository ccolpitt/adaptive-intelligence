"""exp-022: search during TRAINING only; evaluation stays bare inference.

Plain language: exp-020 showed the champion network plays 4x better when it
is allowed to look 2-3 moves ahead at move time. The human's proposal: let
the policy look ahead while PRACTICING, but judge it on instant, one-glance
play — the way a chess student calculates during study and plays on
instinct at the board. Two arms, so we learn which part matters:

  Arm A (seeds 24-26): the learner picks its self-play moves with 2-ply
         lookahead. It learns from stronger games. Targets unchanged (TD).
  Arm B (seeds 27-29): as A, AND each stored move's training target becomes
         the search-backed value of that move — the network is trained to
         reproduce what the search concluded (distillation / expert
         iteration, AlphaZero's policy-improvement step).

The benchmark (168 ladder games) and head-to-head are unchanged bare
inference, so every score is comparable with all 20 prior experiments.

Run:  caffeinate -is python3 experiments/exp_022_train_search.py
"""

from _common import DECISION_CRITERIA, Store, context_paragraph, measure_incumbent

from harness.agents.dqn import TrainerConfig
from harness.improvement_loop import LoopConfig, run_loop
from harness.tasks.connect4_ladder import Connect4LadderTask

SEARCH_DEPTH = 2
# exp-022 = exp-021 re-run under the corrected plateau detector (warm-up 12
# iterations, patience 8 — exp-021-seed24 exposed that the old detector
# killed cold-start runs before their breakout). Arm 0 re-measures the recipe
# baseline under the SAME detector so the comparison is clean: the old
# 3-seed baseline (0.054/0.060/0.116) was itself truncated in two of three.
ARMS = {
    "0-recipe-baseline": {"seeds": (30, 31, 32), "search_depth": 0, "search_targets": False},
    "A-search-moves": {"seeds": (24, 25, 26), "search_depth": SEARCH_DEPTH, "search_targets": False},
    "B-search-targets": {"seeds": (27, 28, 29), "search_depth": SEARCH_DEPTH, "search_targets": True},
}
BASELINE_3SEED = (0.054, 0.060, 0.116)  # recipe under the OLD detector (exp-016/017/012)


def main() -> None:
    store = Store("store")
    task = Connect4LadderTask()

    for arm, spec in ARMS.items():
        for seed in spec["seeds"]:
            incumbent_id, incumbent = measure_incumbent(store, task)
            targets_txt = (
                "no search at all: this arm re-measures the recipe baseline under the "
                "corrected plateau detector (warm-up 12, patience 8) so all arms share one "
                "stopping rule"
                if spec["search_depth"] == 0 else
                "AND search_targets True: each stored move's training target is the "
                "search-backed value of that move (supervised, done=True) instead of "
                "the one-step TD estimate; tactical shaping is NOT added on top (the "
                "search already prices the tactics)"
                if spec["search_targets"] else
                "targets unchanged (one-step TD + tactical shaping, as in the recipe)"
            )
            hypothesis = f"""\
{context_paragraph(incumbent_id, incumbent)}
Evidence chain: exp-020 — the unchanged champion scores 0.071 bare, 0.217
with 2-ply lookahead, 0.310 with 3-ply: its knowledge is good enough to
SEARCH over but not to ACT on in one forward pass. Every training lever that
left inference untouched moved the ladder by hundredths (exp-002..019).
Recipe 3-seed bare baseline: {BASELINE_3SEED} (mean 0.077, spread 0.062).

HYPOTHESIS (human-directed split: search in training, bare at evaluation):
letting the learner look ahead while practicing improves the network's
one-glance play, because (A) it trains on stronger, less blunder-filled
games, and (B — this arm's extra claim if search_targets) regressing Q onto
search-backed values distills the lookahead into the weights (expert
iteration). Exact changes vs the recipe: train_search_depth 0 -> {spec['search_depth']};
{targets_txt}. Benchmark and head-to-head: bare inference, unchanged.
This run: arm {arm}, seed {seed}.

PREDICTION (scored per arm over its three seeds, bare-inference ladder):
Arm A mean >= 0.14 (baseline mean + spread) if stronger self-play data alone
helps; Arm B mean >= 0.20 — distillation should move the bare network a
good way toward exp-020's 2-ply search score (0.217), and the policy cards'
take-the-win / block-the-loss rates should approach 1.0 since those are
exactly what a 2-ply search prices. If B <= A, the targets add nothing over
the data; if both <= baseline, search knowledge does not transfer into
one-glance weights at this scale and inference-time search becomes the
design question again. Training cost ~50x more forward passes per self-play
move — recorded, acceptable.

{DECISION_CRITERIA}"""

            run_loop(
                task,
                store,
                LoopConfig(
                    experiment_id=f"exp-022-{arm}-seed{seed}",
                    hypothesis=hypothesis,
                    changes=(f"vs recipe: train_search_depth 0->{spec['search_depth']}"
                             if spec["search_depth"] else
                             "recipe baseline re-measured under the corrected plateau detector")
                    + ("; search_targets False->True (shaping not applied on top)"
                       if spec["search_targets"] else "")
                    + f"; arm {arm}, seed {seed}; evaluation bare (unchanged); "
                    "plateau warm-up 12 / patience 8",
                    prior_art=[
                        "exp-021-A-search-moves-seed24",
                        "exp-020-lookahead-inference",
                        "exp-019-mc-targets-seed21",
                        "exp-017-recipe-seed-replication",
                        "exp-012-signal-package",
                    ],
                    tags=["train-search", arm, "multi-seed", "ladder-v1", "gate-v2"],
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
                        train_search_depth=spec["search_depth"],
                        search_targets=spec["search_targets"],
                    ),
                ),
            )


if __name__ == "__main__":
    main()
