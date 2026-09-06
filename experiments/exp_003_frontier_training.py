"""exp-003: does training against the frontier-depth solver raise absolute
strength where pure self-play (exp-002) failed to?

Run:  python3 experiments/exp_003_frontier_training.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.agents.dqn import TrainerConfig
from harness.improvement_loop import LoopConfig, Store, run_loop
from harness.tasks.connect4_ladder import Connect4LadderTask


def main() -> None:
    store = Store("store")
    task = Connect4LadderTask()
    incumbent_id = store.archive.current_champion(task.task_id)

    hypothesis = f"""\
CONTEXT: Same policy, representation, and learner as exp-002-more-self-play
(AlphaZero-style two-plane mover-perspective board; small conv net; DQN).
exp-002 showed the problem this experiment attacks: 1500 extra self-play
episodes produced 3 head-to-head promotions but the ladder score stayed flat
(0.030 -> 0.042) and the frontier stayed at depth 1 - the policy got better
at beating copies of itself, not at beating the solver that defines absolute
strength. The incumbent is {incumbent_id}.

HYPOTHESIS: Changing the MODE OF LEARNING - playing half of all training
episodes against the solver at the champion's current frontier depth (the
shallowest depth it cannot beat 90% of the time), the other half self-play -
will convert training into absolute strength. The frontier solver is the
graded opponent just beyond current ability (the zone of proximal
development); self-play alone never exposes the policy to the tactics the
ladder actually measures.

PREDICTION: If true, we expect the final champion's ladder score to be at
least 0.09 above the incumbent's, concentrated in rung 1 (frontier depth
should move past 1), because half the training signal now comes from exactly
the opponent distribution the benchmark measures.

KNOWN CONFOUND (recorded so this cannot silently mislead later): this run
warm-starts from exp-002's final champion, so it also embodies more
cumulative training. It tests "does frontier-mixing produce further gain
where continued self-play plateaued", NOT the clean A/B "frontier-mix vs
self-play from an identical parent" - that requires forked-lineage support in
the harness (future work). If this experiment succeeds, the clean A/B is the
required follow-up before crediting the mechanism.

DECISION CRITERIA: identical statistical gates to exp-002 (h2h 95% Wilson
lower bound > 0.5 over 200 games to promote; block on significant ladder
regression; ~0.09 ladder minimum detectable effect; sub-threshold gains are
INCONCLUSIVE, never refuted)."""

    cfg = LoopConfig(
        experiment_id="exp-003-frontier-training-mix",
        hypothesis=hypothesis,
        changes="mode of learning only: 50% of training episodes vs solver at the "
        "champion's frontier depth (frontier_opponent_fraction=0.5); all else "
        "identical to exp-002",
        prior_art=["exp-002-more-self-play"],
        tags=["mode-of-learning", "curriculum", "frontier", "ladder-v1"],
        seed=2,
        iterations=5,
        episodes_per_iteration=300,
        gate_games=200,
        frontier_opponent_fraction=0.5,
        trainer=TrainerConfig(seed=2),
    )
    summary = run_loop(task, store, cfg)
    print(summary)


if __name__ == "__main__":
    main()
