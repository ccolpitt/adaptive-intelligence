"""exp-002: does more self-play training raise absolute (ladder) strength?

Run:  python3 experiments/exp_002_more_training.py
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

    # -- context measurement (read-only): where does the incumbent stand? --
    incumbent_id = store.archive.current_champion(task.task_id)
    import numpy as np
    import torch

    module, _ = store.archive.load_policy(incumbent_id)

    def q(obs):
        with torch.no_grad():
            return module(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).numpy()

    incumbent = task.benchmark.score(q)
    frontier = incumbent.detail["frontier_depth"]
    print(f"incumbent {incumbent_id}: ladder {incumbent.score:.3f}, frontier depth {frontier}")
    print(f"per rung: {incumbent.detail['per_rung']}")

    hypothesis = f"""\
CONTEXT: We improve the policy by self-play: a challenger learns by playing a
frozen copy of the current champion (epsilon-greedy DQN, uniform replay,
negamax targets). The board and player are represented AlphaZero-style as two
6x7 planes - plane 0 the mover's pieces, plane 1 the opponent's - flipped to
the mover's perspective every move, so there is no whose-turn input. The
policy is a small conv net (2 conv layers of 32 filters, one 128-unit hidden
layer, 7 Q-value outputs). The current champion ({incumbent_id}) has trained
for 1500 episodes and scores {incumbent.score:.3f} on the solver ladder
(depths 1-6; 1.0 = beats every depth nearly always; its frontier - the
shallowest solver it cannot beat 90% of the time - is depth {frontier}).

HYPOTHESIS: 1500 additional self-play episodes (5 iterations of 300), with no
other change, will raise absolute strength. The champion is early in training,
so more of the same training should still convert directly into strength.

PREDICTION: If true, we expect the final champion's ladder score to be at
least 0.09 above the incumbent's ({incumbent.score:.3f} -> {incumbent.score + 0.09:.3f}),
because early-stage DQN gains scale with episodes and 0.09 is the smallest
gain our 168-game ladder evaluation can distinguish from noise.

DECISION CRITERIA (recorded so a false positive or false negative can be
traced back before it sends us down a bad path): a challenger is promoted only
if its head-to-head score against the champion clears a 95% Wilson lower bound
above 0.5 over 200 games (about a 5% chance of promoting a policy that is no
better), and promotion is blocked if its ladder score is significantly below
the champion's - the gate that catches beats-the-champion-but-weaker-in-
absolute-terms cycling. Gains smaller than 0.09 ladder / 0.06 head-to-head are
invisible at these game counts and will be recorded INCONCLUSIVE, never
refuted; the follow-up for an inconclusive result is more evaluation games,
not abandoning the direction."""

    cfg = LoopConfig(
        experiment_id="exp-002-more-self-play",
        hypothesis=hypothesis,
        changes="training volume only: +1500 self-play episodes (5 x 300); "
        "no architecture, hyperparameter, or representation changes",
        prior_art=["exp-001-close-the-loop"],
        tags=["training-volume", "baseline", "ladder-v1"],
        seed=1,
        iterations=5,
        episodes_per_iteration=300,
        gate_games=200,
        trainer=TrainerConfig(seed=1),
    )
    summary = run_loop(task, store, cfg)
    print(summary)


if __name__ == "__main__":
    main()
