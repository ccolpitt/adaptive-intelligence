"""Learnability canary — the "harness is not blocking learning" proof.

A small network trained inside the harness against a trivially exploitable
opponent (always plays the lowest legal column) must reach a strong win rate
within a fixed episode budget at a fixed seed. If this canary dies, the bug
is in the harness plumbing (rewards, masks, perspective flip, Bellman sign),
not in the science.

Run before every experiment batch:  pytest -m slow
"""

import pytest

from harness.agents.dqn import DQNTrainer, TrainerConfig
from harness.scripted_opponents import FixedColumnOpponent
from harness.policy_evaluator import Evaluator
from harness.tasks.connect4 import Connect4Task


@pytest.mark.slow
def test_canary_learns_to_beat_fixed_column():
    task = Connect4Task()
    cfg = TrainerConfig(
        seed=0,
        batch_size=64,
        buffer_capacity=10000,
        eps_start=0.6,
        eps_end=0.05,
        eps_decay=0.995,
        train_steps_per_episode=4,
    )
    trainer = DQNTrainer(6, 7, 7, cfg)
    opponent = FixedColumnOpponent()

    budget = 600
    threshold = 0.90
    checkpoint_every = 100
    evaluator = Evaluator(task, seed=99)

    best = 0.0
    for episode in range(budget):
        env = task.make_env()
        trainer.play_episode(env, opponent_q=None, opponent_act=opponent.act)
        if (episode + 1) % checkpoint_every == 0:
            result = evaluator.vs_opponent(
                trainer.q_function(), opponent, games=40, policy_label="canary"
            )
            best = max(best, result.score)
            if best >= threshold:
                break

    assert best >= threshold, (
        f"canary died: best score {best:.2f} < {threshold} vs fixed-column after "
        f"{budget} episodes — suspect harness plumbing (rewards / masks / "
        f"perspective flip / Bellman sign), not the agent"
    )
