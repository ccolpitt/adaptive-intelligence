"""Reproducibility: same seed -> bit-identical trajectories, buffer contents,
and network weights. Any nondeterminism here silently corrupts every A/B
comparison the whole program depends on."""

import numpy as np
import torch

from harness.agents.dqn import DQNTrainer, TrainerConfig
from harness.scripted_opponents import RandomOpponent
from harness.tasks.connect4 import Connect4Env


def run_episodes(seed, episodes=25):
    trainer = DQNTrainer(6, 7, 7, TrainerConfig(seed=seed, batch_size=32))
    opp = RandomOpponent(seed=seed + 1000)
    winners = []
    for _ in range(episodes):
        env = Connect4Env()
        stats = trainer.play_episode(env, opponent_q=None, opponent_act=opp.act)
        winners.append(stats.winner)
    return trainer, winners


def test_identical_seeds_identical_runs():
    t1, w1 = run_episodes(seed=7)
    t2, w2 = run_episodes(seed=7)

    assert w1 == w2, "episode outcomes diverged under identical seeds"
    assert len(t1.buffer) == len(t2.buffer)
    for a, b in zip(t1.buffer.buffer, t2.buffer.buffer):
        assert a[1] == b[1] and a[2] == b[2] and a[5] == b[5]
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[3], b[3])
    for (n1, p1), (n2, p2) in zip(
        t1.net.state_dict().items(), t2.net.state_dict().items()
    ):
        assert n1 == n2
        assert torch.equal(p1, p2), f"weights diverged at {n1}"
    assert t1.eps == t2.eps


def test_different_seeds_differ():
    """Sanity check that the seed actually reaches the stochastic parts."""
    _, w1 = run_episodes(seed=1)
    _, w2 = run_episodes(seed=2)
    assert w1 != w2
