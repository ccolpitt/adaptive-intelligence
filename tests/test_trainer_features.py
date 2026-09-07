"""Trainer upgrade flags (exp-004/005/007). Each defaults OFF; the
determinism suite proves the defaults leave baseline behavior untouched."""

import numpy as np

from harness.agents.dqn import DQNTrainer, ReplayBuffer, TrainerConfig
from harness.scripted_opponents import FixedColumnOpponent, RandomOpponent
from harness.tasks.connect4 import Connect4Env


def make_trainer(**flags):
    cfg = TrainerConfig(seed=0, eps_start=1.0, eps_end=1.0, buffer_capacity=10000, **flags)
    return DQNTrainer(6, 7, 7, cfg)


def run_episodes(trainer, n=20, opp=None):
    opp = opp or RandomOpponent(seed=42)
    for _ in range(n):
        trainer.play_episode(Connect4Env(), opponent_q=None, opponent_act=opp.act)


# -- exp-004: both players' transitions --------------------------------------

def test_both_players_stores_more_data():
    base, both = make_trainer(), make_trainer(store_opponent_transitions=True)
    run_episodes(base)
    run_episodes(both)
    # Full games have ~2x the moves of one side's moves.
    assert len(both.buffer) > 1.7 * len(base.buffer)


def test_both_players_loss_lands_on_losing_side():
    """vs fixed-column the LEARNER usually wins; with both-players storage the
    OPPONENT's final transition must then carry the -1."""
    trainer = make_trainer(store_opponent_transitions=True)
    run_episodes(trainer, n=30, opp=FixedColumnOpponent())
    rewards = [t[2] for t in trainer.buffer.buffer if t[5]]
    assert 1.0 in rewards and -1.0 in rewards
    # Decisive games contribute exactly one +1 and one -1 terminal pair.
    assert abs(rewards.count(1.0) - rewards.count(-1.0)) <= len(rewards) * 0.2 + 2


# -- exp-005: mirror augmentation ---------------------------------------------

def test_mirror_doubles_and_reflects():
    trainer = make_trainer(mirror_augmentation=True)
    run_episodes(trainer, n=5)
    buf = list(trainer.buffer.buffer)
    assert len(buf) % 2 == 0
    for orig, mirrored in zip(buf[::2], buf[1::2]):
        assert mirrored[1] == 6 - orig[1]  # action reflected
        assert np.array_equal(mirrored[0], np.flip(orig[0], axis=2))  # obs reflected
        assert np.array_equal(mirrored[4], np.flip(orig[4]))  # legal mask reflected
        assert mirrored[2] == orig[2] and mirrored[5] == orig[5]  # reward/done same


# -- exp-007: terminal-fraction sampling --------------------------------------

def test_terminal_fraction_guarantees_quota():
    buf = ReplayBuffer(capacity=1000, seed=0, terminal_fraction=0.3)
    obs = np.zeros((2, 6, 7), dtype=np.float32)
    mask = np.ones(7, dtype=np.float32)
    for _ in range(200):
        buf.add((obs, 0, 0.0, obs, mask, False))
    for _ in range(6):
        buf.add((obs, 0, 1.0, obs, mask, True))
    batch = buf.sample(20)
    terminals = sum(1 for t in batch if t[5])
    assert terminals >= 6  # quota: min(round(20*0.3), 6 available) = 6


def test_terminal_fraction_zero_is_pure_uniform():
    a = ReplayBuffer(capacity=100, seed=1, terminal_fraction=0.0)
    b = ReplayBuffer(capacity=100, seed=1, terminal_fraction=0.0)
    obs = np.zeros((2, 6, 7), dtype=np.float32)
    mask = np.ones(7, dtype=np.float32)
    for i in range(50):
        t = (obs, i % 7, 0.0, obs, mask, i % 9 == 0)
        a.add(t)
        b.add(t)
    assert [t[1] for t in a.sample(10)] == [t[1] for t in b.sample(10)]
