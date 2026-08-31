"""Reward semantics: mover-win +1, explicit loss attribution to the loser's
final transition (the connect4-rl post-hoc-patch fix), draw handling,
zero-sum invariant."""

import numpy as np

from harness.agents.dqn import DQNTrainer, TrainerConfig
from harness.scripted_opponents import FixedColumnOpponent, RandomOpponent
from harness.tasks.connect4 import Connect4Env


def make_trainer(seed=0, eps=1.0):
    cfg = TrainerConfig(seed=seed, eps_start=eps, eps_end=eps, buffer_capacity=10000)
    return DQNTrainer(rows=6, cols=7, n_actions=7, config=cfg)


def terminal_transitions(buffer):
    return [t for t in buffer.buffer if t[5]]


def test_env_never_emits_negative_reward():
    env = Connect4Env()
    env.reset()
    rng = np.random.default_rng(1)
    rewards = []
    for _ in range(200):
        if env.done:
            env = Connect4Env()
            env.reset()
        r = env.step(int(rng.choice(env.legal_actions())))
        rewards.append(r.reward)
    assert min(rewards) >= 0.0  # loss attribution is the collector's job


def test_learner_win_gets_plus_one():
    """vs an opponent that never blocks, an eventual learner win must put a
    (+1, done) transition in the buffer."""
    trainer = make_trainer(seed=3)
    opp = FixedColumnOpponent()
    won = False
    for _ in range(50):
        env = Connect4Env()
        stats = trainer.play_episode(env, opponent_q=None, opponent_act=opp.act)
        learner_player = 1 if (trainer.episodes_done - 1) % 2 == 0 else -1
        if stats.winner == learner_player:
            won = True
            break
    assert won, "learner never won a game vs fixed-column in 50 random episodes"
    terms = terminal_transitions(trainer.buffer)
    assert any(t[2] == 1.0 and t[5] for t in terms)


def test_learner_loss_attributed_minus_one():
    """When the opponent wins, the learner's final transition must be
    rewritten to reward -1, done True — explicitly, at collection time."""
    trainer = make_trainer(seed=0)
    opp = RandomOpponent(seed=7)
    lost = False
    for _ in range(100):
        env = Connect4Env()
        stats = trainer.play_episode(env, opponent_q=None, opponent_act=opp.act)
        learner_player = 1 if (trainer.episodes_done - 1) % 2 == 0 else -1
        if stats.winner not in (None, 0, learner_player):
            lost = True
            break
    assert lost, "learner never lost vs random in 100 episodes (suspicious)"
    terms = terminal_transitions(trainer.buffer)
    assert any(t[2] == -1.0 and t[5] for t in terms)


def test_zero_sum_terminal_rewards():
    """Every decisive game contributes exactly one terminal transition to the
    learner's buffer, valued +1 (learner won) or -1 (learner lost)."""
    trainer = make_trainer(seed=11)
    opp = RandomOpponent(seed=5)
    decisive = 0
    for _ in range(30):
        env = Connect4Env()
        stats = trainer.play_episode(env, opponent_q=None, opponent_act=opp.act)
        if stats.winner in (1, -1):
            decisive += 1
    terms = terminal_transitions(trainer.buffer)
    decisive_terms = [t for t in terms if t[2] in (1.0, -1.0)]
    assert len(decisive_terms) == decisive  # draws (reward 0.0) excluded
