"""Monte-Carlo target labeling (mc_targets): the human's explicit
back-propagation-of-credit scheme — every stored move labeled with the final
outcome discounted by gamma^(plies from end), done=True."""

import numpy as np

from harness.agents.dqn import DQNTrainer, TrainerConfig
from harness.scripted_opponents import RandomOpponent
from harness.tasks.connect4 import Connect4Env


def _play_games(cfg, n=8):
    trainer = DQNTrainer(6, 7, 7, cfg)
    opp = RandomOpponent(seed=1)
    for _ in range(n):
        trainer.play_episode(Connect4Env(), opponent_q=None, opponent_act=opp.act)
    return trainer


def test_mc_labels_all_transitions_terminal_and_discounted():
    trainer = _play_games(TrainerConfig(seed=0, batch_size=8, mc_targets=True,
                                        store_opponent_transitions=True))
    transitions = list(trainer.buffer.buffer)
    assert transitions
    # every stored transition is terminal-labeled (target = reward exactly)
    assert all(t[5] for t in transitions)
    rewards = np.array([t[2] for t in transitions])
    # labels live in [-1, 1] (no shaping in this config)
    assert np.all(np.abs(rewards) <= 1.0 + 1e-9)
    # decisive games produce both a +1 (winner's final move) and a -1
    # (loser's final move), and intermediate discounted magnitudes
    assert np.isclose(rewards.max(), 1.0)  # winner's final move: gamma^0
    # loser's final move is exactly one ply before the end: -1 * gamma^1
    assert np.isclose(rewards.min(), -0.99)
    mids = rewards[(np.abs(rewards) > 0) & (np.abs(rewards) < 1.0)]
    assert mids.size > 0, "expected discounted intermediate labels"
    # discount structure: every nonzero |label| is gamma^d for integer d
    gamma = 0.99
    for r in rewards[np.abs(rewards) > 0]:
        d = np.log(abs(r)) / np.log(gamma)
        assert abs(d - round(d)) < 1e-6


def test_mc_alternating_signs_within_one_game():
    """In a decisive game, consecutive plies belong to opposite movers, so
    consecutive MC labels alternate in sign."""
    cfg = TrainerConfig(seed=3, batch_size=8, mc_targets=True,
                        store_opponent_transitions=True)
    trainer = DQNTrainer(6, 7, 7, cfg)
    opp = RandomOpponent(seed=4)
    env = Connect4Env()
    trainer.play_episode(env, opponent_q=None, opponent_act=opp.act)
    if env.winner in (None, 0):
        return  # draw — nothing to assert
    rewards = [t[2] for t in trainer.buffer.buffer]
    signs = [np.sign(r) for r in rewards if r != 0]
    assert all(a != b for a, b in zip(signs, signs[1:]))


def test_mc_off_preserves_default_behavior():
    trainer = _play_games(TrainerConfig(seed=0, batch_size=8))
    rewards = {t[2] for t in trainer.buffer.buffer}
    # default TD mode: only pure win/lose/none rewards in the buffer
    assert rewards <= {-1.0, 0.0, 1.0}
