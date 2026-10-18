"""Lookahead wrapper: real-rules search over a learned Q-function."""

import numpy as np

from harness.lookahead import ILLEGAL, LookaheadQ
from harness.tasks.connect4 import Connect4Env


def _board_with(moves):
    env = Connect4Env()
    env.reset()
    for m in moves:
        env.step(m)
    return env


def test_depth1_takes_immediate_win_even_when_q_is_blind():
    # X plays 0,1,2 on the bottom row; O plays elsewhere. X to move: col 3 wins.
    env = _board_with([0, 6, 1, 6, 2, 6])
    assert env.current_player == 1 and 3 in env.winning_moves(1)
    blind = lambda obs: np.zeros(7)  # the net knows nothing
    la = LookaheadQ(blind, depth=1)
    vals = la(env.obs())
    assert int(np.argmax(vals)) == 3
    assert np.isclose(vals[3], 1.0)


def test_depth2_blocks_opponent_win_even_when_q_is_blind():
    # O has 0,1,2 on the bottom row; X to move must block col 3.
    env = _board_with([6, 0, 6, 1, 5, 2])
    assert env.current_player == 1 and 3 in env.winning_moves(-1)
    blind = lambda obs: np.zeros(7)
    la = LookaheadQ(blind, depth=2)
    vals = la(env.obs())
    assert int(np.argmax(vals)) == 3
    # every non-blocking move lets O win at once: value -gamma * 1
    for c in (4, 5, 6):
        assert np.isclose(vals[c], -0.99)


def test_illegal_columns_masked_and_leaf_q_used():
    # Fill column 0 with alternating pieces (X,O,X,O,X,O): no four-in-a-row,
    # column 0 becomes illegal.
    env = Connect4Env(); env.reset()
    for _ in range(6):
        env.step(0)
    assert not env.done and 0 not in env.legal_actions()
    la = LookaheadQ(lambda obs: np.linspace(-0.5, 0.5, 7), depth=1)
    vals = la(env.obs())
    assert vals[0] == ILLEGAL
    assert la.leaf_evals > 0


def test_depth1_equals_live_bellman_target():
    """With no terminal in reach, depth-1 lookahead is exactly
    r - gamma * max_a' Q(next) — the trainer's own target, computed live."""
    env = Connect4Env(); env.reset()
    env.step(3)
    rng = np.random.default_rng(0)
    table = {}

    def q(obs):
        key = obs.tobytes()
        if key not in table:
            table[key] = rng.uniform(-0.3, 0.3, size=7)
        return table[key]

    la = LookaheadQ(q, depth=1, gamma=0.99)
    vals = la(env.obs())
    for a in env.legal_actions():
        child = Connect4Env(); child.board = env.board.copy(); child.current_player = env.current_player
        res = child.step(a)
        expected = -0.99 * np.max(q(res.next_obs)[child.legal_actions()])
        assert np.isclose(vals[a], expected)
