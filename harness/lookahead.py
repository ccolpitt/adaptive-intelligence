"""Lookahead at move time: wrap a learned Q-function in a shallow search.

Today a policy answers in ONE forward pass — it looks at the board and
emits seven numbers; whichever is largest is the move. It never tries a move
and looks at what happens. Every ladder opponent does exactly that (negamax
to depth 1-6). This module gives the learned policy the same ability:

    value(move) = reward(move) - gamma * max_a' value'(reply a')   ...k plies deep,
    leaf value  = max_a Q_net(position, a)

i.e. the SAME Bellman recursion the trainer uses to build its targets, but
computed live at decision time with real game rules instead of trusted from
the memorized Q. Depth 1 recovers "take any immediate win, otherwise trust
Q"; depth 2 additionally sees "if I play here, can the opponent win at once?"
— the two tactical blunders the cards keep flagging — without any training.

The wrapper is itself a QFunction (obs -> 7 values), so it drops into the
benchmark, the evaluator, the cards, and interactive play unchanged.
Connect-N specific only in `_env_from_obs` (board reconstruction).
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from .interfaces import QFunction
from .tasks.connect4 import Connect4Env

ILLEGAL = -1e9


def _env_from_obs(obs: np.ndarray) -> Connect4Env:
    """Canonical obs: channel 0 = mover's pieces, channel 1 = opponent's."""
    env = Connect4Env(rows=obs.shape[1], cols=obs.shape[2])
    env.board = (obs[0] - obs[1]).astype(np.int8)
    env.current_player = 1  # mover is always +1 in canonical obs
    env.done = False
    env.winner = None
    return env


class LookaheadQ:
    """Negamax over real game rules, ``depth`` plies deep, with the learned
    Q-function valuing the leaves. Returns search-backed values per column."""

    def __init__(self, base_q: QFunction, depth: int = 2, gamma: float = 0.99):
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.base_q = base_q
        self.depth = depth
        self.gamma = gamma
        self.leaf_evals = 0  # diagnostics: how much the search cost

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        env = _env_from_obs(obs)
        out = np.full(env.cols, ILLEGAL, dtype=np.float64)
        for a in env.legal_actions():
            out[a] = self._value_of_move(env, a, self.depth)
        return out

    def _value_of_move(self, env: Connect4Env, action: int, depth: int) -> float:
        """Value of ``action`` for the mover, searching ``depth`` plies."""
        child = Connect4Env(rows=env.rows, cols=env.cols)
        child.board = env.board.copy()
        child.current_player = env.current_player
        result = child.step(action)
        if result.done:
            # win for the mover = +1 (reward); draw = DRAW_VALUE (0.0)
            return float(result.reward)
        # Opponent to move. Their best value from here, negated for us.
        return -self.gamma * self._state_value(child, depth - 1)

    def _state_value(self, env: Connect4Env, depth: int) -> float:
        """Value of ``env`` for the player to move: max over their options."""
        legal = env.legal_actions()
        if depth == 0:
            self.leaf_evals += 1
            q = np.asarray(self.base_q(env.obs()), dtype=np.float64)
            return float(np.max(q[legal]))
        return max(self._value_of_move(env, a, depth) for a in legal)


def with_lookahead(base_q: QFunction, depth: int, gamma: float = 0.99) -> QFunction:
    """Convenience: wrap and return a plain QFunction."""
    return LookaheadQ(base_q, depth=depth, gamma=gamma)
