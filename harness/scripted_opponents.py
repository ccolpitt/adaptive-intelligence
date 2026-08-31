"""Scripted opponents: baselines for evaluation fixtures and the learnability
canary. These are Opponents (yardstick ingredients), not Policies."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .interfaces import Opponent
from .tasks.connect4 import Connect4Env


class RandomOpponent(Opponent):
    """Uniform over legal moves, seeded."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, legal: Sequence[int]) -> int:
        return int(self.rng.choice(list(legal)))


class FixedColumnOpponent(Opponent):
    """Always plays the lowest-index legal column. Trivially exploitable —
    the learnability-canary opponent."""

    name = "fixed-column"

    def act(self, obs: np.ndarray, legal: Sequence[int]) -> int:
        return int(min(legal))


class OneStepWinOpponent(Opponent):
    """Plays an immediate winning move when one exists, blocks the opponent's
    immediate win when one exists, otherwise random. Strictly stronger than
    RandomOpponent — used to test that the evaluator recovers known strength
    orderings."""

    name = "one-step-win"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, legal: Sequence[int]) -> int:
        # Reconstruct a board from the canonical obs: channel 0 = my pieces.
        env = Connect4Env()
        env.board = (obs[0] - obs[1]).astype(np.int8)  # me = +1, them = -1
        env.current_player = 1
        wins = env.winning_moves(1)
        if wins:
            return wins[0]
        blocks = env.winning_moves(-1)
        if blocks:
            return blocks[0]
        return int(self.rng.choice(list(legal)))
