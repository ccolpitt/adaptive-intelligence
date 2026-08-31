"""Evaluator — plays policies against opponents/each other and stamps every
result with (policy, task@v, benchmark@v or opponent). Component 0: this is
the load-bearing piece; everything else is uninterpretable if this is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .interfaces import Opponent, QFunction, Task


@dataclass(frozen=True)
class EvalResult:
    policy_label: str
    opponent_label: str
    task_ref: str
    games: int
    wins: int
    losses: int
    draws: int
    detail: Dict = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.wins / self.games

    @property
    def score(self) -> float:
        """Wins + half draws, in [0, 1]."""
        return (self.wins + 0.5 * self.draws) / self.games


def _masked_argmax(q: np.ndarray, legal, n: int) -> int:
    masked = np.full(n, -np.inf, dtype=np.float64)
    masked[list(legal)] = q[list(legal)]
    return int(np.argmax(masked))


def _softmax_sample(q: np.ndarray, legal, n: int, temperature: float, rng) -> int:
    legal = list(legal)
    logits = q[legal] / temperature
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(rng.choice(legal, p=probs))


class Evaluator:
    """Runs matches on a Task's environment. Two-player, perspective-flipped
    convention: whoever is to move sees the canonical observation."""

    def __init__(self, task: Task, seed: int = 0):
        self.task = task
        self.seed = seed

    def vs_opponent(
        self,
        q_function: QFunction,
        opponent: Opponent,
        games: int,
        policy_label: str,
        temperature: Optional[float] = None,
    ) -> EvalResult:
        rng = np.random.default_rng(self.seed)
        n = self.task.action_spec.n
        wins = losses = draws = 0
        for g in range(games):
            policy_is_p1 = g % 2 == 0
            env = self.task.make_env()
            env.reset()
            while not env.done:
                legal = env.legal_actions()
                obs = env.obs()
                if (env.current_player == 1) == policy_is_p1:
                    q = q_function(obs)
                    if temperature:
                        action = _softmax_sample(q, legal, n, temperature, rng)
                    else:
                        action = _masked_argmax(q, legal, n)
                else:
                    action = opponent.act(obs, legal)
                env.step(action)
            if env.winner == 0:
                draws += 1
            elif (env.winner == 1) == policy_is_p1:
                wins += 1
            else:
                losses += 1
        return EvalResult(
            policy_label=policy_label,
            opponent_label=opponent.name,
            task_ref=self.task.ref,
            games=games,
            wins=wins,
            losses=losses,
            draws=draws,
        )

    def head_to_head(
        self,
        q_a: QFunction,
        q_b: QFunction,
        games: int,
        label_a: str,
        label_b: str,
        temperature: float = 0.3,
    ) -> EvalResult:
        """A vs B, alternating sides. Sampled (temperature) rather than
        greedy: two deterministic policies would otherwise produce only two
        distinct games regardless of ``games`` — a known connect4-rl trap.
        Result is from A's perspective."""
        rng = np.random.default_rng(self.seed)
        n = self.task.action_spec.n
        wins = losses = draws = 0
        for g in range(games):
            a_is_p1 = g % 2 == 0
            env = self.task.make_env()
            env.reset()
            while not env.done:
                legal = env.legal_actions()
                obs = env.obs()
                q_fn = q_a if (env.current_player == 1) == a_is_p1 else q_b
                q = q_fn(obs)
                action = _softmax_sample(q, legal, n, temperature, rng)
                env.step(action)
            if env.winner == 0:
                draws += 1
            elif (env.winner == 1) == a_is_p1:
                wins += 1
            else:
                losses += 1
        return EvalResult(
            policy_label=label_a,
            opponent_label=label_b,
            task_ref=self.task.ref,
            games=games,
            wins=wins,
            losses=losses,
            draws=draws,
            detail={"temperature": temperature},
        )
