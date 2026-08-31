"""Evaluator: recovers known strength orderings from fixture opponents, and
stamps every result with (policy, task@v, opponent)."""

import numpy as np

from harness.scripted_opponents import OneStepWinOpponent, RandomOpponent
from harness.policy_evaluator import Evaluator
from harness.tasks.connect4 import Connect4Task


def opponent_as_q(opponent, n=7):
    """Adapt an Opponent to the QFunction interface: chosen move gets the
    max value."""

    def q(obs):
        legal = [c for c in range(n) if obs[0, 0, c] == 0 and obs[1, 0, c] == 0]
        values = np.full(n, -1.0, dtype=np.float32)
        values[opponent.act(obs, legal)] = 1.0
        return values

    return q


def test_one_step_win_beats_random():
    """Known ordering: win-if-possible/block-if-needed must clearly beat
    uniform random over enough games."""
    task = Connect4Task()
    ev = Evaluator(task, seed=123)
    strong = opponent_as_q(OneStepWinOpponent(seed=1))
    result = ev.vs_opponent(strong, RandomOpponent(seed=2), games=100, policy_label="one-step")
    assert result.score > 0.85, f"expected dominance, got {result.score}"


def test_result_stamping():
    task = Connect4Task()
    ev = Evaluator(task, seed=0)
    q = opponent_as_q(RandomOpponent(seed=3))
    result = ev.vs_opponent(q, RandomOpponent(seed=4), games=10, policy_label="pol-x")
    assert result.task_ref == "connect4@v1"
    assert result.policy_label == "pol-x"
    assert result.opponent_label == "random"
    assert result.wins + result.losses + result.draws == result.games == 10


def test_head_to_head_symmetry():
    """Two copies of the same random-ish policy should split roughly evenly,
    and sides must alternate (no first-move bias in the harness)."""
    task = Connect4Task()
    ev = Evaluator(task, seed=9)
    q_a = opponent_as_q(RandomOpponent(seed=10))
    q_b = opponent_as_q(RandomOpponent(seed=11))
    result = ev.head_to_head(q_a, q_b, games=100, label_a="a", label_b="b", temperature=1.0)
    assert 0.3 < result.score < 0.7, f"mirror-match score should be near 0.5, got {result.score}"


def test_benchmark_is_frozen():
    """Same policy, same benchmark -> identical score. The yardstick does not
    move."""
    task = Connect4Task(benchmark_games=50)
    # Fresh identically-seeded policy instances: the POLICY must be identical
    # across calls for this to test the benchmark's own determinism.
    r1 = task.benchmark.score(opponent_as_q(OneStepWinOpponent(seed=5)))
    r2 = task.benchmark.score(opponent_as_q(OneStepWinOpponent(seed=5)))
    assert r1.score == r2.score
    assert r1.benchmark_ref == "connect4-benchmark@v0"
