"""Ladder benchmark: frozen procedure, sane ordering (a tactical fixture
scores above random-ish play), frontier reporting, and task-v2 wiring."""

import numpy as np

from harness.scripted_opponents import OneStepWinOpponent, RandomOpponent
from harness.tasks.connect4_ladder import (
    Connect4LadderTask,
    SolverLadderBenchmark,
    _fixed_openings,
)


def opponent_as_q(opponent, n=7):
    def q(obs):
        legal = [c for c in range(n) if obs[0, 0, c] == 0 and obs[1, 0, c] == 0]
        values = np.full(n, -1.0, dtype=np.float32)
        values[opponent.act(obs, legal)] = 1.0
        return values

    return q


def test_openings_are_fixed_and_distinct():
    a, b = _fixed_openings(), _fixed_openings()
    assert a == b  # frozen forever
    assert len(a) == 14 and len(set(a)) == 14
    assert all(len(o) in (1, 2) for o in a)


def test_benchmark_is_deterministic_for_deterministic_policy():
    bench = SolverLadderBenchmark(min_rung=1, max_rung=2)
    r1 = bench.score(opponent_as_q(OneStepWinOpponent(seed=3)))
    r2 = bench.score(opponent_as_q(OneStepWinOpponent(seed=3)))
    assert r1.score == r2.score
    assert r1.detail["per_rung"] == r2.detail["per_rung"]


def test_result_shape_and_stamping():
    bench = SolverLadderBenchmark(min_rung=1, max_rung=2)
    result = bench.score(opponent_as_q(RandomOpponent(seed=1)))
    assert result.benchmark_ref == "connect4-benchmark@v1"
    assert set(result.detail["per_rung"].keys()) == {"1", "2"}
    assert result.detail["games_per_rung"] == 28
    assert result.detail["games_total"] == 56
    assert 0.0 <= result.score <= 1.0


def test_frontier_depth_reporting():
    bench = SolverLadderBenchmark(min_rung=1, max_rung=2)
    # A weak policy's frontier should be the first rung.
    weak = bench.score(opponent_as_q(RandomOpponent(seed=2)))
    assert weak.detail["frontier_depth"] == 1
    assert weak.score < 0.9


def test_tactical_policy_outscores_weak_policy():
    """Ordering sanity: win-if-possible/block-if-needed must outscore
    random play on the same ladder."""
    bench = SolverLadderBenchmark(min_rung=1, max_rung=2)
    strong = bench.score(opponent_as_q(OneStepWinOpponent(seed=5)))
    weak = bench.score(opponent_as_q(RandomOpponent(seed=5)))
    assert strong.score > weak.score


def test_task_v2_definition():
    task = Connect4LadderTask()
    d = task.definition()
    assert d["task_id"] == "connect4"
    assert d["version"] == 2
    assert d["benchmark_ref"] == "connect4-benchmark@v1"
    assert task.mastery_threshold == 0.90
