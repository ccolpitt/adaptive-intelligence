"""Pinning the benchmark's scale with players of known strength.

A ladder score in [0, 1] was read for most of this project's life as though
1.0 were reachable. It is not: the benchmark forces both players through fixed
openings and plays the policy in both seats, so some games begin from
objectively lost positions. These tests cover the machinery that makes a score
readable against measured reference points instead.
"""

import numpy as np
import pytest

from harness.benchmark_reference import (
    REFERENCE_PLAYERS,
    ReferenceBoard,
    ReferenceBoardStore,
    ReferenceScore,
    centre_column_policy,
    measure_reference_board,
    random_as_policy,
    render,
    solver_as_policy,
)
from harness.tasks.connect4_ladder import SolverLadderBenchmark
from harness.tasks.connect4 import Connect4Env


def a_board():
    return ReferenceBoard(
        benchmark_ref="connect4-benchmark@v1", version=1,
        scores=[
            ReferenceScore("random", "no knowledge", 0.0, {"1": 0.0}),
            ReferenceScore("solver-depth-4", "four plies", 0.488, {"1": 0.73}),
            ReferenceScore("solver-depth-10", "ten plies", 0.830, {"1": 0.88}),
        ])


# ---------------------------------------------------------------------------
# reading a score against the scale
# ---------------------------------------------------------------------------


def test_a_score_is_placed_against_the_best_measured_reference():
    placement = a_board().place(0.646)
    assert placement["best_reference"] == "solver-depth-10"
    assert placement["best_reference_score"] == 0.830
    assert placement["beats"] == ["random", "solver-depth-4"]
    assert placement["next_reference"] == "solver-depth-10"
    assert 0.77 < placement["fraction_of_best"] < 0.79


def test_the_description_refuses_the_out_of_one_reading():
    """The whole point: 0.646 is 78% of what a strong player manages here, and
    saying '65% of 1.0' is the reading that misled this project."""
    text = a_board().describe(0.646, "the champion")
    assert "0.830" in text
    assert "78%" in text
    assert "NOT" in text, "the misleading reading must be named and rejected"


def test_a_score_above_every_reference_has_no_next_step():
    placement = a_board().place(0.95)
    assert placement["next_reference"] is None
    assert placement["fraction_of_best"] > 1.0


def test_an_empty_board_does_not_pretend_to_know_the_scale():
    empty = ReferenceBoard(benchmark_ref="x@v1", version=1)
    assert empty.best() is None
    assert "no reference players" in empty.describe(0.5)


# ---------------------------------------------------------------------------
# the board is data, stored and reloaded
# ---------------------------------------------------------------------------


def test_a_board_round_trips_through_json(tmp_path):
    store = ReferenceBoardStore(tmp_path)
    path = store.save(a_board())
    assert path.suffix == ".json", "reference scores are data, never a picture"
    loaded = store.load("connect4-benchmark@v1")
    assert loaded is not None
    assert loaded.best().name == "solver-depth-10"
    assert loaded.get("solver-depth-4").score == 0.488
    assert loaded.ref == "connect4-benchmark@v1-reference@v1"


def test_a_missing_board_reads_as_absent(tmp_path):
    assert ReferenceBoardStore(tmp_path).load("never-measured@v1") is None


def test_the_rendered_table_states_the_ceiling_caveat():
    text = render(a_board())
    assert "0.830" in text
    assert "not against 1.000" in text
    # The lesson that was learned the hard way: the ceiling is a measurement.
    assert "measurement, not a deduction" in text


# ---------------------------------------------------------------------------
# the reference players themselves
# ---------------------------------------------------------------------------


def test_every_registered_reference_player_is_playable():
    env = Connect4Env()
    obs = env.reset()
    for name, (description, build) in REFERENCE_PLAYERS.items():
        scores = build()(obs)
        assert scores.shape == (7,), f"{name} must score every column"
        assert np.isfinite(scores).any(), f"{name} produced nothing usable"
        assert description, f"{name} needs a description a human can read"


def test_the_solver_reference_takes_an_immediate_win():
    """It is a reference only if it plays correctly. One ply is enough to take
    a win, which is why rung 1 is not a weak opponent."""
    env = Connect4Env()
    env.reset()
    for m in (0, 6, 1, 6, 2, 6):
        env.step(m)
    assert 3 in env.winning_moves(1)
    for depth in (1, 2, 4):
        scores = solver_as_policy(depth)(env.obs())
        legal = env.legal_actions()
        best = max(legal, key=lambda c: scores[c])
        assert best == 3, f"the depth-{depth} reference missed a win in one"


def test_the_weak_references_score_nothing_so_the_floor_is_real():
    """If the benchmark handed out free points, a score near zero would be
    impossible and the scale would be compressed. It does not."""
    benchmark = SolverLadderBenchmark(min_rung=1, max_rung=1)
    assert benchmark.score(random_as_policy(0)).score == 0.0
    assert benchmark.score(centre_column_policy()).score == 0.0


def test_measuring_refuses_an_unknown_player():
    with pytest.raises(ValueError, match="unknown reference player"):
        measure_reference_board(SolverLadderBenchmark(min_rung=1, max_rung=1),
                                players=["solver-depth-99"], verbose=False)


def test_measuring_one_rung_produces_a_usable_board():
    board = measure_reference_board(
        SolverLadderBenchmark(min_rung=1, max_rung=1),
        players=["random", "solver-depth-1"], verbose=False)
    assert len(board.scores) == 2
    assert board.best().name == "solver-depth-1"
    assert board.get("solver-depth-1").score > board.get("random").score
