"""One-move tactics split by geometry and by where the positions came from.

The single take-the-win and block-the-loss numbers mixed three questions
together and read pessimistically, which nearly sent an investigation after
the value head when the fault was the move head off its own distribution.
These tests cover the split, and especially the attribution rule: a position
that cannot be ascribed to one geometry is not counted under any.
"""

import numpy as np
import pytest

from harness.tactical_probe import (
    DIRECTIONS,
    GEOMETRIES,
    TacticalBreakdown,
    geometry_of_winning_move,
    lines_completed_at,
    probe_in_distribution,
    probe_off_distribution,
    render,
)
from harness.tasks.connect4 import Connect4Env


def board_from(rows):
    """Build a position from a picture. 'X' = player 1, 'O' = player -1.
    Row 0 is the TOP, matching the environment."""
    env = Connect4Env()
    env.reset()
    for r, line in enumerate(rows):
        for c, ch in enumerate(line.replace(" ", "")):
            env.board[r, c] = {".": 0, "X": 1, "O": -1}[ch]
    return env


# ---------------------------------------------------------------------------
# recognising which kind of line a move completes
# ---------------------------------------------------------------------------


def test_a_vertical_win_is_named_vertical():
    env = board_from([
        ".......",
        ".......",
        ".......",
        "X......",
        "X......",
        "X......",
    ])
    assert 0 in env.winning_moves(1)
    assert geometry_of_winning_move(env, 1, 0) == "vertical"


def test_a_horizontal_win_is_named_horizontal():
    env = board_from([
        ".......",
        ".......",
        ".......",
        ".......",
        ".......",
        "XXX.OOO",
    ])
    assert 3 in env.winning_moves(1)
    assert geometry_of_winning_move(env, 1, 3) == "horizontal"


def test_a_rising_diagonal_is_distinguished_from_a_falling_one():
    """X on (5,0), (4,1), (3,2); playing column 3 lands on row 2 and
    completes the line that rises to the right. Every column below a piece
    must be filled, or the position could not occur."""
    env = board_from([
        ".......",
        ".......",
        ".......",
        "..XO...",
        ".XOO...",
        "XOOO...",
    ])
    assert 3 in env.winning_moves(1)
    assert geometry_of_winning_move(env, 1, 3) == "diagonal/"

    mirrored = board_from([
        ".......",
        ".......",
        ".......",
        "...OX..",
        "...OOX.",
        "...OOOX",
    ])
    assert 3 in mirrored.winning_moves(1)
    assert geometry_of_winning_move(mirrored, 1, 3) == "diagonal\\"


def test_a_move_completing_two_lines_at_once_is_not_attributed():
    """Deliberate: a double line cannot be ascribed to one geometry, so
    counting it anywhere would corrupt the comparison it exists to support."""
    env = board_from([
        ".......",
        ".......",
        "......X",
        ".....XO",
        "....XOO",
        "XXX.OOO",
    ])
    assert 3 in env.winning_moves(1)
    # Playing column 3 lands on row 5 and completes the bottom row AND the
    # diagonal rising to the right through (4,4), (3,5), (2,6).
    env.board[5, 3] = 1
    assert lines_completed_at(env, 5, 3, 1) == {"horizontal", "diagonal/"}
    env.board[5, 3] = 0
    assert geometry_of_winning_move(env, 1, 3) is None


def test_a_move_that_wins_nothing_is_not_attributed():
    env = board_from([
        ".......",
        ".......",
        ".......",
        ".......",
        ".......",
        "X......",
    ])
    assert geometry_of_winning_move(env, 1, 6) is None


def test_the_probe_never_mutates_the_position():
    env = board_from([
        ".......",
        ".......",
        ".......",
        "X......",
        "X......",
        "X......",
    ])
    before = env.board.copy()
    for col in range(7):
        geometry_of_winning_move(env, 1, col)
        geometry_of_winning_move(env, -1, col)
    assert np.array_equal(env.board, before)


def test_every_direction_has_a_name():
    assert set(DIRECTIONS.values()) == set(GEOMETRIES)


# ---------------------------------------------------------------------------
# the breakdown
# ---------------------------------------------------------------------------


def test_rates_are_reported_per_geometry_and_overall():
    b = TacticalBreakdown("in-distribution")
    b.took_of["vertical"] = 10
    b.took["vertical"] = 9
    b.took_of["diagonal/"] = 10
    b.took["diagonal/"] = 4
    assert b.take_rate("vertical") == 0.9
    assert b.take_rate("diagonal/") == 0.4
    assert b.take_rate() == pytest.approx(0.65)


def test_a_geometry_with_no_samples_reports_nothing_rather_than_zero():
    b = TacticalBreakdown("off-distribution")
    assert b.take_rate("vertical") is None
    assert b.block_rate() is None, "no samples is not the same as failing"


def test_the_weakest_block_ignores_geometries_with_too_few_samples():
    b = TacticalBreakdown("off-distribution")
    b.blocked_of["vertical"] = 100
    b.blocked["vertical"] = 95
    b.blocked_of["horizontal"] = 100
    b.blocked["horizontal"] = 60
    b.blocked_of["diagonal/"] = 3      # too few to rank
    b.blocked["diagonal/"] = 0
    assert b.worst_block() == ("horizontal", 0.6)


def test_the_breakdown_serialises_with_its_sample_counts():
    b = TacticalBreakdown("in-distribution")
    b.blocked_of["vertical"] = 4
    b.blocked["vertical"] = 3
    data = b.to_dict()
    assert data["distribution"] == "in-distribution"
    assert data["block_the_loss"]["vertical"] == 0.75
    assert data["block_the_loss_samples"]["vertical"] == 4, (
        "a rate without its sample count invites over-reading")


# ---------------------------------------------------------------------------
# running the probe
# ---------------------------------------------------------------------------


def test_a_policy_that_always_takes_a_win_scores_one():
    """The probe has to be able to report a perfect score, or a low number
    means nothing."""
    def perfect(obs):
        env = Connect4Env()
        env.board = (obs[0] - obs[1]).astype(np.int8)
        env.current_player = 1
        scores = np.zeros(7)
        for col in env.winning_moves(1):
            scores[col] = 10.0
        for col in env.winning_moves(-1):
            scores[col] = max(scores[col], 5.0)
        return scores

    breakdown = probe_off_distribution(perfect, games=40, seed=0)
    assert breakdown.take_rate() == 1.0
    assert breakdown.block_rate() == 1.0


def test_a_policy_that_ignores_tactics_scores_poorly():
    constant = lambda obs: np.array([9.0, 0, 0, 0, 0, 0, 0])
    breakdown = probe_off_distribution(constant, games=40, seed=1)
    assert breakdown.take_rate() < 0.5
    assert sum(breakdown.took_of.values()) > 0, "the probe found no tactics"


def test_both_distributions_are_probed_and_reported_separately():
    from harness.agents.policy_value import PolicyValueConfig, PolicyValueTrainer

    trainer = PolicyValueTrainer(6, 7, 7, PolicyValueConfig(
        seed=0, search_budget=16))
    in_dist = probe_in_distribution(trainer.q_function(), trainer.search,
                                    games=3, seed=0)
    off_dist = probe_off_distribution(trainer.q_function(), games=10, seed=0)
    assert in_dist.distribution == "in-distribution"
    assert off_dist.distribution == "off-distribution"
    text = render([in_dist, off_dist])
    assert "in-distribution" in text and "off-distribution" in text
    for geometry in GEOMETRIES:
        assert geometry in text
