"""Solver correctness: bitboard round-trip, cross-validation against the
environment's win detection, tactical ground truth (take wins, block losses,
prefer faster wins), exact endgame values, and depth-ladder ordering."""

import numpy as np
import pytest

from harness.tasks.connect4 import Connect4Env
from harness.tasks.connect4_solver import (
    BitBoard,
    Connect4Solver,
    HEURISTIC_CAP,
    SolverOpponent,
    WIN_BASE,
    has_alignment,
)


def env_after(moves):
    env = Connect4Env()
    env.reset()
    for c in moves:
        env.step(c)
    return env


# -- bitboard core ---------------------------------------------------------

def test_from_array_round_trip_counts():
    env = env_after([3, 3, 4, 2, 5])
    bb = BitBoard.from_array(env.board, env.current_player)
    assert bb.plies == 5
    assert bin(bb.mask).count("1") == 5
    assert bin(bb.position).count("1") == 2  # mover (P2) has 2 stones


def test_bitboard_win_detection_matches_env():
    """Property test: for random playouts, is_winning_move must agree with
    the environment's ground truth on every legal move of every position."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        env = Connect4Env()
        env.reset()
        while not env.done:
            bb = BitBoard.from_array(env.board, env.current_player)
            for col in env.legal_actions():
                clone = env_after([])
                clone.board = env.board.copy()
                clone.current_player = env.current_player
                result = clone.step(col)
                env_says_win = result.winner == env.current_player
                assert bb.is_winning_move(col) == env_says_win, (
                    f"bitboard/env disagree at col {col}\n{env.render()}"
                )
            env.step(int(rng.choice(env.legal_actions())))


def test_has_alignment_directions():
    env = env_after([0, 1, 0, 2, 0, 3, 0])  # P1 vertical win in col 0
    bb = BitBoard.from_array(env.board, 1)
    assert has_alignment(bb.position)


# -- tactics ---------------------------------------------------------------

def test_takes_immediate_win_at_depth_1():
    env = env_after([0, 6, 0, 6, 0, 6])  # P1 to move: col 0 wins now
    solver = Connect4Solver(max_depth=1)
    assert solver.best_move(env.board, env.current_player) == 0


def test_blocks_immediate_loss():
    # P2 threatens col 6 (three stacked). P1 must block at 6.
    env = env_after([0, 6, 1, 6, 0, 6])  # P1 to move, no win available
    solver = Connect4Solver(max_depth=4)
    assert solver.best_move(env.board, env.current_player) == 6


def test_prefers_faster_win():
    # P1 has an immediate win at col 0 (three stacked) — must take it rather
    # than any slower forced win elsewhere.
    env = env_after([0, 5, 0, 5, 0, 6])  # P1: col0 x3 -> immediate win at 0
    solver = Connect4Solver(max_depth=8)
    evals = {e.col: e for e in solver.evaluate_moves(env.board, env.current_player)}
    assert evals[0].exact and evals[0].score == WIN_BASE - 7  # win at ply 7
    best = max(evals.values(), key=lambda e: e.score)
    assert best.col == 0


def test_exact_endgame_value_win():
    """Near-endgame with a forced win must come back exact (not heuristic)."""
    # Double threat: P1 has stones at cols 2,3 on the bottom row with 1 and 4
    # both open — unstoppable at depth >= 3.
    env = env_after([2, 2, 3, 3])  # P1: bottom 2,3; P2 stacked above
    solver = Connect4Solver(max_depth=10)
    evals = {e.col: e for e in solver.evaluate_moves(env.board, env.current_player)}
    # Playing 1 or 4 creates the open three -> check the position is winning.
    best = max(evals.values(), key=lambda e: e.score)
    assert best.score > HEURISTIC_CAP, "forced win not found as exact value"


def test_exact_draw_detection():
    """A board with one empty cell left and no winner: proven draw."""
    env = Connect4Env()
    env.reset()
    cols = []
    for block in ([0, 1, 2], [3, 4, 5]):
        for _ in range(3):
            cols.extend(block * 2)
    cols.extend([6] * 5)  # leave ONE empty cell at top of col 6
    for c in cols:
        env.step(c)
    assert not env.done
    solver = Connect4Solver(max_depth=None)  # exact mode
    evals = solver.evaluate_moves(env.board, env.current_player)
    assert len(evals) == 1
    assert evals[0].exact and evals[0].score == 0


def test_deterministic():
    env = env_after([3, 2, 4])
    a = Connect4Solver(max_depth=6).evaluate_moves(env.board, env.current_player)
    b = Connect4Solver(max_depth=6).evaluate_moves(env.board, env.current_player)
    assert a == b


# -- depth ladder ----------------------------------------------------------

def _solver_vs_solver(depth_a, depth_b, games=6):
    """A as P1 in half the games. Returns A's (wins, losses, draws).
    Both deterministic, so alternate openings to vary the games."""
    wins = losses = draws = 0
    for g in range(games):
        a_is_p1 = g % 2 == 0
        env = Connect4Env()
        env.reset()
        env.step(g % 7)  # varied forced opening move (counts as P1's move)
        sa, sb = Connect4Solver(depth_a), Connect4Solver(depth_b)
        while not env.done:
            mover_is_a = (env.current_player == 1) == a_is_p1
            s = sa if mover_is_a else sb
            env.step(s.best_move(env.board, env.current_player))
        if env.winner == 0:
            draws += 1
        elif (env.winner == 1) == a_is_p1:
            wins += 1
        else:
            losses += 1
    return wins, losses, draws


@pytest.mark.slow
def test_depth_ladder_ordering():
    """Depth 6 must dominate depth 1 — the ladder's rungs are really ordered."""
    wins, losses, draws = _solver_vs_solver(6, 1, games=6)
    assert wins > losses, f"depth 6 did not beat depth 1: {wins}-{losses}-{draws}"


def test_solver_opponent_wrapper():
    """SolverOpponent reconstructs the board from the canonical obs and takes
    an immediate win."""
    env = env_after([0, 6, 0, 6, 0, 6])  # mover wins at col 0
    opp = SolverOpponent(depth=2)
    move = opp.act(env.obs(), env.legal_actions())
    assert move == 0
