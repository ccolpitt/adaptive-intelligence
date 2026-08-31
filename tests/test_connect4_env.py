"""Environment correctness: wins in all four directions (including edges),
illegal-move rejection, draw, turn alternation, perspective-flip invariant,
and a golden game."""

import numpy as np
import pytest

from harness.tasks.connect4 import Connect4Env, DRAW_VALUE


def play(env, moves):
    result = None
    for col in moves:
        result = env.step(col)
    return result


def test_reset_and_alternation():
    env = Connect4Env()
    env.reset()
    assert env.current_player == 1
    env.step(0)
    assert env.current_player == -1
    env.step(1)
    assert env.current_player == 1


def test_vertical_win():
    env = Connect4Env()
    env.reset()
    # P1 stacks column 0; P2 wanders.
    result = play(env, [0, 1, 0, 2, 0, 3, 0])
    assert result.done and result.winner == 1 and result.reward == 1.0


def test_horizontal_win():
    env = Connect4Env()
    env.reset()
    result = play(env, [0, 0, 1, 1, 2, 2, 3])
    assert result.done and result.winner == 1


def test_diagonal_up_right_win():
    env = Connect4Env()
    env.reset()
    # P1 builds the / diagonal at columns 0-3.
    moves = [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3]
    result = play(env, moves)
    assert result.done and result.winner == 1


def test_diagonal_down_right_win():
    env = Connect4Env()
    env.reset()
    # Mirror: P1 builds the \ diagonal at columns 3-0.
    moves = [3, 2, 2, 1, 1, 0, 1, 0, 0, 6, 0]
    result = play(env, moves)
    assert result.done and result.winner == 1


def test_win_at_right_edge():
    env = Connect4Env()
    env.reset()
    result = play(env, [3, 3, 4, 4, 5, 5, 6])
    assert result.done and result.winner == 1


def test_player2_can_win():
    env = Connect4Env()
    env.reset()
    result = play(env, [0, 6, 1, 6, 0, 6, 1, 6])
    assert result.done and result.winner == -1 and result.reward == 1.0  # mover reward


def test_illegal_move_full_column():
    env = Connect4Env()
    env.reset()
    for _ in range(3):
        env.step(0)
        env.step(0)
    with pytest.raises(ValueError):
        env.step(0)


def test_illegal_move_out_of_range():
    env = Connect4Env()
    env.reset()
    with pytest.raises(ValueError):
        env.step(7)


def test_step_after_done_raises():
    env = Connect4Env()
    env.reset()
    play(env, [0, 1, 0, 1, 0, 1, 0])
    with pytest.raises(ValueError):
        env.step(3)


def test_draw_full_board():
    env = Connect4Env()
    env.reset()
    # Column-pair pattern that provably yields no 4-in-a-row: fill columns in
    # the order 0,1,2 / 1,2,0 alternating blocks. Verified draw sequence:
    cols = []
    for block in ([0, 1, 2], [3, 4, 5]):
        for _ in range(3):
            cols.extend(block * 2)
    # fill column 6 last
    cols.extend([6] * 6)
    result = None
    for c in cols:
        assert not env.done, "game ended early — not a draw sequence"
        result = env.step(c)
    assert result.done and result.winner == 0 and result.reward == DRAW_VALUE


def test_perspective_flip_invariant():
    """Channel 0 is ALWAYS the mover's pieces."""
    env = Connect4Env()
    obs = env.reset()
    assert obs.sum() == 0
    r1 = env.step(3)  # P1 played; obs now from P2's perspective
    assert r1.next_obs[1, 5, 3] == 1.0  # P1's piece is in the OPPONENT channel
    assert r1.next_obs[0].sum() == 0  # P2 has no pieces yet
    r2 = env.step(2)  # P2 played; back to P1's perspective
    assert r2.next_obs[0, 5, 3] == 1.0  # P1's own piece in channel 0
    assert r2.next_obs[1, 5, 2] == 1.0  # P2's piece in channel 1


def test_winning_moves_helper():
    env = Connect4Env()
    env.reset()
    play(env, [0, 6, 0, 6, 0, 6])  # P1 has 3 in column 0; P2 has 3 in column 6
    assert env.winning_moves(1) == [0]
    assert env.winning_moves(-1) == [6]


def test_golden_game_board_state():
    env = Connect4Env()
    env.reset()
    play(env, [3, 3, 4])
    expected = np.zeros((6, 7), dtype=np.int8)
    expected[5, 3] = 1
    expected[4, 3] = -1
    expected[5, 4] = 1
    assert (env.board == expected).all()
    assert env.current_player == -1
