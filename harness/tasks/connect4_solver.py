"""Connect 4 solver — the Phase 1 oracle (roadmap: Measurement Foundation).

This is NOT an agent and NOT trained: it is exhaustive negamax search with
alpha-beta pruning over a bitboard, the mechanical textbook algorithm. It
never competes with the learning system; it grades it (and, depth-limited,
provides a ladder of frozen opponents).

Score semantics (always from the side-to-move's perspective):
- Exact win/loss scores are outside ±WIN_BASE/2 and encode distance:
  +WIN_BASE - ply  = side to move forces a win in `ply` more plies
  -WIN_BASE + ply  = side to move loses in `ply` plies against perfect play
  0 with exact=True = draw under perfect play
- When the depth horizon is hit, a bounded heuristic score in
  (-HEURISTIC_CAP, +HEURISTIC_CAP) is returned (threat counting + center
  control), and the result is marked inexact.

Bitboard layout (standard 7-column x (6+1)-bit encoding):
bit index = col * 7 + row, row 0 = BOTTOM. The 7th bit of each column is a
sentinel that makes shift-based alignment checks safe across column borders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

WIDTH = 7
HEIGHT = 6
H1 = HEIGHT + 1  # bits per column including sentinel

BOTTOM_MASK = sum(1 << (c * H1) for c in range(WIDTH))
BOARD_MASK = BOTTOM_MASK * ((1 << HEIGHT) - 1)

WIN_BASE = 10_000
HEURISTIC_CAP = 1_000
# Center-out move ordering: dramatically improves alpha-beta pruning.
MOVE_ORDER = (3, 2, 4, 1, 5, 0, 6)


def _top_mask(col: int) -> int:
    return 1 << (HEIGHT - 1 + col * H1)


def _bottom_mask(col: int) -> int:
    return 1 << (col * H1)


def _column_mask(col: int) -> int:
    return ((1 << HEIGHT) - 1) << (col * H1)


def has_alignment(p: int) -> bool:
    """True if bitboard ``p`` contains four in a row (any direction)."""
    # vertical
    m = p & (p >> 1)
    if m & (m >> 2):
        return True
    # horizontal
    m = p & (p >> H1)
    if m & (m >> (2 * H1)):
        return True
    # diagonal /
    m = p & (p >> HEIGHT)
    if m & (m >> (2 * HEIGHT)):
        return True
    # diagonal \
    m = p & (p >> (HEIGHT + 2))
    if m & (m >> (2 * (HEIGHT + 2))):
        return True
    return False


def winning_spots(position: int, mask: int) -> int:
    """Bitmask of empty cells that would complete four-in-a-row for
    ``position``'s stones. Used by the leaf heuristic (threat counting)."""
    # vertical
    r = (position << 1) & (position << 2) & (position << 3)
    for shift in (H1, HEIGHT, HEIGHT + 2):  # horizontal, diag /, diag \
        p = (position << shift) & (position << (2 * shift))
        r |= p & (position << (3 * shift))
        r |= p & (position >> shift)
        p = (position >> shift) & (position >> (2 * shift))
        r |= p & (position >> (3 * shift))
        r |= p & (position << shift)
    return r & (BOARD_MASK ^ mask)


class BitBoard:
    """Position from the side-to-move's perspective: ``position`` holds the
    mover's stones, ``mask`` all stones."""

    __slots__ = ("position", "mask", "plies")

    def __init__(self, position: int = 0, mask: int = 0, plies: int = 0):
        self.position = position
        self.mask = mask
        self.plies = plies

    @classmethod
    def from_array(cls, board: np.ndarray, current_player: int) -> "BitBoard":
        """Build from the env's (6, 7) array (row 0 = TOP) and mover id."""
        position = mask = 0
        plies = 0
        for col in range(WIDTH):
            for row in range(HEIGHT):  # row index in array, 0 = top
                v = int(board[row, col])
                if v == 0:
                    continue
                bit = 1 << (col * H1 + (HEIGHT - 1 - row))  # flip to bottom-up
                mask |= bit
                plies += 1
                if v == current_player:
                    position |= bit
        return cls(position, mask, plies)

    def can_play(self, col: int) -> bool:
        return (self.mask & _top_mask(col)) == 0

    def legal_moves(self) -> List[int]:
        return [c for c in MOVE_ORDER if self.can_play(c)]

    def play(self, col: int) -> "BitBoard":
        """Returns the successor position (perspective flipped to next mover).

        NOTE: the flip uses the OLD mask — the next mover's stones must not
        include the stone just played (it belongs to the previous mover)."""
        next_position = self.position ^ self.mask  # opponent's stones
        new_mask = self.mask | (self.mask + _bottom_mask(col))
        return BitBoard(next_position, new_mask, self.plies + 1)

    def is_winning_move(self, col: int) -> bool:
        pos = self.position | ((self.mask + _bottom_mask(col)) & _column_mask(col))
        return has_alignment(pos)

    def is_full(self) -> bool:
        return self.mask == BOARD_MASK

    def key(self) -> int:
        """Unique position key (standard trick: position + mask is injective)."""
        return self.position + self.mask + BOTTOM_MASK


def _heuristic(board: BitBoard) -> int:
    """Bounded leaf evaluation from the mover's perspective: my open threats
    minus opponent's, plus center control. Never used when exact values are
    reachable — only at the depth horizon."""
    me = board.position
    opp = board.position ^ board.mask
    my_threats = bin(winning_spots(me, board.mask)).count("1")
    opp_threats = bin(winning_spots(opp, board.mask)).count("1")
    center = _column_mask(3)
    my_center = bin(me & center).count("1")
    opp_center = bin(opp & center).count("1")
    score = 30 * (my_threats - opp_threats) + 4 * (my_center - opp_center)
    return max(-HEURISTIC_CAP + 1, min(HEURISTIC_CAP - 1, score))


@dataclass(frozen=True)
class MoveEval:
    col: int
    score: int
    exact: bool


_EXACT, _LOWER, _UPPER = 0, 1, 2


class Connect4Solver:
    """Depth-limited (or exact) negamax with alpha-beta and a transposition
    table. ``max_depth=None`` means search to the end of the game: exact
    game-theoretic values, tractable for endgames and (slowly) mid-game."""

    def __init__(self, max_depth: Optional[int] = None):
        self.max_depth = max_depth
        self._tt: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.nodes = 0

    # -- public API -------------------------------------------------------

    def evaluate_moves(self, board: np.ndarray, current_player: int) -> List[MoveEval]:
        """Score every legal move from the mover's perspective. The best move
        is the max. Exactness is per-move: a proven win/loss/draw is exact,
        a horizon heuristic value is not."""
        bb = BitBoard.from_array(board, current_player)
        depth = self.max_depth if self.max_depth is not None else (42 - bb.plies)
        out = []
        for col in sorted(bb.legal_moves()):
            if bb.is_winning_move(col):
                out.append(MoveEval(col, WIN_BASE - (bb.plies + 1), True))
                continue
            child = bb.play(col)
            score = -self._negamax(child, depth - 1, -WIN_BASE, WIN_BASE)
            exact = abs(score) > HEURISTIC_CAP or self._exhausted(child, depth - 1)
            out.append(MoveEval(col, score, exact))
        return out

    def best_move(self, board: np.ndarray, current_player: int) -> int:
        evals = self.evaluate_moves(board, current_player)
        return max(evals, key=lambda e: e.score).col

    # -- search -----------------------------------------------------------

    def _exhausted(self, bb: BitBoard, depth: int) -> bool:
        return depth >= 42 - bb.plies

    def _negamax(self, bb: BitBoard, depth: int, alpha: int, beta: int) -> int:
        """Score of ``bb`` for its side to move."""
        self.nodes += 1

        if bb.is_full():
            return 0
        # Immediate win available?
        legal = bb.legal_moves()
        for col in legal:
            if bb.is_winning_move(col):
                return WIN_BASE - (bb.plies + 1)
        if depth <= 0:
            return _heuristic(bb)

        # Transposition table with bound flags: under alpha-beta cutoffs a
        # stored value may be only a lower/upper bound; reusing bounds as
        # exact values silently corrupts the oracle.
        key = (bb.key(), -1 if self._exhausted(bb, depth) else depth)
        entry = self._tt.get(key)
        if entry is not None:
            flag, value = entry
            if flag == _EXACT:
                return value
            if flag == _LOWER and value >= beta:
                return value
            if flag == _UPPER and value <= alpha:
                return value

        alpha_orig = alpha
        best = -WIN_BASE
        for col in legal:
            child = bb.play(col)
            score = -self._negamax(child, depth - 1, -beta, -alpha)
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break  # cutoff

        if best <= alpha_orig:
            flag = _UPPER
        elif best >= beta:
            flag = _LOWER
        else:
            flag = _EXACT
        self._tt[key] = (flag, best)
        return best


class SolverOpponent:
    """Wraps the solver as a harness Opponent-compatible player (frozen,
    deterministic given depth): the depth ladder's rungs.

    Reconstructs the board from the canonical 2-channel observation, so it
    plugs into the evaluator exactly like the scripted opponents.
    """

    def __init__(self, depth: int):
        self.depth = depth
        self.name = f"solver-d{depth}"
        self._solver = Connect4Solver(max_depth=depth)

    def act(self, obs: np.ndarray, legal) -> int:
        board = (obs[0] - obs[1]).astype(np.int8)  # mover = +1, opponent = -1
        return self._solver.best_move(board, current_player=1)
