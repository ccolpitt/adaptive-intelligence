"""Connect 4 — seed task #1.

Ported from connect4-rl's ConnectFourEnvironment with the same canonical
representation (2-channel, perspective-flipped) but with reward semantics made
explicit and testable:

- ``step`` returns the reward from the MOVING player's perspective
  (+1 win, DRAW_VALUE draw, 0 otherwise) and ``next_obs`` from the NEXT
  player's perspective (the negamax convention the Bellman target relies on).
- The environment itself never emits -1 (the mover cannot lose on their own
  move). Loss attribution to the loser's final transition is done explicitly
  by the episode collector (agents/dqn.py), NOT by a post-hoc buffer patch —
  that patch was a known trap in connect4-rl.

Benchmark v0 (Phase 0 placeholder until the Phase 1 solver exists): win rate
vs a seeded random opponent, both sides, deterministic policy. Fixed and
external — it does not move when the population moves. The solver-based
move-accuracy benchmark replaces it as connect4-benchmark@v1 in Phase 1.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..interfaces import Benchmark, BenchmarkResult, Environment, QFunction, StepResult, Task
from ..task_specs import ActionSpec, ObsSpec

ROWS = 6
COLS = 7
CONNECT_N = 4
DRAW_VALUE = 0.0


class Connect4Env:
    """Two-player Connect 4 with canonical (mover-perspective) observations.

    Board: (ROWS, COLS) int8; 0 empty, +1 player 1, -1 player 2. Row 0 is the
    TOP of the board; pieces fall to the highest-index empty row.
    Observation: float32 (2, ROWS, COLS); channel 0 = current player's pieces,
    channel 1 = opponent's. No turn channel — the flip makes it unnecessary.
    """

    def __init__(self, rows: int = ROWS, cols: int = COLS, connect_n: int = CONNECT_N):
        self.rows = rows
        self.cols = cols
        self.connect_n = connect_n
        self.board = np.zeros((rows, cols), dtype=np.int8)
        self.current_player = 1
        self.last_move: Optional[Tuple[int, int]] = None
        self.done = False
        self.winner: Optional[int] = None  # +1 / -1 / 0 draw / None ongoing

    def reset(self) -> np.ndarray:
        self.board[:] = 0
        self.current_player = 1
        self.last_move = None
        self.done = False
        self.winner = None
        return self.obs()

    def obs(self, player: Optional[int] = None) -> np.ndarray:
        """Canonical observation from ``player``'s perspective (default: mover)."""
        p = self.current_player if player is None else player
        out = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        out[0] = self.board == p
        out[1] = self.board == -p
        return out

    def legal_actions(self) -> List[int]:
        return [c for c in range(self.cols) if self.board[0, c] == 0]

    def step(self, action: int) -> StepResult:
        if self.done:
            raise ValueError("step() called on a finished game")
        if not (0 <= action < self.cols) or self.board[0, action] != 0:
            raise ValueError(f"illegal move: column {action}")

        row = int(np.max(np.where(self.board[:, action] == 0)))
        self.board[row, action] = self.current_player
        self.last_move = (row, action)

        won = self._wins_at(row, action, self.current_player)
        full = not any(self.board[0, c] == 0 for c in range(self.cols))

        if won:
            self.done, self.winner = True, self.current_player
            reward = 1.0
        elif full:
            self.done, self.winner = True, 0
            reward = DRAW_VALUE
        else:
            reward = 0.0

        self.current_player *= -1
        return StepResult(next_obs=self.obs(), reward=reward, done=self.done, winner=self.winner)

    def _wins_at(self, row: int, col: int, player: int) -> bool:
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= self.connect_n:
                return True
        return False

    def winning_moves(self, player: int) -> List[int]:
        """Columns where ``player`` would win immediately (used by scripted
        fixture opponents and, later, the solver benchmark)."""
        wins = []
        for c in self.legal_actions():
            empties = np.where(self.board[:, c] == 0)[0]
            r = int(np.max(empties))
            self.board[r, c] = player
            if self._wins_at(r, c, player):
                wins.append(c)
            self.board[r, c] = 0
        return wins

    def render(self) -> str:
        symbols = {0: ".", 1: "X", -1: "O"}
        rows = [" ".join(symbols[int(v)] for v in row) for row in self.board]
        return "\n".join(rows) + "\n" + " ".join(str(c) for c in range(self.cols))


class RandomVsBenchmark(Benchmark):
    """connect4-benchmark@v0 — deterministic-policy win rate vs seeded random,
    half the games as player 1 and half as player 2.

    Score = (wins + 0.5 * draws) / games. Every ingredient is frozen: the
    opponent seed, the game count, the side split, greedy action selection.
    """

    benchmark_id = "connect4-benchmark"
    version = 0

    def __init__(self, games: int = 200, seed: int = 20260831, task: Optional["Connect4Task"] = None):
        self.games = games
        self.seed = seed
        self._task = task

    def score(self, q_function: QFunction) -> BenchmarkResult:
        rng = np.random.default_rng(self.seed)
        wins = draws = 0
        for g in range(self.games):
            policy_is_p1 = g % 2 == 0
            env = Connect4Env()
            env.reset()
            while not env.done:
                legal = env.legal_actions()
                policy_to_move = (env.current_player == 1) == policy_is_p1
                if policy_to_move:
                    q = q_function(env.obs())
                    masked = np.full(env.cols, -np.inf, dtype=np.float64)
                    masked[legal] = q[legal]
                    action = int(np.argmax(masked))
                else:
                    action = int(rng.choice(legal))
                env.step(action)
            if env.winner == 0:
                draws += 1
            elif (env.winner == 1) == policy_is_p1:
                wins += 1
        score = (wins + 0.5 * draws) / self.games
        return BenchmarkResult(
            benchmark_ref=self.ref,
            score=score,
            detail={"wins": wins, "draws": draws, "games": self.games},
        )


class Connect4Task(Task):
    task_id = "connect4"
    version = 1

    def __init__(self, benchmark_games: int = 200):
        self._benchmark = RandomVsBenchmark(games=benchmark_games)

    @property
    def obs_spec(self) -> ObsSpec:
        return ObsSpec(shape=(2, ROWS, COLS), description="ch0 mover pieces, ch1 opponent")

    @property
    def action_spec(self) -> ActionSpec:
        return ActionSpec(n=COLS, description="drop column")

    def make_env(self) -> Environment:
        return Connect4Env()

    @property
    def benchmark(self) -> Benchmark:
        return self._benchmark

    @property
    def mastery_threshold(self) -> float:
        # vs-random is a weak yardstick; near-perfection required before the
        # v1 solver benchmark takes over in Phase 1.
        return 0.97
