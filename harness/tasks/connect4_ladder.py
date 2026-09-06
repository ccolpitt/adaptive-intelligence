"""connect4-benchmark@v1 — the progressive solver ladder.

The benchmark is a FIXED, versioned measurement procedure (never changes when
the population changes). It plays the policy against the solver oracle at
every rung depth d = 1..MAX_RUNG and reports:

- per-rung score s_d = (wins + 0.5 * draws) / games
- aggregate = mean over rungs (the scalar used by gates), in [0, 1]
- frontier_depth = the lowest rung with s_d < MASTERY_PER_RUNG (0.90) —
  "the next depth to beat". The PROGRESSION is a property of the agent
  climbing, not of the benchmark moving: every policy, weak or strong, is
  measured against the identical full ladder.

Determinism note: the policy plays greedily and the solver is deterministic,
so a (side, opening) pair defines exactly one game. Playing more copies of
the same game adds zero information. Variety therefore comes from a FIXED
set of forced openings (all 7 single-move openings + 7 seeded two-move
openings), each played from both sides: 28 distinct games per rung,
168 per benchmark call. At z=1.645 the aggregate's detectable effect is
about 9 percentage points (see stats.detectable_effect) — hypotheses must
state this.

Rules are per-move time-unbounded but depth-bounded: rung d = solver with
max_depth d. Task v2 = same Connect 4 environment + reward, measured by this
ladder (changed benchmark => new task version, per the task registry rule).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..interfaces import Benchmark, BenchmarkResult, QFunction, Task
from ..task_specs import ActionSpec, ObsSpec
from .connect4 import COLS, Connect4Env, ROWS
from .connect4_solver import Connect4Solver

MIN_RUNG = 1
MAX_RUNG = 6
MASTERY_PER_RUNG = 0.90
OPENING_SEED = 20260901


def _fixed_openings() -> List[Tuple[int, ...]]:
    """7 single-move openings + 7 seeded two-move openings. Frozen: same
    list forever for benchmark v1."""
    openings: List[Tuple[int, ...]] = [(c,) for c in range(COLS)]
    rng = np.random.default_rng(OPENING_SEED)
    seen = set(openings)
    while len(openings) < 14:
        pair = (int(rng.integers(0, COLS)), int(rng.integers(0, COLS)))
        if pair not in seen:
            seen.add(pair)
            openings.append(pair)
    return openings


class SolverLadderBenchmark(Benchmark):
    benchmark_id = "connect4-benchmark"
    version = 1

    def __init__(self, min_rung: int = MIN_RUNG, max_rung: int = MAX_RUNG):
        self.rungs = list(range(min_rung, max_rung + 1))
        self.openings = _fixed_openings()

    @property
    def games_per_rung(self) -> int:
        return 2 * len(self.openings)

    @property
    def games_total(self) -> int:
        return self.games_per_rung * len(self.rungs)

    def score(self, q_function: QFunction) -> BenchmarkResult:
        per_rung = {}
        for depth in self.rungs:
            solver = Connect4Solver(max_depth=depth)  # TT persists across games
            wins = draws = 0
            for policy_is_p1 in (True, False):
                for opening in self.openings:
                    winner = self._play(q_function, solver, opening, policy_is_p1)
                    if winner == 0:
                        draws += 1
                    elif (winner == 1) == policy_is_p1:
                        wins += 1
            per_rung[depth] = (wins + 0.5 * draws) / self.games_per_rung

        aggregate = float(np.mean(list(per_rung.values())))
        frontier = next(
            (d for d in self.rungs if per_rung[d] < MASTERY_PER_RUNG),
            self.rungs[-1] + 1,
        )
        return BenchmarkResult(
            benchmark_ref=self.ref,
            score=aggregate,
            detail={
                "per_rung": {str(d): round(s, 4) for d, s in per_rung.items()},
                "frontier_depth": frontier,
                "games_per_rung": self.games_per_rung,
                "games_total": self.games_total,
                "mastery_per_rung": MASTERY_PER_RUNG,
            },
        )

    @staticmethod
    def _play(
        q_function: QFunction,
        solver: Connect4Solver,
        opening: Sequence[int],
        policy_is_p1: bool,
    ) -> Optional[int]:
        env = Connect4Env()
        env.reset()
        for col in opening:  # forced opening moves, imposed on both players
            env.step(col)
            if env.done:  # unreachable for <=2 moves, guarded anyway
                return env.winner
        while not env.done:
            legal = env.legal_actions()
            if (env.current_player == 1) == policy_is_p1:
                q = q_function(env.obs())
                masked = np.full(env.cols, -np.inf, dtype=np.float64)
                masked[legal] = q[legal]
                env.step(int(np.argmax(masked)))
            else:
                env.step(solver.best_move(env.board, env.current_player))
        return env.winner


class Connect4LadderTask(Task):
    """connect4@v2: identical environment and reward to v1; measured by the
    solver ladder instead of vs-random."""

    task_id = "connect4"
    version = 2

    def __init__(self, min_rung: int = MIN_RUNG, max_rung: int = MAX_RUNG):
        self._benchmark = SolverLadderBenchmark(min_rung, max_rung)

    @property
    def obs_spec(self) -> ObsSpec:
        return ObsSpec(shape=(2, ROWS, COLS), description="ch0 mover pieces, ch1 opponent")

    @property
    def action_spec(self) -> ActionSpec:
        return ActionSpec(n=COLS, description="drop column")

    def make_env(self) -> Connect4Env:
        return Connect4Env()

    @property
    def benchmark(self) -> SolverLadderBenchmark:
        return self._benchmark

    @property
    def mastery_threshold(self) -> float:
        # Aggregate ladder score when every rung is at MASTERY_PER_RUNG.
        # The loop additionally reports frontier_depth; true mastery of the
        # task = frontier beyond MAX_RUNG.
        return MASTERY_PER_RUNG
