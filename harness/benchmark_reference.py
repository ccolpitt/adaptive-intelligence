"""What a score on the benchmark is worth, measured against known players.

A ladder score lives in [0, 1], and for the whole project so far it has been
read as though 1.0 were the target and 0.90 at every rung were mastery. That
reading was never checked, and it is wrong.

Measured here: an exhaustive negamax solver searching eight plies — far
stronger than anything this project has trained — scores **0.717** on the
ladder, and one searching ten plies scores **0.830**. Neither is close to
1.0. Part of the reason is the benchmark's own design: both players are forced
through fixed opening moves, and the policy plays both sides of every opening.
In Connect 4 the first player wins with perfect play only by opening in the
centre; forced into another column the game is a draw or a loss. So a share of
the 168 games begin from positions that are objectively lost, and against an
opponent that takes every win and blocks every threat — which every rung does,
including depth 1, because the solver checks for an immediate win before it
checks its depth limit — no amount of skill recovers them.

A caution that was learned the hard way here: the ceiling is a measurement,
not a deduction, and it moves with the reference. On the evidence of depth 8
alone it looked as though roughly 0.72 were the practical limit and the 0.90
mastery bar were unreachable. Depth 10 then scored 0.830, with 0.875 at two
separate rungs. Whether 0.90 is reachable is still open; what is settled is
that 1.0 is not, and that a score must be read against the best reference
measured rather than against the top of the range.

This does not make the benchmark bad. It is fixed, versioned and perfectly
reproducible; re-scoring an archived policy reproduces its recorded per-rung
numbers digit for digit. What it makes is a benchmark whose **scale needs
calibrating**, because "0.65 out of 1.0" and "0.65 against a reference of
0.72" support completely different conclusions, and the project has been
drawing the first kind.

So the scale is pinned with reference players: fixed, reproducible opponents
of known construction, scored on the identical benchmark. A policy's number
is then read against them. This is the same argument the project already
makes for using an absolute benchmark instead of self-play Elo, applied one
level up: an absolute score still needs to be anchored before it means
anything.

Reference players are expensive to compute (minutes each) and never change,
so a board is measured once, stored as JSON, and reused.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from .jsonl_store import now_iso


@dataclass
class ReferenceScore:
    """One known player's standing on the benchmark."""

    name: str
    description: str
    score: float
    per_rung: Dict[str, float]
    seconds: float = 0.0

    def to_dict(self) -> Dict:
        return {"name": self.name, "description": self.description,
                "score": self.score, "per_rung": self.per_rung,
                "seconds": round(self.seconds, 1)}


@dataclass
class ReferenceBoard:
    """The benchmark's scale, pinned by players whose strength is known by
    construction rather than by training."""

    benchmark_ref: str
    version: int
    scores: List[ReferenceScore] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)

    @property
    def ref(self) -> str:
        return f"{self.benchmark_ref}-reference@v{self.version}"

    def best(self) -> Optional[ReferenceScore]:
        """The strongest reference score — the practical ceiling, as far as
        anything measured so far reaches."""
        return max(self.scores, key=lambda s: s.score) if self.scores else None

    def get(self, name: str) -> Optional[ReferenceScore]:
        return next((s for s in self.scores if s.name == name), None)

    def to_dict(self) -> Dict:
        return {"benchmark_ref": self.benchmark_ref, "version": self.version,
                "ref": self.ref, "created_at": self.created_at,
                "scores": [s.to_dict() for s in self.scores]}

    @classmethod
    def from_dict(cls, data: Dict) -> "ReferenceBoard":
        return cls(
            benchmark_ref=data["benchmark_ref"], version=data["version"],
            created_at=data.get("created_at", now_iso()),
            scores=[ReferenceScore(**{k: v for k, v in s.items()})
                    for s in data.get("scores", [])],
        )

    # -- reading a score against the board ------------------------------

    def place(self, score: float) -> Dict:
        """Where a score sits on the pinned scale."""
        ranked = sorted(self.scores, key=lambda s: s.score)
        below = [s for s in ranked if s.score <= score]
        above = [s for s in ranked if s.score > score]
        best = self.best()
        return {
            "score": score,
            "beats": [s.name for s in below],
            "next_reference": above[0].name if above else None,
            "next_reference_score": above[0].score if above else None,
            "best_reference": best.name if best else None,
            "best_reference_score": best.score if best else None,
            "fraction_of_best": (score / best.score) if best and best.score else None,
        }

    def describe(self, score: float, label: str = "this policy") -> str:
        """One or two sentences a human can read without the table."""
        placement = self.place(score)
        best, best_score = placement["best_reference"], placement["best_reference_score"]
        if best is None:
            return f"{label} scores {score:.3f}; no reference players are measured."
        lines = [f"{label} scores {score:.3f} on {self.benchmark_ref}."]
        lines.append(
            f"The strongest reference player, {best}, scores {best_score:.3f} on "
            f"the identical games, so {score:.3f} is "
            f"{placement['fraction_of_best']:.0%} of what a known-strong player "
            f"achieves here — NOT {score:.0%} of a reachable 1.0.")
        if placement["beats"]:
            lines.append(f"It is at or above: {', '.join(placement['beats'])}.")
        if placement["next_reference"]:
            lines.append(
                f"The next reference above it is {placement['next_reference']} at "
                f"{placement['next_reference_score']:.3f}.")
        return " ".join(lines)


class ReferenceBoardStore:
    """One board per benchmark version, stored as data and rendered on demand."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, benchmark_ref: str, version: int = 1) -> Path:
        safe = benchmark_ref.replace("/", "-").replace("@", "-at-")
        return self.root / f"{safe}-reference-v{version}.json"

    def save(self, board: ReferenceBoard) -> Path:
        path = self.path_for(board.benchmark_ref, board.version)
        path.write_text(json.dumps(board.to_dict(), indent=2))
        return path

    def load(self, benchmark_ref: str, version: int = 1) -> Optional[ReferenceBoard]:
        path = self.path_for(benchmark_ref, version)
        if not path.exists():
            return None
        return ReferenceBoard.from_dict(json.loads(path.read_text()))


# ---------------------------------------------------------------------------
# the reference players
# ---------------------------------------------------------------------------


def solver_as_policy(depth: int) -> Callable[[np.ndarray], np.ndarray]:
    """An exhaustive negamax solver, searching ``depth`` plies, presented as
    a policy so the benchmark can score it exactly like a network.

    Note what this is NOT: a perfect player. Beyond its depth limit it falls
    back on a crude heuristic, and a deeper search over a crude heuristic can
    play WORSE than a shallow one — a depth-8 solver loses to a depth-1 solver
    in some specific openings. Reference players anchor the scale; they are
    not an upper bound on play.
    """
    from .tasks.connect4_solver import Connect4Solver

    solver = Connect4Solver(max_depth=depth)

    def q(obs: np.ndarray) -> np.ndarray:
        board = (obs[0] - obs[1]).astype(np.int8)  # canonical: mover is +1
        scores = np.full(7, -1e9, dtype=np.float64)
        for ev in solver.evaluate_moves(board, current_player=1):
            scores[ev.col] = ev.score
        return scores

    return q


def random_as_policy(seed: int = 0) -> Callable[[np.ndarray], np.ndarray]:
    """The floor: no knowledge at all."""
    rng = np.random.default_rng(seed)

    def q(obs: np.ndarray) -> np.ndarray:
        return rng.random(7)

    return q


def centre_column_policy() -> Callable[[np.ndarray], np.ndarray]:
    """A trivial fixed habit — prefer the middle. Worth a line on the board
    because it costs nothing and catches a policy that has learned less than
    one sentence of Connect 4 advice."""
    def q(obs: np.ndarray) -> np.ndarray:
        return np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])

    return q


REFERENCE_PLAYERS = {
    "random": ("a player with no knowledge at all: the floor",
               lambda: random_as_policy(0)),
    "centre-habit": ("prefer the middle column, nothing else — one sentence "
                     "of Connect 4 advice and no search",
                     centre_column_policy),
    "solver-depth-1": ("exhaustive search one ply deep: takes every immediate "
                       "win, blocks every immediate threat, then judges by "
                       "threat count and centre control",
                       lambda: solver_as_policy(1)),
    "solver-depth-4": ("exhaustive search four plies deep",
                       lambda: solver_as_policy(4)),
    "solver-depth-8": ("exhaustive search eight plies deep",
                       lambda: solver_as_policy(8)),
    "solver-depth-10": ("exhaustive search ten plies deep — the strongest "
                        "reference measured, and far stronger than anything "
                        "this project has trained",
                        lambda: solver_as_policy(10)),
}


def measure_reference_board(benchmark, players: Optional[List[str]] = None,
                            version: int = 1, verbose: bool = True) -> ReferenceBoard:
    """Score each reference player on the benchmark. Minutes, not seconds —
    run once and store the result."""
    names = players or list(REFERENCE_PLAYERS)
    board = ReferenceBoard(benchmark_ref=benchmark.ref, version=version)
    for name in names:
        if name not in REFERENCE_PLAYERS:
            raise ValueError(f"unknown reference player {name!r}; "
                             f"known: {sorted(REFERENCE_PLAYERS)}")
        description, build = REFERENCE_PLAYERS[name]
        start = time.time()
        result = benchmark.score(build())
        elapsed = time.time() - start
        board.scores.append(ReferenceScore(
            name=name, description=description, score=result.score,
            per_rung=result.detail.get("per_rung", {}), seconds=elapsed))
        if verbose:
            print(f"  {name:16s} {result.score:.4f}  ({elapsed:.0f}s)")
    return board


def render(board: ReferenceBoard) -> str:
    """The board as a table, for reading in a terminal."""
    lines = [f"REFERENCE PLAYERS ON {board.benchmark_ref}",
             "=" * 76,
             "A score is only meaningful against players whose strength is known by",
             "construction. 1.0 is NOT the practical ceiling on this benchmark: both",
             "sides are forced through fixed openings and the policy plays both seats,",
             "so some games start from objectively lost positions.",
             ""]
    rungs = sorted({r for s in board.scores for r in s.per_rung})
    lines.append(f"  {'player':18s} {'score':>7}  " +
                 " ".join(f"r{r:>4}" for r in rungs))
    for score in sorted(board.scores, key=lambda s: -s.score):
        cells = " ".join(f"{score.per_rung.get(r, float('nan')):>5.2f}" for r in rungs)
        lines.append(f"  {score.name:18s} {score.score:>7.4f}  {cells}")
    lines.append("")
    for score in sorted(board.scores, key=lambda s: -s.score):
        lines.append(f"  {score.name}: {score.description}")
    best = board.best()
    if best:
        near_mastery = [r for r, v in best.per_rung.items() if v >= 0.85]
        lines += ["",
                  f"Read every policy score against {best.score:.3f} — the best "
                  f"measured here — and not against 1.000.",
                  "",
                  "The ceiling is a measurement, not a deduction: it moves as "
                  "stronger references are added.",
                  f"Mastery is defined as 0.90 at EVERY rung. The strongest "
                  f"reference reaches 0.85 or better at "
                  f"{len(near_mastery)} of {len(best.per_rung)} rungs, so that bar "
                  f"is demanding but not",
                  "obviously impossible. Until a reference clears it, treat a "
                  "frontier depth of 1 as uninformative."]
    return "\n".join(lines)
