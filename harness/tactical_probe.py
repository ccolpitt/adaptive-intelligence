"""One-move tactics, split by where the positions came from and by geometry.

The cycle records carry two tactical numbers — how often the policy takes an
available win, and how often it blocks a forced loss — and they were a single
figure each, mixing together three different questions. That mixing nearly
sent an investigation after the value head when the fault was elsewhere. This
module separates them.

**Two distributions, reported separately.** A policy is trained on positions
it generates itself, and is *scored* against solver opponents whose positions
look nothing like self-play. Those are different questions and they get
different answers:

    in-distribution   positions from the policy's own play, with search,
                      exactly as training produces them
    off-distribution  positions produced by random play, which the policy
                      never trains on and which resemble the varied
                      positions the benchmark's forced openings create

**Four geometries, reported separately.** A line of four can be vertical,
horizontal, or either diagonal, and these are not equally easy to see. A
vertical block is the easiest tactic in the game: the column to play *is* the
column holding the threat, so the whole pattern lives in one column. A
diagonal block is the hardest: the policy must notice a line spread across
four columns and then work out which column's landing row intersects it,
which depends on how full the neighbouring columns are.

Measured on the champion of lineage 5 (2026-10-26), blocking off-distribution:

    vertical    0.95
    diagonal/   0.68
    horizontal  0.59
    diagonal\\   0.45

That spread is the subject. Taking a win shows no comparable geometric
effect — it is poor at roughly 0.5 in every direction off-distribution and
around 0.7 in-distribution — so the two failures have different causes and
should not be treated as one weakness.

**Attribution is unambiguous or the position is skipped.** A position with
two winning columns, or one winning column that completes two lines at once,
cannot be attributed to a single geometry, so it is not counted. That costs
samples and buys a number that means what it says.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .tasks.connect4 import Connect4Env

# (row step, column step) -> name. A line of four in each of the four ways it
# can be made; the two diagonals are named by the direction they rise.
DIRECTIONS: Dict[Tuple[int, int], str] = {
    (0, 1): "horizontal",
    (1, 0): "vertical",
    (1, 1): "diagonal\\",
    (1, -1): "diagonal/",
}

GEOMETRIES = ("vertical", "horizontal", "diagonal\\", "diagonal/")
DISTRIBUTIONS = ("in-distribution", "off-distribution")


def lines_completed_at(env: Connect4Env, row: int, col: int, player: int) -> set:
    """Which kinds of line a piece at (row, col) would complete for player."""
    found = set()
    for (dr, dc), name in DIRECTIONS.items():
        count = 1
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while (0 <= r < env.rows and 0 <= c < env.cols
                   and env.board[r, c] == player):
                count += 1
                r += sign * dr
                c += sign * dc
        if count >= env.connect_n:
            found.add(name)
    return found


def geometry_of_winning_move(env: Connect4Env, player: int,
                             col: int) -> Optional[str]:
    """The single kind of line that playing ``col`` completes for ``player``,
    or None if it completes none or more than one. Returning None for the
    ambiguous case is deliberate: a position that cannot be attributed to one
    geometry must not be counted under any of them."""
    empties = np.where(env.board[:, col] == 0)[0]
    if empties.size == 0:
        return None
    row = int(np.max(empties))
    env.board[row, col] = player
    try:
        found = lines_completed_at(env, row, col, player)
    finally:
        env.board[row, col] = 0  # the probe never mutates the position
    return found.pop() if len(found) == 1 else None


@dataclass
class TacticalBreakdown:
    """Hit rates by geometry, for one distribution of positions."""

    distribution: str
    took: Counter = field(default_factory=Counter)
    took_of: Counter = field(default_factory=Counter)
    blocked: Counter = field(default_factory=Counter)
    blocked_of: Counter = field(default_factory=Counter)

    def take_rate(self, geometry: Optional[str] = None) -> Optional[float]:
        return self._rate(self.took, self.took_of, geometry)

    def block_rate(self, geometry: Optional[str] = None) -> Optional[float]:
        return self._rate(self.blocked, self.blocked_of, geometry)

    @staticmethod
    def _rate(hits: Counter, totals: Counter,
              geometry: Optional[str]) -> Optional[float]:
        if geometry is None:
            n = sum(totals.values())
            return sum(hits.values()) / n if n else None
        return hits[geometry] / totals[geometry] if totals[geometry] else None

    def worst_block(self) -> Optional[Tuple[str, float]]:
        """The geometry the policy blocks least reliably, with its rate."""
        rated = [(g, self.block_rate(g)) for g in GEOMETRIES
                 if self.blocked_of[g] >= 20]
        return min(rated, key=lambda p: p[1]) if rated else None

    def to_dict(self) -> Dict:
        return {
            "distribution": self.distribution,
            "take_the_win": {g: self.take_rate(g) for g in GEOMETRIES},
            "take_the_win_samples": {g: self.took_of[g] for g in GEOMETRIES},
            "take_the_win_overall": self.take_rate(),
            "block_the_loss": {g: self.block_rate(g) for g in GEOMETRIES},
            "block_the_loss_samples": {g: self.blocked_of[g] for g in GEOMETRIES},
            "block_the_loss_overall": self.block_rate(),
        }


def _observe(env: Connect4Env, move_scores: Callable, breakdown: TacticalBreakdown) -> None:
    """Score the policy on this position, if it is unambiguously tactical.

    Always asked from the MOVER's side, using the canonical observation — that
    is, "would you take this win if it were your move". Positions where a win
    and a required block coexist are counted only as a win, since taking the
    win ends the game and the block is moot.
    """
    mover = env.current_player
    legal = env.legal_actions()
    if not legal:
        return
    scores = np.asarray(move_scores(env.obs()), dtype=np.float64)
    masked = np.full(env.cols, -np.inf)
    masked[legal] = scores[legal]
    chosen = int(np.argmax(masked))

    wins = env.winning_moves(mover)
    if len(wins) == 1:
        geometry = geometry_of_winning_move(env, mover, wins[0])
        if geometry is not None:
            breakdown.took_of[geometry] += 1
            breakdown.took[geometry] += int(chosen == wins[0])
        return
    if wins:
        return  # more than one win: cannot attribute to one geometry

    threats = env.winning_moves(-mover)
    if len(threats) == 1:
        geometry = geometry_of_winning_move(env, -mover, threats[0])
        if geometry is not None:
            breakdown.blocked_of[geometry] += 1
            breakdown.blocked[geometry] += int(chosen == threats[0])


def probe_off_distribution(move_scores: Callable, games: int = 400,
                           seed: int = 0) -> TacticalBreakdown:
    """Positions produced by random play: varied, and nothing like self-play.

    The policy is scored at every position but never steers the game, so the
    positions stay off its own distribution — which is the point. These
    resemble what the benchmark's forced openings produce.
    """
    breakdown = TacticalBreakdown("off-distribution")
    rng = np.random.default_rng(seed)
    for _ in range(games):
        env = Connect4Env()
        env.reset()
        while not env.done:
            _observe(env, move_scores, breakdown)
            env.step(int(rng.choice(env.legal_actions())))
    return breakdown


def probe_in_distribution(move_scores: Callable, search, games: int = 60,
                          sampling_moves: int = 12,
                          seed: int = 0) -> TacticalBreakdown:
    """Positions from the policy's own play, driven by its search exactly as
    training produces them. ``search`` is a BudgetedSearch over this policy."""
    breakdown = TacticalBreakdown("in-distribution")
    rng = np.random.default_rng(seed)
    for _ in range(games):
        env = Connect4Env()
        env.reset()
        moves = 0
        while not env.done:
            _observe(env, move_scores, breakdown)
            result = search.run(env.obs())
            action = result.best_action_by_visits
            if moves < sampling_moves:
                policy = result.visit_policy(temperature=1.0)
                if policy.sum() > 0:
                    candidate = int(rng.choice(len(policy), p=policy))
                    if candidate in env.legal_actions():
                        action = candidate
            env.step(action)
            moves += 1
    return breakdown


def probe_policy(module, games_off: int = 400, games_in: int = 60,
                 budget: int = 200, seed: int = 0) -> List[TacticalBreakdown]:
    """Both distributions for an archived policy, single- or two-headed."""
    from .interactive_play import has_two_heads

    two_heads = has_two_heads(module)
    if two_heads:
        from .agents.policy_value import move_scores_from, policy_value_from
        from .budgeted_search import (
            BudgetedSearch, PerfectConnect4Model, PolicyValueEvaluator, SearchConfig,
        )

        move_scores = move_scores_from(module)
        search = BudgetedSearch(
            PolicyValueEvaluator(policy_value_from(module), 7),
            PerfectConnect4Model(),
            SearchConfig(node_budget=budget, dirichlet_noise=0.25,
                         select_by="visits", seed=seed + 1))
    else:
        import torch

        from .budgeted_search import make_search

        def move_scores(obs):
            with torch.no_grad():
                return module(torch.from_numpy(obs).float().unsqueeze(0)
                              ).squeeze(0).numpy()

        search = make_search(move_scores, budget=budget, spend_rule="puct",
                             dirichlet_noise=0.25, select_by="visits",
                             seed=seed + 1)

    return [probe_in_distribution(move_scores, search, games=games_in, seed=seed),
            probe_off_distribution(move_scores, games=games_off, seed=seed)]


def render(breakdowns: List[TacticalBreakdown], label: str = "") -> str:
    lines = [f"ONE-MOVE TACTICS BY GEOMETRY{(' — ' + label) if label else ''}",
             "=" * 76,
             "Measured on the policy alone, no search. A line of four can be made four",
             "ways and they are not equally easy to see: a vertical block is the easiest",
             "tactic in the game, because the column to play IS the column holding the",
             "threat. A diagonal block is the hardest -- the line spans four columns and",
             "which column blocks it depends on how full the neighbours are.",
             "",
             "Positions with two winning columns, or one that completes two lines, are",
             "skipped: they cannot be attributed to a single geometry."]
    for breakdown in breakdowns:
        lines += ["", f"{breakdown.distribution}:",
                  f"  {'geometry':14s} {'takes win':>10s} {'n':>6s}   "
                  f"{'blocks loss':>12s} {'n':>6s}"]
        for geometry in GEOMETRIES:
            take = breakdown.take_rate(geometry)
            block = breakdown.block_rate(geometry)
            lines.append(
                f"  {geometry:14s} "
                f"{(f'{take:.2f}' if take is not None else '—'):>10s} "
                f"{breakdown.took_of[geometry]:>6d}   "
                f"{(f'{block:.2f}' if block is not None else '—'):>12s} "
                f"{breakdown.blocked_of[geometry]:>6d}")
        take, block = breakdown.take_rate(), breakdown.block_rate()
        lines.append(
            f"  {'ALL':14s} {(f'{take:.2f}' if take is not None else '—'):>10s} "
            f"{sum(breakdown.took_of.values()):>6d}   "
            f"{(f'{block:.2f}' if block is not None else '—'):>12s} "
            f"{sum(breakdown.blocked_of.values()):>6d}")
        worst = breakdown.worst_block()
        if worst:
            lines.append(f"  weakest block: {worst[0]} at {worst[1]:.2f}")
    return "\n".join(lines)
