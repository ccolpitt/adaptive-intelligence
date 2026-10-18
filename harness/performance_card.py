"""Policy performance cards — the diagnostic instrument panel for a policy.

Win/loss is one number; a hypothesis needs to know WHY. A card bundles the
diagnostics the connect4-rl precursor tracked on its training dashboard —
value calibration (are Q estimates inside the ±1 world they live in?),
terminal calibration (does the net know a winning move when it takes one?),
tactical accuracy (take the win, block the loss), value-propagation depth
(Q-decay by moves-from-end), exploration coverage, and network health
(dead / exploding / non-finite weights) — plus a pure read of the
experiment's stored training telemetry (loss trend, dormant neurons,
benchmark trajectory). Without this instrumentation, hypotheses are guesses
in the dark.

Design rules:

- Cards are POST-HOC. Building one probes an archived policy and reads
  stored telemetry; it never trains and never mutates the store beyond
  writing the card artifact itself.
- Generic core, task-specific extras. Every probe in the core works for any
  two-player alternating Task behind the harness interfaces. Tactical
  probes activate automatically when the task's env exposes
  ``winning_moves`` (the Connect-N family). Other task families register
  their own probes in ``TASK_PROBES`` keyed by task_id — the card system
  never needs to know the game.
- Every reading carries an EXPECTATION (what healthy looks like, inherited
  from the connect4-rl dashboard annotations) and a STATUS in
  {ok, warn, fail, info} so a card can be scanned in seconds. Status bands
  are v0 heuristics, named in ``Bands`` below — tune them as evidence
  accumulates, one registered change at a time.
- Diagnostic ladder (roadmap Phase 5): today a human reads cards to form
  the next hypothesis; next an agent reads them; eventually the policy
  generates and analyzes its own cards. The card format is therefore
  structured JSON first, rendered text second.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .interfaces import QFunction, Task

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"


class Bands:
    """v0 healthy-range bands. Sources: connect4-rl dashboard expectations
    ("|Q| should trend toward [0,1]", "terminal |Q| should trend to 1",
    "dead/exploding fractions should stay constant") plus the reward design
    (win +1 / loss -1 / draw 0, gamma < 1 => true Q in [-1, 1])."""

    Q_MAX_ABS_OK = 1.5       # |Q| beyond the reward scale but tolerable
    Q_MAX_ABS_WARN = 3.0     # beyond this: value estimates are broken
    Q_MEAN_ABS_OK = 1.0
    Q_MEAN_ABS_WARN = 2.0
    WIN_MOVE_Q_OK = 0.6      # Q of a move that won the game -> +1
    WIN_MOVE_Q_WARN = 0.0
    TAKE_WIN_OK = 0.95       # fraction of immediate wins actually taken
    TAKE_WIN_WARN = 0.70
    BLOCK_OK = 0.90          # fraction of must-block positions blocked
    BLOCK_WARN = 0.60
    NON_BLOCK_Q_OK = -0.30   # Q of NOT blocking a loss -> -1
    NON_BLOCK_Q_WARN = 0.10
    ENDGAME_Q_OK = 0.6       # |max Q| at the final pre-terminal state -> 1
    ENDGAME_Q_WARN = 0.2
    DEAD_FRAC_OK = 0.30      # fraction of |w| < 0.01
    DEAD_FRAC_WARN = 0.60
    EXPLODING_FRAC_OK = 1e-4  # fraction of |w| > 10
    EXPLODING_FRAC_WARN = 1e-2
    LOSS_RATIO_OK = 0.7      # last-fifth mean / first-fifth mean
    LOSS_RATIO_WARN = 1.1
    DEAD_WEIGHT_THRESHOLD = 0.01
    EXPLODING_WEIGHT_THRESHOLD = 10.0


@dataclass
class Reading:
    """One diagnostic measurement with its healthy expectation."""

    name: str
    status: str  # ok | warn | fail | info
    expectation: str
    value: Optional[float] = None
    note: str = ""
    detail: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "status": self.status,
            "value": self.value,
            "expectation": self.expectation,
            "note": self.note,
            "detail": self.detail,
        }


@dataclass
class PerformanceCard:
    policy_id: str
    task_ref: str
    generated_at: str
    probe_config: Dict
    sections: Dict[str, List[Reading]]

    # -- summary --------------------------------------------------------

    def worst_status(self) -> str:
        order = {INFO: 0, OK: 1, WARN: 2, FAIL: 3}
        worst = INFO
        for readings in self.sections.values():
            for r in readings:
                if order[r.status] > order[worst]:
                    worst = r.status
        return worst

    def failures(self) -> List[Reading]:
        return [r for rs in self.sections.values() for r in rs if r.status == FAIL]

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "policy_id": self.policy_id,
            "task_ref": self.task_ref,
            "generated_at": self.generated_at,
            "probe_config": self.probe_config,
            "worst_status": self.worst_status(),
            "sections": {
                name: [r.to_dict() for r in readings]
                for name, readings in self.sections.items()
            },
        }

    def render(self) -> str:
        badge = {OK: "[ ok ]", WARN: "[WARN]", FAIL: "[FAIL]", INFO: "[info]"}
        lines = [
            f"POLICY PERFORMANCE CARD — {self.policy_id} on {self.task_ref}",
            f"generated {self.generated_at[:19]} | worst status: {self.worst_status().upper()}",
            f"probe: {self.probe_config}",
            "=" * 72,
        ]
        for section, readings in self.sections.items():
            lines.append(f"\n{section.upper()}")
            for r in readings:
                val = "" if r.value is None else f" = {r.value:.4g}"
                lines.append(f"  {badge[r.status]} {r.name}{val}")
                lines.append(f"         expect: {r.expectation}")
                if r.note:
                    lines.append(f"         note:   {r.note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# status helpers
# ---------------------------------------------------------------------------


def _band_high(value: float, ok: float, warn: float) -> str:
    """Higher is better."""
    if not math.isfinite(value):
        return FAIL
    if value >= ok:
        return OK
    if value >= warn:
        return WARN
    return FAIL


def _band_low(value: float, ok: float, warn: float) -> str:
    """Lower is better."""
    if not math.isfinite(value):
        return FAIL
    if value <= ok:
        return OK
    if value <= warn:
        return WARN
    return FAIL


# ---------------------------------------------------------------------------
# rollout probe: play games, harvest diagnostics from live positions
# ---------------------------------------------------------------------------


@dataclass
class RolloutHarvest:
    games: int = 0
    policy_score: float = 0.0  # wins + draws/2, informational only
    total_plies: int = 0
    unique_states: int = 0
    q_mean_abs: float = float("nan")   # mean |Q| over legal actions, all states
    q_max_abs: float = float("nan")
    q_nonfinite: int = 0
    win_move_qs: List[float] = field(default_factory=list)
    decay_sums: np.ndarray = field(default_factory=lambda: np.zeros(64))
    decay_counts: np.ndarray = field(default_factory=lambda: np.zeros(64))
    # tactical (only filled when the env exposes winning_moves)
    tactical_available: bool = False
    offense_n: int = 0
    offense_taken: int = 0
    offense_win_q: List[float] = field(default_factory=list)
    defense_n: int = 0
    defense_blocked: int = 0
    defense_nonblock_q: List[float] = field(default_factory=list)


def collect_rollouts(
    task: Task, q_function: QFunction, games: int = 100, seed: int = 0
) -> RolloutHarvest:
    """Play ``games`` vs a seeded random opponent (policy alternates seats)
    and harvest diagnostics from every mover-perspective state encountered.
    Positions are facts about the game: Q is probed on ALL states, whichever
    seat produced them."""
    rng = np.random.default_rng(seed)
    h = RolloutHarvest(games=games)
    seen = set()
    abs_sum, abs_n, max_abs = 0.0, 0, 0.0

    for g in range(games):
        env = task.make_env()
        obs = env.reset()
        policy_is_p1 = g % 2 == 0
        ply = 0
        per_state_maxq: List[float] = []
        tactical = hasattr(env, "winning_moves")
        h.tactical_available = h.tactical_available or tactical

        while True:
            legal = list(env.legal_actions())
            mover = 1 if ply % 2 == 0 else -1
            seen.add(obs.tobytes())

            q = np.asarray(q_function(obs), dtype=np.float64)
            h.q_nonfinite += int(np.sum(~np.isfinite(q[legal])))
            q_legal = q[legal]
            finite = q_legal[np.isfinite(q_legal)]
            if finite.size:
                abs_sum += float(np.abs(finite).sum())
                abs_n += finite.size
                max_abs = max(max_abs, float(np.abs(finite).max()))
            masked = np.full(len(q), -np.inf)
            masked[legal] = np.nan_to_num(q[legal], nan=-np.inf)
            greedy = int(np.argmax(masked))
            per_state_maxq.append(
                float(np.abs(masked[greedy])) if math.isfinite(masked[greedy]) else float("nan")
            )

            # tactical probes on live positions
            if tactical:
                my_wins = env.winning_moves(mover)
                opp_wins = env.winning_moves(-mover)
                if my_wins:
                    h.offense_n += 1
                    h.offense_taken += int(greedy in my_wins)
                    h.offense_win_q.append(float(np.max(q[my_wins])))
                elif opp_wins:
                    non_block = [c for c in legal if c not in opp_wins]
                    h.defense_n += 1
                    h.defense_blocked += int(greedy in opp_wins)
                    if non_block:
                        h.defense_nonblock_q.append(float(np.mean(q[non_block])))

            # choose the actual move
            policy_to_move = (mover == 1) == policy_is_p1
            action = greedy if policy_to_move else int(rng.choice(legal))
            result = env.step(action)
            ply += 1

            if result.done:
                if result.winner is not None and result.winner != 0:
                    # the move just taken won the game: terminal calibration
                    if math.isfinite(q[action]):
                        h.win_move_qs.append(float(q[action]))
                    if (result.winner == 1) == policy_is_p1:
                        h.policy_score += 1.0
                elif result.winner == 0:
                    h.policy_score += 0.5
                break
            obs = result.next_obs

        h.total_plies += ply
        for idx, v in enumerate(reversed(per_state_maxq)):
            if idx < len(h.decay_sums) and math.isfinite(v):
                h.decay_sums[idx] += v
                h.decay_counts[idx] += 1

    h.unique_states = len(seen)
    h.q_mean_abs = abs_sum / abs_n if abs_n else float("nan")
    h.q_max_abs = max_abs
    h.policy_score /= max(games, 1)
    return h


# ---------------------------------------------------------------------------
# section builders
# ---------------------------------------------------------------------------


def value_calibration_readings(h: RolloutHarvest) -> List[Reading]:
    readings = [
        Reading(
            name="q_nonfinite_count",
            value=float(h.q_nonfinite),
            status=OK if h.q_nonfinite == 0 else FAIL,
            expectation="0 — any NaN/inf Q value means training is broken",
        ),
        Reading(
            name="q_mean_abs",
            value=h.q_mean_abs,
            status=_band_low(h.q_mean_abs, Bands.Q_MEAN_ABS_OK, Bands.Q_MEAN_ABS_WARN),
            expectation="within [0, 1] — rewards are ±1 and gamma < 1, so true Q lives in [-1, 1]",
        ),
        Reading(
            name="q_max_abs",
            value=h.q_max_abs,
            status=_band_low(h.q_max_abs, Bands.Q_MAX_ABS_OK, Bands.Q_MAX_ABS_WARN),
            expectation=f"<= {Bands.Q_MAX_ABS_OK} — estimates far outside the reward scale mean divergence",
        ),
    ]
    if h.win_move_qs:
        v = float(np.mean(h.win_move_qs))
        readings.append(
            Reading(
                name="q_of_game_winning_moves",
                value=v,
                status=_band_high(v, Bands.WIN_MOVE_Q_OK, Bands.WIN_MOVE_Q_WARN),
                expectation="-> +1 — the net should KNOW a winning move at the moment it is played",
                detail={"n": len(h.win_move_qs)},
            )
        )
    else:
        readings.append(
            Reading(
                name="q_of_game_winning_moves",
                status=INFO,
                expectation="-> +1",
                note="no decisive games in probe — increase probe games",
            )
        )
    return readings


def tactical_readings(h: RolloutHarvest) -> List[Reading]:
    if not h.tactical_available:
        return [
            Reading(
                name="tactical_probes",
                status=INFO,
                expectation="task env exposes winning_moves (Connect-N family)",
                note="not available for this task family",
            )
        ]
    out: List[Reading] = []
    if h.offense_n:
        rate = h.offense_taken / h.offense_n
        out.append(
            Reading(
                name="take_the_win_rate",
                value=rate,
                status=_band_high(rate, Bands.TAKE_WIN_OK, Bands.TAKE_WIN_WARN),
                expectation="-> 1.0 — an immediate win on the board should always be taken",
                detail={"n": h.offense_n},
            )
        )
        wq = float(np.mean(h.offense_win_q))
        out.append(
            Reading(
                name="q_of_winning_move_available",
                value=wq,
                status=_band_high(wq, Bands.WIN_MOVE_Q_OK, Bands.WIN_MOVE_Q_WARN),
                expectation="-> +1 (connect4-rl offensive threat check)",
                detail={"n": h.offense_n},
            )
        )
    else:
        out.append(Reading(name="take_the_win_rate", status=INFO,
                           expectation="-> 1.0", note="no winnable positions in probe"))
    if h.defense_n:
        rate = h.defense_blocked / h.defense_n
        out.append(
            Reading(
                name="block_the_loss_rate",
                value=rate,
                status=_band_high(rate, Bands.BLOCK_OK, Bands.BLOCK_WARN),
                expectation="-> 1.0 — an opponent's immediate win must be blocked",
                detail={"n": h.defense_n},
            )
        )
        if h.defense_nonblock_q:
            nq = float(np.mean(h.defense_nonblock_q))
            out.append(
                Reading(
                    name="q_of_not_blocking",
                    value=nq,
                    status=_band_low(nq, Bands.NON_BLOCK_Q_OK, Bands.NON_BLOCK_Q_WARN),
                    expectation="-> -1 (connect4-rl defensive threat check)",
                    detail={"n": len(h.defense_nonblock_q)},
                )
            )
    else:
        out.append(Reading(name="block_the_loss_rate", status=INFO,
                           expectation="-> 1.0", note="no must-block positions in probe"))
    return out


def propagation_readings(h: RolloutHarvest) -> List[Reading]:
    """Q-decay by moves-from-end — connect4-rl's most distinctive diagnostic.
    Position 0 (the final pre-terminal state) should approach 1; how far back
    the curve stays elevated shows how deep value has propagated."""
    with np.errstate(divide="ignore", invalid="ignore"):
        curve = np.where(h.decay_counts > 0, h.decay_sums / h.decay_counts, 0.0)
    endgame = float(curve[0]) if h.decay_counts[0] > 0 else float("nan")
    horizon = 0
    for i in range(len(curve)):
        if h.decay_counts[i] > 0 and curve[i] >= 0.5:
            horizon = i
        elif h.decay_counts[i] > 0 and curve[i] < 0.5 and i > horizon + 2:
            break
    n = int(np.max(np.nonzero(h.decay_counts)[0])) + 1 if h.decay_counts.any() else 0
    return [
        Reading(
            name="endgame_q_magnitude",
            value=endgame,
            status=_band_high(endgame, Bands.ENDGAME_Q_OK, Bands.ENDGAME_Q_WARN),
            expectation="-> 1 at the final state; earlier states decay until value propagates back",
        ),
        Reading(
            name="value_horizon_plies",
            value=float(horizon),
            status=INFO,
            expectation="grows with training — how many plies from the end |maxQ| stays >= 0.5",
            detail={"decay_curve": [round(float(c), 4) for c in curve[:n]]},
        ),
    ]


def exploration_readings(h: RolloutHarvest) -> List[Reading]:
    return [
        Reading(
            name="unique_states_in_probe",
            value=float(h.unique_states),
            status=INFO,
            expectation="more is better; compare across cards at equal probe size",
            detail={"games": h.games, "avg_game_plies": round(h.total_plies / max(h.games, 1), 2)},
        ),
        Reading(
            name="probe_score_vs_random",
            value=h.policy_score,
            status=INFO,
            expectation="informational — the versioned benchmark is the authoritative score",
        ),
    ]


def network_health_readings(module: torch.nn.Module) -> List[Reading]:
    weights = [p.detach().float() for p in module.parameters()]
    if not weights:
        return [Reading(name="network_health", status=INFO,
                        expectation="module exposes parameters", note="no parameters found")]
    flat = torch.cat([w.reshape(-1) for w in weights])
    nonfinite = int((~torch.isfinite(flat)).sum().item())
    finite = flat[torch.isfinite(flat)]
    dead = float((finite.abs() < Bands.DEAD_WEIGHT_THRESHOLD).float().mean().item()) if finite.numel() else 1.0
    exploding = float((finite.abs() > Bands.EXPLODING_WEIGHT_THRESHOLD).float().mean().item()) if finite.numel() else 1.0
    return [
        Reading(
            name="nonfinite_weights",
            value=float(nonfinite),
            status=OK if nonfinite == 0 else FAIL,
            expectation="0 — NaN/inf weights mean the optimizer diverged",
        ),
        Reading(
            name="dead_weight_fraction",
            value=dead,
            status=_band_low(dead, Bands.DEAD_FRAC_OK, Bands.DEAD_FRAC_WARN),
            expectation=f"|w| < {Bands.DEAD_WEIGHT_THRESHOLD}: should stay roughly constant over training "
                        "(rising = plasticity loss, the Phase 4 signal)",
        ),
        Reading(
            name="exploding_weight_fraction",
            value=exploding,
            status=_band_low(exploding, Bands.EXPLODING_FRAC_OK, Bands.EXPLODING_FRAC_WARN),
            expectation=f"|w| > {Bands.EXPLODING_WEIGHT_THRESHOLD}: ~0 — growth means instability",
            detail={
                "weight_std": float(finite.std().item()) if finite.numel() > 1 else None,
                "weight_max_abs": float(finite.abs().max().item()) if finite.numel() else None,
                "total_params": int(flat.numel()),
            },
        ),
    ]


def telemetry_readings(telemetry: Dict) -> List[Reading]:
    """Pure read over the experiment telemetry the loop already writes:
    loss trend, benchmark trajectory, dormant-neuron trend."""
    out: List[Reading] = []
    losses = [x for x in telemetry.get("episode_loss_means", []) if x is not None]
    if len(losses) >= 2:
        k = max(1, len(losses) // 5)
        first, last = float(np.mean(losses[:k])), float(np.mean(losses[-k:]))
        ratio = last / first if first > 0 else float("inf")
        out.append(
            Reading(
                name="training_loss_trend",
                value=ratio,
                status=_band_low(ratio, Bands.LOSS_RATIO_OK, Bands.LOSS_RATIO_WARN),
                expectation="< 1 — loss (MSE Bellman error) should decrease over the run",
                detail={"first_fifth_mean": first, "last_fifth_mean": last},
            )
        )
    q_series = [x for x in telemetry.get("episodes", {}).get("q_mean_abs", []) if x is not None]
    if q_series:
        recent = float(np.mean(q_series[-max(1, len(q_series) // 5):]))
        out.append(
            Reading(
                name="training_q_mean_abs_recent",
                value=recent,
                status=_band_low(recent, Bands.Q_MEAN_ABS_OK, Bands.Q_MEAN_ABS_WARN),
                expectation="within [0, 1] throughout training — rewards are ±1, gamma < 1; "
                            "drift above ~1 during training is value miscalibration",
                detail={"max_over_run": float(np.max(q_series))},
            )
        )
    norms = [x for x in telemetry.get("episode_grad_norm_means", []) if x is not None]
    if len(norms) >= 2:
        k = max(1, len(norms) // 5)
        first, last = float(np.mean(norms[:k])), float(np.mean(norms[-k:]))
        ratio = last / first if first > 0 else float("inf")
        # steady or declining is healthy; explosion = instability; collapse
        # toward zero pairs with dead_weight_fraction to flag plasticity loss
        status = OK if ratio <= 3.0 else (WARN if ratio <= 10.0 else FAIL)
        if ratio < 0.01:
            status = WARN
        out.append(
            Reading(
                name="gradient_norm_trend",
                value=ratio,
                status=status,
                expectation="roughly steady or declining — exploding = instability; "
                            "collapsing to ~0 with rising dead weights = plasticity loss",
                detail={"first_fifth_mean": first, "last_fifth_mean": last},
            )
        )
    iterations = telemetry.get("iterations", [])
    bench = [it.get("benchmark_score") for it in iterations if it.get("benchmark_score") is not None]
    if len(bench) >= 2:
        delta = float(bench[-1] - bench[0])
        out.append(
            Reading(
                name="benchmark_trajectory",
                value=delta,
                status=OK if delta > 0 else (WARN if delta == 0 else FAIL),
                expectation="rising across iterations — the loop's whole point",
                detail={"scores": [round(float(b), 4) for b in bench]},
            )
        )
    dorm = [it.get("dormant_neurons") for it in iterations if isinstance(it.get("dormant_neurons"), dict)]
    if len(dorm) >= 2:
        first = float(np.mean(list(dorm[0].values())))
        last = float(np.mean(list(dorm[-1].values())))
        rising = last - first
        out.append(
            Reading(
                name="dormant_neuron_trend",
                value=rising,
                status=OK if rising <= 0.02 else (WARN if rising <= 0.10 else FAIL),
                expectation="flat — a rising dormant fraction is plasticity loss (Dohare et al. 2024)",
                detail={"first_mean": round(first, 4), "last_mean": round(last, 4)},
            )
        )
    if not out:
        out.append(Reading(name="training_telemetry", status=INFO,
                           expectation="telemetry file with loss/benchmark/dormant series",
                           note="no telemetry found for this policy's experiment"))
    return out


def solver_calibration_readings(
    task: Task, q_function: QFunction, games: int = 10, seed: int = 0,
    solver_depth: int = 8, max_positions: int = 50,
) -> List[Reading]:
    """Ground-truth value calibration against the perfect-play solver.

    Verifies that reward is actually back-propagating through the game tree:
    - positions the solver PROVES won for the mover (a forced win exists
      within solver_depth): the policy's best legal Q should approach +1;
    - positions the solver PROVES lost (every move loses within depth):
      the policy's best legal Q should approach -1 — there is no good move,
      and a calibrated value function knows it.
    Positions are harvested from live games vs a seeded random opponent, so
    the distribution matches what the policy actually encounters."""
    try:
        from .tasks.connect4_solver import Connect4Solver, HEURISTIC_CAP
    except ImportError:
        return [Reading(name="solver_calibration", status=INFO,
                        expectation="requires the Connect 4 solver", note="solver unavailable")]
    rng = np.random.default_rng(seed + 1)
    positions = []  # (board copy, mover, obs)
    for g in range(games):
        env = task.make_env()
        if not hasattr(env, "board"):
            return [Reading(name="solver_calibration", status=INFO,
                            expectation="env exposes .board/.current_player",
                            note="not a solver-compatible task family")]
        obs = env.reset()
        policy_is_p1 = g % 2 == 0
        while True:
            positions.append((env.board.copy(), env.current_player, obs.copy()))
            legal = list(env.legal_actions())
            q = np.asarray(q_function(obs), dtype=np.float64)
            mover = env.current_player
            policy_to_move = (mover == 1) == policy_is_p1
            if policy_to_move:
                masked = np.full(len(q), -np.inf)
                masked[legal] = np.nan_to_num(q[legal], nan=-np.inf)
                action = int(np.argmax(masked))
            else:
                action = int(rng.choice(legal))
            result = env.step(action)
            if result.done:
                break
            obs = result.next_obs

    rng.shuffle(positions)
    solver = Connect4Solver(max_depth=solver_depth)
    won_qs, lost_qs = [], []
    evaluated = 0
    for board, mover, obs in positions:
        if evaluated >= max_positions:
            break
        evals = solver.evaluate_moves(board, mover)
        if not evals:
            continue
        evaluated += 1
        q = np.asarray(q_function(obs), dtype=np.float64)
        best_q = float(np.max([q[e.col] for e in evals]))
        if any(e.exact and e.score > HEURISTIC_CAP for e in evals):
            won_qs.append(best_q)  # forced win exists -> best Q should -> +1
        elif all(e.exact and e.score < -HEURISTIC_CAP for e in evals):
            lost_qs.append(best_q)  # every move loses -> best Q should -> -1

    out: List[Reading] = []
    if won_qs:
        v = float(np.mean(won_qs))
        out.append(Reading(
            name="q_at_solver_proven_wins",
            value=v,
            status=_band_high(v, Bands.WIN_MOVE_Q_OK, Bands.WIN_MOVE_Q_WARN),
            expectation="-> +1: the solver proves a forced win exists from here",
            detail={"n": len(won_qs), "solver_depth": solver_depth},
        ))
    if lost_qs:
        v = float(np.mean(lost_qs))
        out.append(Reading(
            name="q_at_solver_proven_losses",
            value=v,
            status=OK if v <= -Bands.WIN_MOVE_Q_OK else (WARN if v <= 0.0 else FAIL),
            expectation="-> -1: the solver proves EVERY move loses from here — "
                        "a calibrated value function knows when it is beaten",
            detail={"n": len(lost_qs), "solver_depth": solver_depth},
        ))
    if not out:
        out.append(Reading(
            name="solver_calibration", status=INFO,
            expectation="proven-won / proven-lost positions in probe sample",
            note=f"no positions provable at depth {solver_depth} in {evaluated} sampled",
        ))
    return out


# Task-specific probe registry: task_id -> callable(task, q_function, games,
# seed) -> {section_name: [Reading]}. The Connect-N tactical probes activate
# automatically via winning_moves; register genuinely bespoke probes here.
TASK_PROBES: Dict[str, Callable] = {}


# ---------------------------------------------------------------------------
# card assembly
# ---------------------------------------------------------------------------


def resolve_task(task_ref: str) -> Task:
    """Build the Task instance an archived policy was trained on."""
    task_id, _, v = task_ref.partition("@v")
    version = int(v) if v else 1
    if task_id == "connect4":
        if version >= 2:
            from .tasks.connect4_ladder import Connect4LadderTask

            return Connect4LadderTask()
        from .tasks.connect4 import Connect4Task

        return Connect4Task()
    raise ValueError(f"no task factory registered for {task_ref}")


def build_card(
    store,
    policy_id: str,
    task: Optional[Task] = None,
    games: int = 100,
    seed: int = 0,
) -> PerformanceCard:
    entry = store.archive.get_entry(policy_id)
    if entry is None:
        raise ValueError(f"unknown policy: {policy_id}")
    task = task or resolve_task(entry["task_ref"])
    module, meta = store.archive.load_policy(policy_id)

    def q_function(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(obs).float().unsqueeze(0)
            return module(t).squeeze(0).numpy()

    sections: Dict[str, List[Reading]] = {}

    # identity & provenance (pure read)
    identity = [
        Reading(name="generation", value=float(entry.get("generation", 0)), status=INFO,
                expectation="lineage depth from gen-0 seed"),
        Reading(name="parent", status=INFO, expectation="lineage link",
                note=str(entry.get("parent_id"))),
        Reading(name="experiment", status=INFO, expectation="the hypothesis that produced this policy",
                note=str(entry.get("experiment_id"))),
    ]
    epitaph = store.archive.epitaph(policy_id)
    if epitaph:
        identity.append(Reading(name="epitaph", status=INFO,
                                expectation="why it lived or died", note=epitaph))
    sections["identity"] = identity

    # benchmark history (pure read)
    bench_readings = []
    for rec in meta.get("eval_records", []):
        bench_readings.append(
            Reading(
                name=f"score vs {rec.get('opponent', rec['benchmark_ref'])}",
                value=float(rec["score"]),
                status=INFO,
                expectation="authoritative, versioned score",
                detail=rec.get("detail", {}),
            )
        )
    sections["benchmark"] = bench_readings or [
        Reading(name="benchmark", status=WARN, expectation="every policy carries eval records",
                note="no eval records in metadata")
    ]

    # live probes
    harvest = collect_rollouts(task, q_function, games=games, seed=seed)
    sections["value_calibration"] = value_calibration_readings(harvest)
    sections["tactics"] = tactical_readings(harvest)
    sections["value_propagation"] = propagation_readings(harvest)
    sections["exploration"] = exploration_readings(harvest)
    sections["network_health"] = network_health_readings(module)
    sections["solver_calibration"] = solver_calibration_readings(
        task, q_function, games=max(4, games // 10), seed=seed
    )

    # training telemetry (pure read)
    telemetry: Dict = {}
    exp_id = entry.get("experiment_id")
    if exp_id:
        path = Path(store.telemetry_dir) / f"{exp_id}.json"
        if path.exists():
            telemetry = json.loads(path.read_text())
    sections["training_telemetry"] = telemetry_readings(telemetry)

    # bespoke task probes, if registered
    extra = TASK_PROBES.get(task.task_id)
    if extra is not None:
        for name, readings in extra(task, q_function, games, seed).items():
            sections[name] = readings

    return PerformanceCard(
        policy_id=policy_id,
        task_ref=entry["task_ref"],
        generated_at=datetime.now(timezone.utc).isoformat(),
        probe_config={"games": games, "seed": seed, "opponent": "seeded-random"},
        sections=sections,
    )


def save_card(store, card: PerformanceCard) -> Dict[str, str]:
    """Write the card as JSON (for machines) and text (for humans) under
    store/cards/. Cards are derived artifacts — regenerable, so overwriting
    with a fresh probe is fine and does not violate append-only invariants."""
    cards_dir = Path(store.root) / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    json_path = cards_dir / f"{card.policy_id}.json"
    txt_path = cards_dir / f"{card.policy_id}.txt"
    json_path.write_text(json.dumps(card.to_dict(), indent=2))
    txt_path.write_text(card.render() + "\n")
    return {"json": str(json_path), "txt": str(txt_path)}
