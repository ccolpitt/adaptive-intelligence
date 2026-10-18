"""Card-driven diagnosis — the hypothesis generator reads the instruments.

ADR-007 (human-directed): after every experiment, read the best challenger's
performance card and the run's gate outcomes, and turn every failure into a
concrete hypothesis candidate. The human's framing: "if the policy shows
|Q| > 1, or Q(winning move) != 1, or Q(fail to block) != -1, these are
failures we must incorporate into our hypotheses — if we don't fix them we
won't train a good policy."

This is rung 2 of the diagnostic ladder (roadmap Phase 0): a RULE TABLE, not
intelligence. Each rule maps an instrument reading to (what it means, what
to try). Rules are deliberately explicit so a human can audit, add, or
delete them; the policy itself generating these is Phase 5/6 work.

Output is appended to the run's telemetry (``telemetry["diagnosis"]``),
printed at run end, shown on the experiment card, and available via
``python -m harness diagnose --policy <id>``. It never edits
docs/hypothesis-queue.md — that file is the human's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .performance_card import FAIL, WARN, PerformanceCard, Reading


@dataclass
class Suggestion:
    severity: str  # "fail" | "warn" | "signal"
    finding: str  # what the instrument showed
    meaning: str  # what it implies about training
    hypotheses: List[str]  # concrete levers to try, most specific first
    evidence: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "severity": self.severity,
            "finding": self.finding,
            "meaning": self.meaning,
            "hypotheses": self.hypotheses,
            "evidence": self.evidence,
        }


def _get(card: PerformanceCard, name: str) -> Optional[Reading]:
    for readings in card.sections.values():
        for r in readings:
            if r.name == name:
                return r
    return None


def diagnose_card(card: PerformanceCard) -> List[Suggestion]:
    out: List[Suggestion] = []

    def bad(r: Optional[Reading]) -> bool:
        return r is not None and r.status in (FAIL, WARN)

    # -- value scale: |Q| must live in [0, 1] -------------------------------
    for name in ("q_max_abs", "q_mean_abs", "training_q_mean_abs_recent"):
        r = _get(card, name)
        if bad(r):
            out.append(Suggestion(
                severity=r.status,
                finding=f"{name} = {r.value:.3g} (expect within the ±1 reward scale)",
                meaning="Value estimates have left the reward scale: the Bellman "
                        "recursion is inflating (target-net lag, lr too high, or "
                        "shaping rewards stacking beyond ±1).",
                hypotheses=[
                    "target_update_every: sync the target net more often (50 -> 20)",
                    "lr 1e-3 -> 3e-4 (halve the step that overshoots)",
                    "if tactical_shaping: cap shaping so |reward| <= 1 per transition",
                    "clip Q targets to [-1, 1] at training time (bounded-value lever)",
                ],
                evidence={"value": r.value},
            ))
            break  # one scale suggestion is enough

    # -- terminal calibration: knows a win when it plays one ----------------
    r = _get(card, "q_of_game_winning_moves")
    if bad(r):
        out.append(Suggestion(
            severity=r.status,
            finding=f"Q of game-winning moves = {r.value:.2f} (expect -> +1)",
            meaning="The reward at game end is not being learned at the move that "
                    "earns it — terminal transitions are under-sampled or over-"
                    "written by bootstrapped noise.",
            hypotheses=[
                "terminal_fraction 0 -> 0.3 (guarantee endgame transitions per batch)",
                "verify loser rewrite / done flags on the card's eval_records",
            ],
            evidence={"value": r.value},
        ))

    # -- tactics: take the win, block the loss --------------------------------
    take = _get(card, "take_the_win_rate")
    block = _get(card, "block_the_loss_rate")
    notblock_q = _get(card, "q_of_not_blocking")
    win_q = _get(card, "q_of_winning_move_available")
    if bad(take) or bad(block):
        out.append(Suggestion(
            severity=FAIL if (take and take.status == FAIL) or (block and block.status == FAIL) else WARN,
            finding=(f"take_the_win {take.value:.2f}" if take and take.value is not None else "")
                    + (f", block_the_loss {block.value:.2f}" if block and block.value is not None else "")
                    + " (expect -> 1.0)",
            meaning="One-ply tactics are not reliable in one-glance play. The net "
                    "sees these positions too rarely, or its values at them are "
                    "muddled by unrelated outcomes.",
            hypotheses=[
                "tactical_fraction 0 -> 0.3 (guaranteed quota of win-available / "
                "must-block positions per batch)",
                "train_search_depth 0 -> 2 with search_targets (distill the "
                "tactic into the weights — exp-022 is testing this)",
                "tactical_shaping on if off; if on and still failing, the issue "
                "is data exposure not reward",
            ],
            evidence={k: v.value for k, v in (("take", take), ("block", block),
                                               ("q_not_block", notblock_q), ("q_win", win_q))
                      if v is not None},
        ))
    elif bad(notblock_q) or bad(win_q):
        out.append(Suggestion(
            severity=WARN,
            finding="tactical moves chosen correctly but their Q values are "
                    "mis-scaled (Q(win) should -> +1, Q(not blocking) -> -1)",
            meaning="Right actions for weakly-held reasons: the ordering is learned, "
                    "the magnitude is not. Fragile under distribution shift.",
            hypotheses=["terminal_fraction / tactical_fraction quotas sharpen magnitudes",
                        "search_targets provide exact ±1 labels at these positions"],
            evidence={k: v.value for k, v in (("q_not_block", notblock_q), ("q_win", win_q))
                      if v is not None},
        ))

    # -- solver ground truth ------------------------------------------------
    for name, target in (("q_at_solver_proven_wins", "+1"), ("q_at_solver_proven_losses", "-1")):
        r = _get(card, name)
        if bad(r):
            out.append(Suggestion(
                severity=r.status,
                finding=f"{name} = {r.value:.2f} (solver proves outcome; expect -> {target})",
                meaning="Against perfect-play ground truth the value function is "
                        "miscalibrated at provably decided positions.",
                hypotheses=["train_search_depth >= 2 (search finds the forced line; "
                            "distill it)", "more decided positions in the buffer "
                            "(terminal_fraction / tactical_fraction)"],
                evidence={"value": r.value},
            ))

    # -- propagation depth ------------------------------------------------
    r = _get(card, "value_horizon_plies")
    if r is not None and r.value is not None and r.value <= 1:
        out.append(Suggestion(
            severity="signal",
            finding=f"value_horizon_plies = {r.value:.0f}: outcome barely propagates "
                    "past the final move",
            meaning="Credit reaches the endgame but not the decisions that led "
                    "there. MC labeling (exp-019) over-corrected; the middle ground "
                    "is a short n-step target or search-backed targets.",
            hypotheses=["n-step TD (n=3) — untested middle ground between TD(0) and MC",
                        "search_targets (exp-022 Arm B)"],
            evidence={"horizon": r.value},
        ))

    # -- training dynamics (telemetry section) ------------------------------
    r = _get(card, "training_loss_trend")
    if bad(r):
        out.append(Suggestion(
            severity=r.status,
            finding=f"training loss ratio (last/first fifth) = {r.value:.2f} (expect < 1)",
            meaning="Loss is not decreasing: the regression target is moving faster "
                    "than the net can track it, or the signal is noise.",
            hypotheses=["lr 1e-3 -> 3e-4", "target_update_every 50 -> 20",
                        "if mc_targets / shaping on: the labels themselves may be too noisy"],
            evidence={"ratio": r.value},
        ))
    r = _get(card, "gradient_norm_trend")
    if bad(r):
        out.append(Suggestion(
            severity=r.status,
            finding=f"gradient norm trend ratio = {r.value:.2f}",
            meaning="Exploding = instability; collapsing with rising dead weights = "
                    "plasticity loss.",
            hypotheses=["lower lr / tighter grad clip (explosion)",
                        "continual-backprop style reinit of low-utility units (collapse)"],
            evidence={"ratio": r.value},
        ))
    r = _get(card, "dormant_neuron_trend")
    if bad(r):
        out.append(Suggestion(
            severity=r.status,
            finding=f"dormant-neuron fraction rose by {r.value:.3f} over the run",
            meaning="Capacity is going quiet — plasticity loss (Phase 4 signal).",
            hypotheses=["periodic reinit of lowest-utility neurons (Dohare et al.)",
                        "shorter runs / cold start per experiment (already the recipe)"],
            evidence={"delta": r.value},
        ))
    return out


def diagnose_run(card: PerformanceCard, telemetry: Dict) -> List[Suggestion]:
    """Card diagnosis plus run-level signals from the gate outcomes."""
    out = diagnose_card(card)
    iterations = telemetry.get("iterations", [])
    parity_iters = [it["iteration"] for it in iterations if it.get("parity")]
    if parity_iters and not any(it.get("promoted") for it in iterations):
        bench = [it.get("benchmark_score") or 0.0 for it in iterations]
        rising = len(bench) >= 4 and bench[-1] > bench[len(bench) // 2]
        out.insert(0, Suggestion(
            severity="signal",
            finding=f"PARITY with the champion at iteration(s) {parity_iters}, no promotion",
            meaning="The lever got the challenger to even with the incumbent. That is "
                    "evidence the lever is sound and under-dosed, not evidence it failed."
                    + (" Benchmark was still rising at the end." if rising else ""),
            hypotheses=["same lever, 2x the training budget (iterations or episodes)",
                        "same lever + the top card-driven fix below"],
            evidence={"parity_iterations": parity_iters, "bench_rising": rising},
        ))
    if telemetry.get("plateau_stop"):
        out.append(Suggestion(
            severity="signal",
            finding=f"plateau-stopped at iteration {telemetry['plateau_stop']['iteration']}",
            meaning="More episodes of THIS configuration were not helping.",
            hypotheses=["change the lever, not the budget"],
            evidence=telemetry["plateau_stop"],
        ))
    return out


def render_diagnosis(suggestions: List[Suggestion]) -> str:
    if not suggestions:
        return "DIAGNOSIS: no failing instruments — card is clean."
    lines = ["DIAGNOSIS (card-driven hypothesis candidates):"]
    for s in suggestions:
        lines.append(f"  [{s.severity.upper():6s}] {s.finding}")
        lines.append(f"           means: {s.meaning}")
        for h in s.hypotheses:
            lines.append(f"           try:   {h}")
    return "\n".join(lines)
