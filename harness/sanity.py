"""Does this result make sense?

Every cycle produces numbers. Reading them one at a time answers "is it
improving?" but not the more useful question: **is anything here impossible,
or inconsistent with something else here?** A score can rise while a
measurement that the score depends on is self-contradictory, and nothing in
the loop notices.

So each check below states an expectation, says why it must hold, and names
the severity if it does not:

  contradiction — cannot be true given how the system is built. Always a
                  defect in the harness or the learner, never a weak policy.
                  Four of the last five defects in this project were of
                  exactly this kind and were found by eye, late.
  inconsistent  — two measurements disagree in a way that needs explaining
                  before the cycle's result is trusted.
  unexpected    — allowed, but not what the design predicts; worth a look.

The output is written into the cycle record and printed. The standing rule
(docs/learning-rules.md, rule 11) is that a contradiction is investigated
before any further conclusion is drawn from the run, because a number that
cannot be true means something is measuring the wrong thing and every other
number in that cycle is suspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

CONTRADICTION = "contradiction"
INCONSISTENT = "inconsistent"
UNEXPECTED = "unexpected"

SEVERITY_ORDER = {UNEXPECTED: 1, INCONSISTENT: 2, CONTRADICTION: 3}


@dataclass
class Finding:
    severity: str
    check: str
    expected: str       # what should have been true, and why
    observed: str       # what was measured instead
    look_at: List[str] = field(default_factory=list)  # where to investigate

    def to_dict(self) -> Dict:
        return {"severity": self.severity, "check": self.check,
                "expected": self.expected, "observed": self.observed,
                "look_at": self.look_at}

    def __str__(self) -> str:
        return (f"[{self.severity.upper()}] {self.check}\n"
                f"    expected: {self.expected}\n"
                f"    observed: {self.observed}"
                + (f"\n    look at:  {', '.join(self.look_at)}" if self.look_at else ""))


def _num(value) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------


def check_cycle(record, history: Optional[List] = None) -> List[Finding]:
    """Compare one cycle's measurements against what the design requires."""
    history = history or []
    out: List[Finding] = []
    probes = record.probes or {}
    evaluation = record.evaluation or {}
    series = record.series or {}

    # -- things that cannot be true --------------------------------------

    budget = record.search_budget
    spent = [v for v in series.get("search_states", []) if v is not None]
    if budget > 0 and spent:
        worst = max(spent)
        if worst > budget * 1.05:  # a little slack for per-game accumulation
            out.append(Finding(
                CONTRADICTION, "lookahead stayed inside its budget",
                f"at most {budget} distinct positions examined per move, because "
                f"that is the declared budget and the budget is the compute "
                f"currency that makes two policies comparable",
                f"a game averaged {worst:.0f} positions per move",
                ["budgeted_search.BudgetedSearch.run", "the per-move accounting"],
            ))

    value_max = _num(probes.get("q_max_abs"))
    bounded = "move_score_max_abs" in probes  # two-headed: value head is squashed
    if value_max is not None and bounded and value_max > 1.001:
        out.append(Finding(
            CONTRADICTION, "value estimates respect the reward range",
            "no position can be worth more than winning, and the value head is "
            "squashed to enforce that, so the largest estimate cannot exceed 1",
            f"largest value estimate was {value_max:.3f}",
            ["policy_value.PolicyValueNet.forward", "whether the probe read the "
             "value head or the move head"],
        ))

    for layer, stats in (probes.get("weight_distribution") or {}).items():
        if stats.get("nonfinite", 0):
            out.append(Finding(
                CONTRADICTION, "weights are finite",
                "a finite loss and a clipped gradient cannot produce a "
                "non-finite weight",
                f"{layer} holds {stats['nonfinite']} non-finite weights",
                ["the gradient clipping", "the loss for an overflow"],
            ))

    losses = [v for v in series.get("loss", []) if v is not None]
    if losses and not all(np.isfinite(losses)):
        out.append(Finding(
            CONTRADICTION, "training error is a finite number",
            "a bounded target and a bounded prediction give a bounded error",
            "the error was not finite in at least one game",
            ["the training targets", "whether targets are bounded"],
        ))

    # -- things that disagree with each other ----------------------------

    bare = _num(evaluation.get("score"))
    with_search = _num(evaluation.get("score_with_lookahead"))
    if bare is not None and with_search is not None and with_search < bare - 0.03:
        out.append(Finding(
            INCONSISTENT, "thinking does not make the policy worse",
            "searching over a policy evaluates real consequences before "
            "committing, so it should score at least as well as the same "
            "policy answering immediately",
            f"scored {bare:.3f} alone but {with_search:.3f} with "
            f"{evaluation.get('eval_budget')} positions of lookahead",
            ["whether the search selects by visits or by value",
             "whether the value head's sign convention matches the search",
             "budgeted_search._backup"],
        ))

    take_win = _num(probes.get("take_the_win_rate"))
    if budget >= 32 and take_win is not None and take_win < 0.5:
        out.append(Finding(
            INCONSISTENT, "a policy trained with lookahead takes an immediate win",
            f"a search of {budget} positions always sees a win available one "
            f"move away, and the policy is trained toward what the search "
            f"concluded, so missing half of them means what the search found "
            f"is not reaching the weights",
            f"took an available win {take_win:.0%} of the time",
            ["whether the training target is what the search concluded",
             "whether the move actually played is the move trained toward",
             "the size of the search budget relative to the number of moves"],
        ))

    proven_win = _num(probes.get("q_at_solver_proven_wins"))
    proven_loss = _num(probes.get("q_at_solver_proven_losses_mean"))
    if proven_win is not None and proven_loss is not None and proven_win <= proven_loss:
        out.append(Finding(
            INCONSISTENT, "won positions are rated above lost positions",
            "whatever the scale, a position a perfect player proves is won "
            "must be rated higher than one proved lost — this is the weakest "
            "possible statement of a value function working at all",
            f"proven-won rated {proven_win:.2f}, proven-lost {proven_loss:.2f}",
            ["the sign convention in the value target",
             "whether the outcome is recorded from the mover's point of view"],
        ))

    if record.promoted and history:
        previous = [_num(r.evaluation.get("score")) for r in history if r.promoted]
        previous = [p for p in previous if p is not None]
        if previous and bare is not None and bare < previous[-1] - 1e-9:
            out.append(Finding(
                INCONSISTENT, "a promotion does not lower the champion's score",
                "promotion requires beating the champion AND improving on the "
                "task's own measure, so the champion's score cannot fall",
                f"promoted at {bare:.3f} when the previous champion held "
                f"{previous[-1]:.3f}",
                ["promotion_gate.statistical_strict_gate",
                 "which score the gate was given"],
            ))

    states = series.get("unique_states", [])
    if len(states) > 10 and states[-1] <= states[0]:
        out.append(Finding(
            INCONSISTENT, "new positions keep being seen",
            "self-play with any randomness reaches positions it has not seen "
            "before, so the running count must rise across a cycle",
            f"the count went from {states[0]} to {states[-1]}",
            ["whether exploration is still happening",
             "whether the counter is being reset"],
        ))

    if history:
        previous_params = {r.parameter_count for r in history if r.parameter_count}
        if previous_params and record.parameter_count not in previous_params:
            out.append(Finding(
                INCONSISTENT, "the network does not change shape mid-lineage",
                "a lineage trains one network, so its size is fixed; a change "
                "means two different networks are being compared as if they "
                "were one",
                f"this cycle has {record.parameter_count:,} weights, earlier "
                f"cycles had {sorted(previous_params)}",
                ["the rubric for this lineage", "whether a new lineage was meant"],
            ))

    # -- allowed, but not what the design predicts ------------------------

    wr_random = _num(evaluation.get("win_rate_vs_random"))
    if wr_random is not None and wr_random < 0.5 and record.cumulative_games > 2000:
        out.append(Finding(
            UNEXPECTED, "a trained policy beats a random player",
            "after a couple of thousand games, losing to random play is a "
            "floor failure rather than a difficulty with the task",
            f"won {wr_random:.0%} against random play after "
            f"{record.cumulative_games:,} games",
            ["whether the learner is training at all", "the loss curve"],
        ))

    if wr_random is not None and history:
        earlier = [_num(r.evaluation.get("win_rate_vs_random")) for r in history[-5:]]
        earlier = [v for v in earlier if v is not None]
        if earlier and max(earlier) - wr_random > 0.25:
            out.append(Finding(
                UNEXPECTED, "the floor does not fall back",
                "competence against random play, once reached, should persist",
                f"was as high as {max(earlier):.0%} recently, now {wr_random:.0%}",
                ["whether something is being forgotten",
                 "whether exploration was raised"],
            ))

    if len(losses) > 40:
        first = float(np.mean(losses[: len(losses) // 5]))
        last = float(np.mean(losses[-len(losses) // 5:]))
        if first > 0 and last > first * 2.0:
            out.append(Finding(
                UNEXPECTED, "training error does not grow within a cycle",
                "the target moves as the policy improves, so some rise is "
                "normal, but a doubling suggests the target is outrunning the "
                "network",
                f"error went from {first:.4f} to {last:.4f} across the cycle",
                ["the learning rate", "how often the comparison network is "
                 "refreshed"],
            ))

    deepest = record.search_deepest_ply
    if budget >= 64 and 0 < deepest <= 2:
        out.append(Finding(
            UNEXPECTED, "a large budget looks further than a small one",
            f"{budget} positions should follow the deciding line well beyond "
            f"two moves; a shallow reach means the budget is being spread "
            f"instead of concentrated",
            f"the deepest the search reached was {deepest} moves",
            ["the exploration constant", "whether the move head gives useful "
             "priors yet"],
        ))

    return sorted(out, key=lambda f: -SEVERITY_ORDER[f.severity])


def render(findings: List[Finding]) -> str:
    if not findings:
        return "SANITY: every expectation held."
    worst = max(SEVERITY_ORDER[f.severity] for f in findings)
    header = ("SANITY: A CONTRADICTION WAS FOUND — investigate before drawing "
              "any conclusion from this cycle." if worst == SEVERITY_ORDER[CONTRADICTION]
              else "SANITY: results need explaining.")
    return header + "\n" + "\n".join(str(f) for f in findings)


def worst_severity(findings: List[Finding]) -> Optional[str]:
    if not findings:
        return None
    return max(findings, key=lambda f: SEVERITY_ORDER[f.severity]).severity


# ---------------------------------------------------------------------------
# reviewing the checks themselves
# ---------------------------------------------------------------------------
#
# A set of checks written against past defects will drift out of usefulness.
# Some checks will have fired once, been fixed structurally, and can never
# fire again — they cost attention and suggest a vigilance that is no longer
# needed. Others will never have fired because the situation they describe
# has not arisen yet, which is not the same as being useless. And defects
# keep arriving that nothing checks for.
#
# So the check set is reviewed rather than left to accumulate. This produces
# the evidence for that review; a human decides what to cut, because deleting
# a check is a decision about what we are willing to stop watching.


CHECK_REGISTRY = {
    "lookahead stayed inside its budget": {
        "motivated_by": "a budget of 32 positions that examined 400 per move",
        "retired_if": "the budget is enforced inside the search itself AND a "
                      "test asserts it, so overspending cannot compile",
        "still_needed_because": "the accounting is spread across the search, "
                                "the trainer and the record, and any of the "
                                "three can drift",
    },
    "value estimates respect the reward range": {
        "motivated_by": "R-001: a rubric asked for bounded values and the "
                        "setting never reached the trainer (reached 2.428)",
        "retired_if": "every learner squashes its value output and no "
                      "unbounded learner remains in use",
        "still_needed_because": "single-output learners are still selectable "
                                "for reproducing earlier experiments",
    },
    "weights are finite": {
        "motivated_by": "general defence; has never fired",
        "retired_if": "never — it is nearly free and the failure is total",
        "still_needed_because": "a silent overflow would invalidate a run "
                                "while every score still looked plausible",
    },
    "training error is a finite number": {
        "motivated_by": "general defence; has never fired",
        "retired_if": "never — nearly free",
        "still_needed_because": "same reason as non-finite weights",
    },
    "thinking does not make the policy worse": {
        "motivated_by": "R-004: move selection by value rather than by visit "
                        "count, which made searching actively harmful at "
                        "small budgets",
        "retired_if": "search is removed from evaluation entirely",
        "still_needed_because": "it detects a whole class of sign and "
                                "convention errors in the search at once",
    },
    "a policy trained with lookahead takes an immediate win": {
        "motivated_by": "four lineages scoring adequately while missing half "
                        "of all forced blocks",
        "retired_if": "tactical rates exceed 0.95 for twenty consecutive "
                      "cycles across two lineages",
        "still_needed_because": "it is currently the live question: tactics "
                                "lag the score in the active lineage",
    },
    "won positions are rated above lost positions": {
        "motivated_by": "the weakest possible statement of a value function "
                        "working; has never fired",
        "retired_if": "never — it is the floor of value learning",
        "still_needed_because": "it would catch a reversed sign convention "
                                "immediately, which is an easy mistake to "
                                "make when outcomes are recorded per mover",
    },
    "a promotion does not lower the champion's score": {
        "motivated_by": "two promotions each individually forgivable took the "
                        "champion from 0.083 to 0.042",
        "retired_if": "the strict gate is the only gate and the looser ones "
                      "are deleted",
        "still_needed_because": "the looser gates remain selectable, and the "
                                "failure compounds silently",
    },
    "new positions keep being seen": {
        "motivated_by": "general defence; has never fired",
        "retired_if": "a cheaper exploration measure replaces it",
        "still_needed_because": "a collapse in exploration would otherwise "
                                "look like ordinary slow progress",
    },
    "the network does not change shape mid-lineage": {
        "motivated_by": "general defence; has never fired",
        "retired_if": "roadmap phase 8a arrives, where growing a network "
                      "mid-lineage becomes deliberate — then this check must "
                      "be rewritten to compare against the rubric rather than "
                      "against earlier cycles",
        "still_needed_because": "until then a size change means two different "
                                "networks are being compared as one",
    },
    "a trained policy beats a random player": {
        "motivated_by": "the floor every competent policy should clear",
        "retired_if": "the floor is cleared by every lineage within one cycle",
        "still_needed_because": "it separates 'the task is hard' from 'nothing "
                                "is learning'",
    },
    "the floor does not fall back": {
        "motivated_by": "forgetting, which is a central concern of the "
                        "project and has not yet been observed here",
        "retired_if": "never — it is the earliest warning of forgetting, "
                      "which the whole programme is about",
        "still_needed_because": "multi-task training has not started; this is "
                                "the check that will matter most when it does",
    },
    "training error does not grow within a cycle": {
        "motivated_by": "general defence; has never fired",
        "retired_if": "it proves noisy enough to be ignored in practice",
        "still_needed_because": "a target outrunning the network is a real "
                                "failure mode of bootstrapped learning",
    },
    "a large budget looks further than a small one": {
        "motivated_by": "the observation that a search degenerates to "
                        "breadth-first when its guidance is uninformative",
        "retired_if": "the search is replaced by one whose shape is fixed",
        "still_needed_because": "it detects a move head that has stopped "
                                "providing useful guidance, which looks like "
                                "nothing else in the instruments",
    },
}


def review_checks(records: List, retros_path: str = "docs/defect-retros.md") -> Dict:
    """Evidence for deciding what the check set should contain.

    Reports, for every check: how often it has fired, when it last fired,
    the defect that motivated it, and the condition under which it should be
    retired. Also lists defects recorded in the retros that no check covers,
    which is the more important half — a defect that happened once and is
    unwatched will happen again.
    """
    import re
    from pathlib import Path

    fired: Dict[str, List[int]] = {}
    for record in records:
        for finding in (record.sanity or []):
            fired.setdefault(finding["check"], []).append(record.cycle)

    known = set(CHECK_REGISTRY)
    rows = []
    for check, meta in sorted(CHECK_REGISTRY.items()):
        cycles = fired.get(check, [])
        rows.append({
            "check": check,
            "times_fired": len(cycles),
            "last_fired_cycle": cycles[-1] if cycles else None,
            "motivated_by": meta["motivated_by"],
            "retired_if": meta["retired_if"],
            "still_needed_because": meta["still_needed_because"],
        })

    # Checks that fired but are not registered: someone added a check and did
    # not record why, which is how a set becomes unreviewable.
    unregistered = sorted(set(fired) - known)

    # Defects on record that no check mentions. Weak matching on purpose —
    # it is better to raise a few false candidates than to miss one.
    uncovered = []
    path = Path(retros_path)
    if path.exists():
        text = path.read_text()
        for match in re.finditer(r"^## (R-\d+) — (.+)$", text, re.MULTILINE):
            ident, title = match.group(1), match.group(2)
            words = {w.lower() for w in re.findall(r"[a-z]{5,}", title.lower())}
            covered = any(
                words & {w.lower() for w in re.findall(r"[a-z]{5,}", blob.lower())}
                for meta in CHECK_REGISTRY.values()
                for blob in (meta["motivated_by"],)
            )
            if not covered:
                uncovered.append({"defect": ident, "title": title})

    return {
        "checks": rows,
        "never_fired": [r["check"] for r in rows if r["times_fired"] == 0],
        "fired": [r["check"] for r in rows if r["times_fired"] > 0],
        "unregistered_checks": unregistered,
        "defects_without_a_check": uncovered,
        "cycles_reviewed": len(records),
    }


def render_review(review: Dict) -> str:
    lines = [
        "REVIEW OF THE EXPECTATION CHECKS",
        "=" * 76,
        f"{review['cycles_reviewed']} cycles examined. A check that has never "
        f"fired is not useless — it may describe a situation that has not",
        "arisen yet. The question for each is the retirement condition, not the "
        "firing count.",
        "",
        f"{'fired':>5}  {'last':>5}  check",
    ]
    for row in review["checks"]:
        lines.append(f"{row['times_fired']:>5}  "
                     f"{str(row['last_fired_cycle'] or '—'):>5}  {row['check']}")
    lines += ["", "RETIREMENT CONDITIONS (a human decides; deleting a check is a",
              "decision about what we stop watching)"]
    for row in review["checks"]:
        lines.append(f"  {row['check']}")
        lines.append(f"      retire if: {row['retired_if']}")
    if review["unregistered_checks"]:
        lines += ["", "CHECKS THAT FIRED BUT ARE NOT REGISTERED — record why each "
                  "exists, or the set becomes unreviewable:"]
        lines += [f"  - {c}" for c in review["unregistered_checks"]]
    if review["defects_without_a_check"]:
        lines += ["", "DEFECTS ON RECORD THAT NO CHECK WATCHES FOR — the more "
                  "important half of this review:"]
        for d in review["defects_without_a_check"]:
            lines.append(f"  - {d['defect']}: {d['title']}")
    else:
        lines += ["", "Every recorded defect has a check that would catch its class."]
    return "\n".join(lines)
