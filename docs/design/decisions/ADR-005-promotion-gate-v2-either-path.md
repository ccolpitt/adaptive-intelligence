# ADR-005: Promotion Gate v2 — Either-Path Promotion

**Status:** Accepted (2026-09-19) — human-directed

## Context

The statistical double gate required head-to-head superiority (Wilson LB > 0.5)
AND no significant benchmark regression. exp-012 iteration 11 exposed its blind
spot: a cold-started challenger scored 0.116 on the solver ladder — above the
champion's 0.083 — but was blocked because it lost the head-to-head match
(0.41). The double gate let the *relative* match veto *absolute* progress: the
exact inverse of the v32 failure the gate was built to prevent. Principle 1
says the fixed external benchmark is the ground truth; a merely-relative
signal should not be able to overrule it.

## Decision (human's rule, verbatim intent)

"Promote if we either beat the champion or improve on the absolute, and don't
regress on the other test."

`statistical_either_gate` implements it with the existing noise-aware
machinery:

- **Path A (relative):** h2h Wilson LB > 0.5 AND benchmark not significantly
  worse (unchanged from the double gate).
- **Path B (absolute):** benchmark significantly better (one-sided
  two-proportion test) AND h2h not significantly BELOW parity (Wilson upper
  bound ≥ 0.5 — true parity passes; a proven losing matchup does not).

Default for new runs (`LoopConfig.gate_rule = "either"`); the double gate
remains available as `gate_rule = "double"` for reproducing pre-ADR-005
campaigns. The rule used is recorded in every experiment's conditions.

## Consequence for the motivating case

exp-012 iter-11 itself would STILL be rejected under v2: its h2h 0.41 over 200
games has upper bound ≈ 0.47 — significantly below parity, a proven
regression, not a stylistic difference. And its benchmark margin (+0.033 at
n=168) is below significance. The gate change matters for the *future* case
the campaign is trying to produce: a challenger with a significant absolute
gain and a non-losing head-to-head. It also sharpens follow-up design: if
absolute progress keeps arriving with proven h2h regressions, that is a
finding about non-transitivity worth its own experiment, not a gate bug.

## Also in this change (same session, human-directed)

- **Plateau detection** (`stats.plateaued`, LoopConfig `plateau_patience` /
  `plateau_min_delta` / `plateau_action` = flag|stop|off, default flag):
  answers "would more episodes help?" from the run's own benchmark curve;
  "stop" mode reclaims the remaining compute and records `plateau_stop` in
  telemetry.
- **Tactical replay bias** (`TrainerConfig.tactical_fraction`, default 0.0):
  guarantees a batch quota of decision-critical transitions (mover had a
  win-on-next-move, or faced a must-block) — the human's proposal; distinct
  from `terminal_fraction` (one ply earlier: the choice points where games
  are decided). Off by default; to be flipped inside a registered experiment.
