# ADR-007: Strict Promotion Gate + Card-Driven Hypothesis Generation

**Status:** Accepted (2026-10-03) — human-directed

## Context

Gate v2 (ADR-005) allowed promotion on either axis. It produced one promotion
(pol-00148) that was head-to-head stronger but nominally LOWER on the absolute
ladder (0.071 vs 0.083, "within noise"). The human judged that too loose:
"a challenger should not be promoted unless it beats the existing champion
AND improves on the absolute scale." Separately: the loop produced rich
instruments (performance cards: |Q| scale, Q at winning / not-blocking moves,
solver-proven calibration, horizon, loss/grad trends) that no process was
reading — hypotheses were still being chosen by hand.

## Decisions

1. **Gate v3, "strict"** (`statistical_strict_gate`, default): promote only
   if h2h Wilson LB > 0.5 AND benchmark score strictly above the champion's.
   Benchmark significance is REPORTED, not required (at 168 games it would
   need ~+0.09 and freeze promotion). Prior gates remain selectable for
   reproducing earlier campaigns; the rule is recorded per experiment.
2. **Parity is a signal, not a failure.** A rejection at parity (h2h CI
   straddles 0.5, benchmark ≥ champion's) is flagged on the gate decision and
   in telemetry. Interpretation: the lever is probably sound and under-dosed
   — the first hypothesis candidate is "same lever, 2x budget."
3. **The hypothesis generator reads the card** (`harness/diagnosis.py`). After
   every run, the best challenger's performance card is generated and a rule
   table maps each failing instrument to (meaning, concrete levers): |Q| > 1
   → target-net sync / lr / clipping; Q(win) ≠ 1 → terminal quota;
   take/block failures → tactical_fraction / search_targets; horizon ≤ 1 →
   n-step or search targets; loss/grad/dormant trends → their fixes.
   Written to telemetry (`diagnosis`), printed at run end, available via
   `python -m harness diagnose --policy <id>`. The human's framing: "if we
   don't fix these things, we won't train a good policy."
4. **Scope of the generator**: rung 2 of the diagnostic ladder — an explicit,
   auditable rule table, never editing `docs/hypothesis-queue.md` (the
   human's file). Policy-generated hypotheses are Phase 5/6.

## Consequences

- Fewer promotions; each one is unambiguous on both axes. Champion lineage
  becomes monotone on the ladder (fixes the ADR-005 ratchet risk).
- Every experiment ends with a machine-readable list of what to try next,
  grounded in instruments rather than intuition.
- exp-022 (running) was launched under gate v2 — its conditions record that;
  all later runs default to v3.
