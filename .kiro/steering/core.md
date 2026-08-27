# Core Steering — Adaptive Intelligence Project

## Vision (one paragraph)

Build a system that closes an automated improvement loop over learning policies:
train, evaluate against fixed benchmarks, select, archive, repeat — then upgrade the
loop's components one at a time (curriculum, plasticity, architecture) to test the
central hypothesis: **a single network that masters multiple related tasks becomes
increasingly sample-efficient at learning the next one.** Full framework:
`docs/vision.md`. Current position and next task: `STATE.md`.

## Non-Negotiable Principles

1. **Every task has a fixed, external benchmark.** Relative metrics (Elo, champion
   win-rate) are never sufficient evidence of progress. If a task has no absolute
   yardstick yet, building one precedes everything else on that task.
2. **Close the loop before improving components.** No component upgrade without a
   dumb-baseline run to compare against. Boring-but-complete beats clever-but-partial.
3. **One variable at a time.** Every experiment is registered (hypothesis, config,
   changes) in the experiment log BEFORE it runs. Negative results are recorded with
   the same care as positive ones.
4. **Archive, never delete.** Policies, configs, and results are append-only. "Death"
   of a policy means it stops receiving compute, never that it is erased.
5. **Update STATE.md before ending any session.** A session that changed anything and
   didn't update STATE.md is incomplete. This is how the next session resumes.
6. **Respect the deferred list** (`docs/vision.md` §5). Do not build the ecology or
   the learned structural-update policy unless their revisit trigger has fired and the
   human has agreed. (World models were promoted 2026-08-26 — ADR-003.)
7. **Architectural changes follow the tenets** (`docs/vision.md` Component 1):
   diagnosed not divined; scales with compute; names its enrichment; makes a
   falsifiable prediction; one change at a time. Hand-designed architecture
   experiments are welcome in any phase; undirected search is not.

## Relationship to connect4-rl

The sibling repo `~/Development/github/connect4-rl` is **read-only reference
material**: read its loop, league, and experiment-registry code when porting ideas,
but NEVER modify, commit to, or create files in it. All new work lands in this
repository only.

## How to Resume a Session

1. Read `STATE.md` — current phase, last results, next task.
2. Run the test suite — passing tests define what works; failures define what's broken.
3. Check `docs/roadmap.md` for the current phase's definition of done.
4. Continue from STATE.md's "next task." When done, update STATE.md.

## Document Map

| Doc | Purpose | Update frequency |
|---|---|---|
| `.kiro/steering/core.md` (this file) | Vision + principles | Rarely — stable for months |
| `STATE.md` | Current position, next task | Every session |
| `docs/roadmap.md` | Phases, definitions of done | At phase boundaries |
| `docs/vision.md` | Full framework, critiques, references | Rarely |
| `docs/design/decisions/` | ADRs — decisions with rationale and revisit triggers | When decisions are made |
