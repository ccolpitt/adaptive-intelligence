# ADR-003: World Models Promoted from Parked to Roadmap

**Status:** Accepted (2026-08-26) — supersedes the "world models parked" entry in
vision.md Revision 2

## Context

Revision 2 parked world models with the trigger "Phase 2 shows transfer exists but
saturates at shallow feature reuse." The human overrode the trigger directly: world
models are now a core working hypothesis (hypothesis (a) — necessary for deep
transfer and sample efficiency), and requiring one is a stated goal of the environment
progression.

## Decision

Add a learned-dynamics track to Phase 2: the agent is NOT given the game rules; it
must learn a transition model from observed transitions and plan against that learned
model (MuZero-style — Schrittwieser et al., 2020).

## Why board games first

Board games give a uniquely honest testbed for learned world models: the true dynamics
exist (the rules), so world-model error is directly measurable — compare planning
against the learned model vs planning against the true rules, position by position.
In open-world domains that diagnostic is unavailable.

## Consequences

- The `Task` interface must support withholding the simulator from the agent while
  keeping it available to the Evaluator.
- World-model quality becomes a first-class diagnostic (feeds the Phase 0 telemetry
  and the Component 1 tenets).
- The environment ladder (vision.md §3) sequences increasing world-model demands:
  Tier 2 (Crafter) is where learned models must earn their keep without a true-model
  crutch.
