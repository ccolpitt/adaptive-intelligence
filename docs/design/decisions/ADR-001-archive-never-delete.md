# ADR-001: Archive, Never Delete

**Status:** Accepted (2026-08-26)

## Decision

Policies, experiment configs, and evaluation results are append-only. "End of life"
for a policy means it stops receiving training/evaluation compute — it is never erased.

## Rationale

- Strategy spaces are non-transitive (rock-paper-scissors dynamics). A "dead" policy
  may hold the exact skill that defeats a future champion. AlphaStar's league keeps
  past agents as exploiters for this reason (Vinyals et al., 2019).
- The archive is the fossil record: it enables retroactive scoring when new benchmarks
  arrive (see roadmap Phase 1 — scoring connect4-rl's v0→v32 lineage), lineage
  analysis, and regression detection.
- Disk is cheap; a lost lineage is unrecoverable.

## Consequences

- The `Archive` interface has no delete operation.
- Policy metadata must include lineage (parent policy, training config, eval results)
  so the fossil record is interpretable.
