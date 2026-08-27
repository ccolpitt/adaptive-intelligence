# ADR-002: Resource-Competition Ecology Is Parked

**Status:** Accepted (2026-08-26) — parked, not rejected

## Context

The original framework proposed a simulated ecology: multiple agents in a shared
environment competing for limited resources under a conservation law, with death as an
emergent outcome of losing that competition.

## Decision

Do not build the ecology. Selection pressure and turnover — the actual evolutionary
function of death — are delivered by the league + archive in the loop harness:
policies compete in tournaments; losers stop receiving compute; everything is archived
(ADR-001). Same function, roughly two orders of magnitude less engineering.

## Rationale

- The ecology is a large simulation-engineering project (resource dynamics,
  interaction protocols, conservation bookkeeping) that must be built and debugged
  before it teaches us anything about learning.
- Known failure mode: evolution in a hand-built ecology optimizes quirks of the
  ecology — physics bugs, degenerate equilibria — rather than intelligence. See
  Lehman et al. (2020), "The Surprising Creativity of Digital Evolution."

## Revisit Trigger

Reopen this decision if and when the policy population shows diversity collapse
(convergence to one strategy) that league mechanisms — exploiter agents, fitness
sharing, diverse opponent sampling — demonstrably cannot fix. A shared-environment
ecology with niches is a known remedy for diversity collapse, and that would be the
moment its cost is justified.
