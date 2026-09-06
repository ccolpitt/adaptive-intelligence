# ADR-004: Singular Agent — Common Sensory Input Format as the Endgame

**Status:** Accepted (2026-08-31) — human direction

## Context

The Phase 2 transfer design splits the policy into per-task input adapter → shared
trunk → per-task output head. This is the cheapest way to run the transfer science,
but it quietly gives every task its own retina. Humans do not work that way: one
sensory stream feeds every competence (walking, talking, piano), and what
accumulates with mastery is the wiring that assigns contextual and semantic meaning
to the same inputs a baby receives. Left unstated, the adapter pattern could ossify
into the architecture, capping transfer at the strategy level forever — a 6×7
symbolic matrix can never carry anything perceptual toward Atari, throwing, or
talking.

The human set the aspiration explicitly: a policy that is close to input-agnostic,
where the sign of progress is a single input interface supporting more and more
tasks, and where pre-learning about the world (objects, falling pieces, sounds,
invariance to piece color/lighting/viewpoint) makes new visually-presented games
learnable with few samples.

## Decision

1. **Working hypothesis (g), "shared sensory grounding," added to vision.md §3:**
   the deepest transfer requires a shared input interface, not just a shared trunk.
   Falsifiable like the others.
2. **Phase 2's adapters are explicitly designated scaffolding** (Stage A/B), with a
   named Stage C — one shared sensory interface — that replaces them from the
   outside in. Nothing in Stage A/B forecloses Stage C: a pixel rendering is just
   another `obs_spec`.
3. **New roadmap Phase 7 — Sensory Grounding:** (7a) render mastered tasks to a
   common versioned pixel format while keeping the same absolute benchmarks;
   (7b) perceptual pretraining / developmental curriculum — objects, persistence,
   intuitive physics from raw interaction, i.e. the ADR-003 world model moved down
   to the sensory floor; (7c) measure perceptual vs strategic transfer separately,
   which is possible because the same game can be presented through symbolic and
   pixel adapters into the same trunk.
4. **Registered prediction placed in Phase 2:** within-family symbolic transfer
   > 0; symbolic→pixel perceptual transfer ≈ 0; strategic transfer may survive the
   input swap. Phase 7 scores it.

## Why not do this first

Pixels cost one to two orders of magnitude more compute per episode. The core
transfer/forgetting/plasticity science (Phases 2–4) is cheapest on symbolic inputs,
and its results (does transfer compound at all? where does it saturate?) are
prerequisites for knowing whether input-level sharing is the binding constraint.
Promotion trigger: Phase 2 verdict in AND (transfer saturates at shallow feature
reuse OR the ladder adds a natively-pixel task).

## Consequences

- The renderer becomes a versioned task component (`connect4-pixels@v1`); changing
  rendering = new task version. Benchmarks are unchanged — the solver still grades
  the same underlying game, so measurement stays absolute (Component 0 intact).
- The two-player perspective-flipped `Environment` protocol must generalize when
  the first single-player pixel task (Atari) arrives — known, contained refactor.
- The environment ladder's Tier 1+ entries become the natural suppliers of
  natively-pixel tasks.
- The curriculum gains a developmental floor: perception and intuitive physics
  before games — "the tree of expertise" grows from shared roots, and the
  curriculum-design work of Phase 3 applies to it.
