# adaptive-intelligence

An experimental framework for an automated policy-improvement loop: train, evaluate
against fixed benchmarks, select, archive, repeat — then upgrade the loop's components
(curriculum, plasticity, architecture) one at a time.

Central hypothesis: **a single network that masters multiple related tasks becomes
increasingly sample-efficient at learning the next one.**

## Orientation

| Doc | What it is |
|---|---|
| [`docs/vision.md`](docs/vision.md) | The full framework: five foundational components, critiques, verdicts, research references |
| [`docs/roadmap.md`](docs/roadmap.md) | Phased implementation plan with definitions of done |
| [`STATE.md`](STATE.md) | Current position and next task — updated every session |
| [`.kiro/steering/core.md`](.kiro/steering/core.md) | Agent steering: principles and session protocol |
| [`docs/design/decisions/`](docs/design/decisions/) | Architecture decision records |

## Status

Phase 0 (loop harness) — not started. Seed material comes from the sibling
[connect4-rl](../connect4-rl) project: a working champion-challenger self-play loop,
league tournament player, and experiment registry, to be generalized behind
domain-agnostic interfaces.
