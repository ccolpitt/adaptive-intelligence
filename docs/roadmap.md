# Roadmap

Organizing principle (proven empirically in connect4-rl: every clever component change
lost to the plain baseline; only loop-level variables mattered):

> Close the loop with the dumbest possible components first. Then upgrade one
> component at a time, measured against the dumb baseline.

## Phase 0 — Loop Harness

Port and generalize the connect4-rl improvement loop into domain-agnostic interfaces.

- [ ] `Task` interface: environment + reward + fixed-benchmark slot
- [ ] `Policy` artifact: serialized model + embedded metadata (lineage, training
      config, eval results) — extend the TorchScript `_extra_files` pattern
- [ ] `Archive`: append-only policy store (the fossil record). League + archive
      IS the end-of-life mechanism — no ecology.
- [ ] `Evaluator`: runs policy vs opponent(s), records results to the registry
- [ ] `Scheduler`: heuristic task/training selection (dumb on purpose)
- [ ] Experiment registry: hypothesis + config + results, append-only
      (formalize connect4-rl's `experiments.json`)
- [ ] Connect 4 ported as seed task #1; champion-challenger loop runs end-to-end
      inside the harness

**Done when:** one command runs the full loop on Connect 4 and produces an archived,
metadata-rich policy lineage plus an experiment record.

## Phase 1 — Measurement Foundation

- [ ] Minimax/perfect-play solver for Connect 4 (absolute yardstick; game is solved)
- [ ] Benchmark protocol: score vs solver from both sides at fixed depths
- [ ] Transfer-metric protocol: episodes-to-threshold, ± pretraining, fixed seeds,
      matched compute
- [ ] Retroactively score the archived connect4-rl champions (v0→v32) on the
      absolute scale — first real data point on what the old loop achieved

**Done when:** any archived policy can be given an absolute strength score, and the
transfer protocol is written down and unit-tested.

## Phase 2 — Transfer Hypothesis Test

- [ ] Task family: Connect-3, board-size variants, tic-tac-toe, etc., all behind the
      `Task` interface
- [ ] Single network, multiple tasks (I/O adapters as needed)
- [ ] Measure: does mastery of task A accelerate task B? Compounding across 3+ tasks?

**Done when:** we have a falsifiable answer with sample-efficiency curves. Either
outcome is a result.

## Phase 3 — Pedagogical Evolution

- [ ] Auto-curriculum over the task family: sample tasks by learning progress
      (PLR-style / Oudeyer-style)
- [ ] Compare vs uniform and hand-ordered curricula under matched compute

## Phase 4 — Continual Plasticity

- [ ] Sequential-task benchmark: measure catastrophic forgetting explicitly
- [ ] Plasticity-loss tracking (dormant-neuron stats); neuron reinit (extend the
      connect4-rl prototype)
- [ ] Gradient-guided structural growth (RigL-style), vs static baseline

## Phase 5 — Architectural Evolution

- [ ] Constrained architecture grammar (depth, width, blocks, skips)
- [ ] Small population + league selection; PBT-style warm starting
- [ ] Only architectures that beat the fixed benchmark AND the incumbent survive

## Not Phases (running throughout)

- League + archive selection pressure (from Phase 0)
- Experiment registry discipline
- STATE.md updates every session

## Parked (see docs/vision.md §5 for revisit triggers)

- Resource-competition ecology
- Learned structural-update policy
- World models / imagination / planning
- Physical embodiment
