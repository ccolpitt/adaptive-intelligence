# STATE — updated every session

**Last updated:** 2026-08-31 (Phase 0 loop harness closed end-to-end; module files
renamed to be self-describing; roadmap extended with Phase 6 self-directed goals /
self-generated benchmarks and an explicit "prioritization lives in the policy" note)
**Current phase:** Phase 0 — essentially complete; Phase 1 (measurement foundation)
is next

## Where things stand

- **The loop is closed.** `python -m harness run ...` runs
  train → evaluate → select → archive → repeat on Connect 4 and produced an
  archived, metadata-rich lineage plus a registered experiment
  (`exp-001-close-the-loop`, verdict: supported — 1 promotion via double gate,
  benchmark-v0 0.725 → 0.785 over 5×300 episodes).
- **Harness package** (`harness/`) — module files renamed this session so each name
  says what it holds:
  - `task_specs.py` (was specs.py) + `interfaces.py`: `Task`/`Benchmark`/`Opponent`
    contracts with `obs_spec`/`action_spec`
  - `tasks/connect4.py`: Connect 4 seed task, 2-channel perspective-flipped env
  - `agents/dqn.py`: DQN trainer with explicit loss attribution (connect4-rl
    post-hoc buffer patch is gone)
  - `policy_archive.py` (was archive.py): append-only policy store, TorchScript
    `_extra_files` JSON metadata, event-derived champion pointer, epitaphs
  - `task_registry.py` + `experiment_registry.py` (was one registry.py): split
    apart — one holds versioned task definitions, the other holds first-class
    experiments (hypothesis-before-run, one-immutable-verdict, prior-art
    requirement, supersession, bidirectional policy↔experiment links)
  - `promotion_gate.py` (was selection.py): double-gate promotion rule
  - `policy_evaluator.py` (was evaluator.py): temperature head-to-head + benchmark
  - `scripted_opponents.py` (was baselines.py): Random/FixedColumn/OneStepWin
    fixtures (these are Opponents, not Policies)
  - `task_scheduler.py` (was scheduler.py): round-robin (deliberately simple)
  - `mastery_narrative.py` (was report.py): the `report` command's lineage story
  - `improvement_loop.py` (was loop.py): the train→evaluate→select→archive loop +
    `Store` (still exposes `.archive` / `.tasks` / `.experiments`)
  - `jsonl_store.py` (new): shared append-only JSONL helpers, previously duplicated
    in archive and registry
  - `__main__.py`: CLI `run`/`report`/`verdict`
- **Test suite: 54 tests, all green** (`pytest`; slow canary via `pytest -m slow`).
  Covers env correctness (all win directions, draw, perspective flip), reward
  semantics, archive/registry invariants, double-gate quadrants, evaluator
  orderings, bit-identical determinism, loop integration, and the learnability
  canary (DQN must learn to beat fixed-column within 600 episodes — passes in ~10s).
  Re-run the suite before every experiment batch: it is the proof the harness is
  not a source of training errors.
- **Persistent store** at `store/` (archive/, registry/, telemetry/) — committed
  to git so the fossil record travels with the repo (revisit: LFS/external at
  ~500 MB).
- Docs updated this session: roadmap (select = double gate; task registry;
  benchmark precisely defined; experiments first-class with policy_refs; Phase 0
  test spec; Phase 2 adapter design; Phase 3a/3b intrinsic motivation + free play;
  curriculum-exhaustion protocol) and vision (autotelic agents, Colas et al. 2022).
- **Phase 2 additions (human design review):** policy artifact must list supported
  tasks (per-task adapter/head map + versions trained on); per-task replay buffers
  kept as archival assets (rehearsal baseline). Three registered-experiment
  candidates added: task-identity-as-input (explicit embedding vs inferred),
  prediction accuracy as first-class score alongside play strength, and
  imagination-as-experience-multiplier (Dreamer-style exchange rate + prioritized
  high-TD replay). New working hypothesis (h) in vision §3; Schaul et al. 2016
  added to references.
- **ADR-004 accepted (singular agent / common sensory input):** working hypothesis
  (g) "shared sensory grounding" added to vision §3; Phase 2 adapters explicitly
  designated scaffolding with a named Stage C (one shared sensory interface); new
  roadmap **Phase 7 — Sensory Grounding: One Set of Eyes** (7a versioned pixel
  rendering of mastered tasks, same absolute benchmarks; 7b perceptual pretraining /
  developmental curriculum — intuitive physics before games; 7c perceptual-vs-
  strategic transfer measured separately). Registered prediction added to Phase 2:
  within-family symbolic transfer > 0, symbolic→pixel perceptual transfer ≈ 0.
- Roadmap extended this session (human design review): Phase 3b now states
  explicitly that task prioritization is part of the policy and the scheduler is
  not a one-way door (progressive absorption scheduler → LP-scheduler → intrinsic
  reward → agent-chosen practice → agent-chosen goals). New **Phase 6 —
  Self-Directed Goals & Self-Generated Benchmarks** (6a propose-and-ratify against
  an oracle; 6b grounded feedback where no oracle exists; 6c self-set goals gated
  on a grounded-feedback channel). vision.md Component 0 gained a matching note on
  the self-generated-benchmark tension: self-generated *targets* are the goal,
  self-referential *grading* is the trap.

## Remaining Phase 0 odds and ends

- Diagnostic telemetry is partial: loss curves, epsilon, dormant-neuron fractions
  are recorded per iteration; gradient norms and per-task eval history plots not yet.
- Nothing committed to git yet this session — human to decide commit granularity.

## Next task (plan agreed 2026-08-31, for next session)

Goal: (1) Connect 4 solver benchmark working, (2) begin hypothesis-driven mastery
campaign in the harness. In order:

0. Housekeeping: DONE 2026-08-31/09-01 — harness+tests+docs committed and pushed;
   store/ committed to git (fossil record travels with the repo; revisit trigger:
   move policy binaries to LFS/external when store/ nears ~500 MB); requirements.txt
   pinned exactly (torch 2.9.0, numpy 2.3.4, pytest 9.0.2, Python 3.12); README has
   full any-machine setup + usage instructions.
1. Solver: DONE 2026-09-01 — bitboard negamax + alpha-beta + bound-flagged
   transposition table (`harness/tasks/connect4_solver.py`); threat-count
   heuristic at the depth horizon; exact win/loss/draw values with ply distance
   when search completes; `SolverOpponent` wrapper = depth-ladder rungs.
   11 tests incl. env cross-validation property test, tactics, exact endgame
   values, ladder ordering. Depth 10 ≈ 0.5 s from the empty board (Python).
   Interactive UX: `python -m harness play-solver --depth N` (per-move solver
   evals shown; 'd N' changes depth mid-game). LICENSE: Apache-2.0 added.
2. `connect4-benchmark@v1`: DONE 2026-09-01 — implemented as the progressive
   solver LADDER (depths 1-6, 28 fixed-opening games/rung, 168 games/eval,
   frontier = shallowest rung below 90%; harness/tasks/connect4_ladder.py;
   task registered as connect4@v2). Move-accuracy-over-positions variant
   DEFERRED: exact midgame solves infeasible in pure Python; revisit as
   benchmark v2 (endgame-position subset or faster solver).
   Statistical machinery added (harness/stats.py + statistical_double_gate):
   promotion needs h2h Wilson 95% LB > 0.5; regression blocks only when
   significant; detectable-effect sizes stated in every hypothesis.
   Hypothesis template (human-readable: CONTEXT/HYPOTHESIS/PREDICTION/
   DECISION CRITERIA): docs/hypothesis-template.md. Experiments live as
   committed scripts in experiments/.
3. Retroactive scoring: partially done — incumbent pol-00005 scored 0.030 on
   ladder (frontier depth 1, rung-1 0.18). connect4-rl v0→v32 champs still todo.
4. Mastery campaign: 8 experiments run and judged (all hypotheses + verdicts
   in store/registry/experiments.jsonl; scripts in experiments/). Champion
   ladder path across campaign: 0.030 → 0.083 (pol-00065); frontier still
   depth 1 (rung-1 ~0.42). Ledger:
   - exp-002 more self-play: INCONCLUSIVE (+0.012; relative gains, flat absolute)
   - exp-003 frontier-solver training: REFUTED (too strong; ZPD violation)
   - exp-004 both-players transitions: INCONCLUSIVE (+0.026, best single)
   - exp-005 mirror augmentation: REFUTED (hurt; buffer-correlation suspect)
   - exp-006 diverse opponents: INCONCLUSIVE (+0.003 ladder, strongest h2h growth)
   - exp-007 terminal quota 30%: REFUTED (sampling-distribution skew; worst h2h)
   - exp-008 3x volume + slow eps decay: REFUTED (confounded — decay kept
     challenger half-random; volume itself retested clean in exp-009)
   - exp-009 BIG SWING (both-players + diverse opps + 4500 eps + 50k buffer):
     INCONCLUSIVE (+0.012, best absolute 0.083, 5 promotions, no breakout)
   CAMPAIGN CONCLUSION: data-quantity/opponent/volume levers each move the
   ladder ~+0.01-0.03; nothing clears the +0.09 detectability bar. The
   bottleneck is likely elsewhere. Next single-variable hypotheses, in order:
   (a) reward density — solver-graded move quality as shaped reward (ladder
       guards against reward hacking), or simpler: reward for blocking/making
       immediate threats;
   (b) network capacity — 2x32-filter convs may be too small to encode threat
       patterns; try 64 filters / 3 layers;
   (c) inference-time lookahead — Q-net as evaluator inside a 2-ply search
       (ADR-003 direction); philosophically loaded (grafted vs learned), needs
       its own ADR discussion first.
   KNOWN HARNESS GAPS: no forked-lineage A/B (clean mechanism attribution);
   telemetry file lost if a run is interrupted (results survive in archive —
   recovered exp-008 this way — but write telemetry incrementally).
5. Interaction interface: `python -m harness play` — play the current champion
   in the terminal (design added to roadmap Phase 0: optional Task.interact()
   hook + generic two-player terminal loop). Small; good end-of-day item —
   playing the policy is the qualitative check on what the mastery campaign
   actually produced.

Deferred, deliberately: transfer-metric protocol (needed before Phase 2, not
before step 4); ADR-004 (auxiliary next-state-prediction head as transfer
mechanism — hypothesis discussed 2026-08-31, to be pre-registered with factorial
design vs shared-trunk-only confound); Phase 2 task family.

## Open questions for the human

- License choice for the repo

*(Resolved 2026-09-01: store/ IS committed — portability across machines decided
it; requirements.txt pinned exactly for the same reason.)*
