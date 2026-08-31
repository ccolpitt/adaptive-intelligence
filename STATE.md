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
- **Persistent store** at `store/` (archive/, registry/, telemetry/) — currently
  NOT gitignored; decide whether the fossil record lives in git or just on disk.
- Docs updated this session: roadmap (select = double gate; task registry;
  benchmark precisely defined; experiments first-class with policy_refs; Phase 0
  test spec; Phase 2 adapter design; Phase 3a/3b intrinsic motivation + free play;
  curriculum-exhaustion protocol) and vision (autotelic agents, Colas et al. 2022).
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

0. Housekeeping: commit everything (currently NOTHING is committed); decide
   store/-in-git question; optionally pin requirements.txt.
1. Solver: bitboard negamax + alpha-beta + transposition table. Verify vs known
   ground truth (first-player win, center opening; tactical unit tests). Exact
   solving within bounded remaining depth is sufficient for day one.
2. `connect4-benchmark@v1`: move accuracy over ~500 fixed seeded positions +
   depth-ladder win rates (roadmap Phase 1 spec). NOTE: new benchmark ⇒ register
   task version `connect4@v2`; old scores stay attributed to old versions.
3. Retroactive scoring: gen-0 + pol-00005 on v1; if time allows, the connect4-rl
   champions v0→v32 (answers the long-open "how strong was v32 really?").
4. Mastery campaign: registered experiments, one variable each (longer training,
   symmetric augmentation, terminal-ratio buffer, PER), each vs dumb baseline.
   Target mastery on benchmark-v0 (0.97), track absolute progress on v1. Run
   `pytest -m slow` before each batch. Expected to run past one session.
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
- Should `store/` (the fossil record, includes .pt binaries) be committed to git
  or kept disk-only with backups?
- Python deps are used system-wide (torch 2.9, numpy 2.3, pytest 9) — pin a
  requirements.txt?
