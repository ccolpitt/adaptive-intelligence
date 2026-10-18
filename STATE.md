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

- **exp-022 REFUTED (all 9 runs):** baseline 0.064 / search-moves 0.059 /
  search-targets 0.053. Arm B cards: horizon 33-37 plies (distillation
  propagated value) BUT |Q| inflated (max 2.8) and take-the-win collapsed to
  0.27 — bootstrapping through inflated leaf values destroys move ordering.
  Proposed next: bounded distillation (clip targets to [-1,1]).
  **CHAMPION RATCHET:** gate v2 promoted pol-00246 at ladder 0.042
  (0.083→0.071→0.042). Human decision pending: reset champion to ladder-best.
- **ADR-006 ACCEPTED + ADR-007 ACCEPTED (human rules):** eval = naked network,
  no search (training unrestricted). Gate v3 "strict" is default: promote
  only on h2h win AND ladder improvement; parity flagged as signal. New
  `harness/diagnosis.py`: after every run the best challenger's card is read
  and failing instruments map to concrete levers (telemetry `diagnosis`, CLI
  `diagnose`). 150 tests green. exp-022 runs under gate v2 (recorded).
- **HARNESS DEFECT FIXED — plateau detector had no warm-up.** Cold-start
  curves are flat ~8 iterations then break out (exp-012 flat through iter 7,
  0.116 at iter 11); patience-6 stops truncated exp-014/015/016/019 and
  exp-021-seed24 (those scores are lower bounds). Fix: `plateau_min_iterations
  =12`, `plateau_patience=8`; regression test added. exp-021 aborted
  (verdicts recorded as inconclusive/invalidated); **exp-022 running** = same
  design + Arm 0 baseline re-measured under the corrected detector (9 runs).
- **ADR-006 PROPOSED (human leaning yes): search in training, bare at
  evaluation.** Policy = the network; `train_search_depth` / `search_targets`
  are training levers (built + tested, default off); benchmark unchanged.
  **exp-021 running**: Arm A (search picks moves, seeds 24-26) then Arm B
  (search-backed distillation targets, seeds 27-29). Standing rule 5: fixed
  training budget. 142 tests green.
- **exp-020 SUPPORTED — PROGRAM RECORD (lookahead at inference, no training):**
  `harness/lookahead.py` (`LookaheadQ`: k-ply negamax over real rules, leaves
  valued by the net; drop-in QFunction). Unchanged champion pol-00148: bare
  0.071 → depth-1 0.089 → depth-2 0.217 → depth-3 **0.310** (rung-1 0.29→0.68).
  Reflex-only inference was the binding constraint. Frontier still depth 1.
  NEXT (queue #1/#2): search during training (AlphaZero-lite) + decide via ADR
  whether promotion/benchmark are evaluated with search.
- **exp-019 REFUTED (MC targets, seeds 21/22/23):** mean 0.038 vs 0.077;
  value horizon lengthened 0→4 plies (mechanism worked) but take-the-win
  0.91→0.44 / block 0.89→0.48 — high-variance blame destroyed tactics.
- **`docs/hypothesis-queue.md` created** — the human-editable forward plan
  (running/queued/done, standing rules, priority order). Read it before
  launching anything new. **`python -m harness status`** added: running
  processes, archive heartbeat, champion, pending verdicts.
- **exp-018 REFUTED** (conv3 joint over seeds 18/19/20: mean 0.070 vs baseline
  0.077). SIDE EVENT: first gate-v2 promotion — champion is now
  **connect4-pol-00148** (gen 13, bench 0.071, relative path: h2h 0.56 CI
  0.507-0.621; bench 0.071 vs 0.083 not a significant regression). Monitor the
  not-significant-regression ratchet across future promotions.
- **`mc_targets` lever built + tested** (human-proposed Monte-Carlo credit
  labeling: reward_i = outcome * gamma^(plies_from_end), done=True, subsumes
  loser rewrite; default off). **exp-019 running** (seeds 21/22/23): prediction
  mean >= 0.14 or 2/3 > 0.116, mechanistic check = value_horizon_plies >> 0.
- **Solver calibration verified on champion**: Q at solver-proven wins +0.66
  (ok), proven losses -0.67 (ok) — sign AND magnitude both correct; reward IS
  reaching endgame values; the deficit is propagation depth (horizon ~ 0).
- **Mechanism hypothesis adopted (human-directed):** vision §3 + core.md now
  state the primary mechanism claim — generalization is facilitated by learning
  skills and a world model IN PARALLEL (next-state → trajectories → consequences
  of consequences); every move is a training sample even without reward. Four
  new roadmap candidates registered (2026-09-19): parallel world-model head,
  seed-variance management, positive-signal replay filter (with the contrast
  caution), third conv layer, archive-league opponents (de-scaffolding
  principle: scripted fixture AND shaping both consume the winning_moves
  oracle; successors = archive league + world-model-derived signal).
- **Q instrumentation extended:** per-episode mean |Q| in telemetry
  (`episodes.q_mean_abs`) + card reading vs the [0,1] band + dashboard panel;
  NEW solver-calibration card section — Q at solver-PROVEN wins (→ +1) and
  proven losses (→ −1), harvested from live positions, ground-truth check that
  reward back-propagates. 130 tests green.
- **exp-017 SUPPORTED: recipe 3-seed distribution 0.054/0.060/0.116** (mean
  0.077, spread 0.062) — exp-012 was the lucky draw; typical recipe (~0.06)
  does NOT beat the champion (0.083). STANDING RULE: >= 3 seeds for any effect
  smaller than the spread. **exp-018 launched** (conv_layers 2→3, seeds
  18/19/20, sequential; prediction scored on the 3-seed distribution).
- **exp-016 REFUTED — REPRODUCTION FAILURE, headline finding:** the identical
  recipe scored 0.116 at seed 12 but only 0.054 at seed 16 (plateau-stopped
  iter 7/30). Seed variance is comparable to the effects we measure; exp-012's
  0.116 may be substantially luck. **exp-017 launched** (pure replication,
  seed 17, nothing else changed) to get a third seed → mean ± spread. New
  standing rule proposed in exp-017's hypothesis: single-seed deltas below the
  observed seed spread are noise.
- **2x2 decomposition COMPLETE — exp-015 REFUTED** (plateau-stopped iter 6/15):
  diverse-only 0.018 ≈ the 0.015 floor. Cells: pair 0.015 / +both-players 0.060 /
  +diverse 0.018 / full package 0.116 → the data levers are SUPERADDITIVE
  (diversity manufactures tactical lessons; both-players data records them);
  the exp-012 package is the minimal known recipe. **exp-016 launched**
  (recipe ceiling probe: recipe unchanged, 30 iters, plateau-stop decides the
  endpoint, gate v2 active — first run where a recipe policy can promote).
  Registered fork: >= 0.15 → recipe still scaling; plateau at ~0.116 → signal-
  saturated, next bottleneck is reflex-only inference → search (ADR-003).
- **Campaign story ledger** (`python -m harness story` → `store/cards/
  experiments/story.md`): auto-generated levers/expected/observed markdown from
  the registry, one block per experiment.
- **exp-014 INCONCLUSIVE** (plateau-stop fired live, saved 7/15 iterations):
  shaping+cold_start+both-players hit 0.060 — 4x exp-013's 0.015, half the
  package's 0.116; both-players is a large contributor, not the whole recipe.
  **exp-015 launched** (shaping+cold_start+diverse-opponents, seed 15):
  completes the 2x2 over the two data levers.
- **ADR-005: promotion gate v2** (human-directed): `statistical_either_gate` —
  promote on EITHER proven h2h superiority (old path) OR proven benchmark
  superiority with h2h not significantly below parity. Default for new runs
  (`gate_rule="either"`); old double gate kept as `"double"`. Note: exp-012
  iter-11 still fails v2 (h2h 0.41 UB≈0.47 = proven regression, and +0.033
  bench not significant at n=168).
- **Plateau detection** (human request — "how do we know we scaled enough?"):
  `stats.plateaued` + LoopConfig `plateau_patience`/`plateau_min_delta`/
  `plateau_action` (flag|stop|off, default flag). Stop mode ends the run and
  records `plateau_stop` + iterations saved in telemetry.
- **Tactical replay bias lever** (human proposal): `TrainerConfig.
  tactical_fraction` — guaranteed batch quota of decision-critical transitions
  (win-available or must-block positions, one ply before terminals). Default
  0.0; flip inside a registered experiment. Flags are computed independently of
  reward shaping.
- **exp-013 REFUTED**: shaping+cold_start alone stalled at ladder 0.015 in 4500
  episodes (package hit 0.116) — data levers are jointly necessary.
  **exp-014 launched** (shaping+cold_start+both-players data, seed 14, gate v2,
  plateau-stop armed): tests whether both-players data completes the recipe.
- **Per-episode telemetry + matplotlib dashboards** (human request): the loop now
  records a flat `episodes` block per experiment — loss, grad norm, learner
  result (+1/0/-1, a free rolling win-rate signal), cumulative unique states
  (tracked in `DQNTrainer.unique_states_seen`), epsilon — one entry per episode.
  `python -m harness exp-card --plots` renders a 3×3 PNG dashboard
  (connect4-rl style) from stored telemetry; matplotlib==3.10.7 pinned as an
  optional dep. Old telemetry lacks the episodes block — dashboards render
  "no data" panels for those. PNGs for all 12 experiments generated.
- **Gradient norms now tracked** (human request): `_train_step` returns
  (loss, pre-clip L2 grad norm) — free from `clip_grad_norm_`; per-episode means
  in `EpisodeStats.grad_norms`, per-iteration series in telemetry as
  `episode_grad_norm_means`, trend reading on the policy performance card
  (steady/declining = ok; explosion = instability; collapse + rising dead
  weights = plasticity loss). Old telemetry files simply lack the series — cards
  tolerate that.
- **Experiment cards implemented** (`experiment_card.py`, CLI
  `python -m harness exp-card --experiment-id <id>` or `--all`): joins registry
  (if-X-then-Y hypothesis, one change, frozen conditions, prior art, verdict) +
  telemetry (benchmark/h2h/loss/grad-norm/dormant curves as inline-SVG charts,
  promotions marked) + archive (inputs/outputs). Self-contained HTML per
  experiment + index.html under `store/cards/experiments/` — no matplotlib, no
  PNGs. All 12 existing experiments render; 8 new tests.
- **Policy performance cards implemented** (`performance_card.py`, CLI
  `python -m harness card --policy <id>`): post-hoc diagnostic probe + telemetry
  read producing ok/warn/fail readings per section — value calibration, terminal
  calibration, Connect-N tactical accuracy (auto-detected via `winning_moves`),
  Q-decay/value propagation, exploration, network health, loss/dormant trends.
  Cards saved to `store/cards/*.{json,txt}`; 11 new tests (all green). First real
  card on champion connect4-pol-00065 immediately surfaced: take-the-win 0.93,
  block-the-loss 0.90, and a RISING loss trend in exp-009 — promoted on h2h alone.
  Roadmap now records the diagnostic ladder (human reads cards → agent reads →
  policy self-diagnoses).

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
   Second wave (exp-010..012), diagnosis-driven:
   - exp-010 capacity probe (REFUTED, decisive): current 2conv-32ch net fits
     3200 solver-labeled positions to 100% TRAIN accuracy (large 3conv-64ch
     identical; val 0.405 vs 0.425, both ~chance-adjacent). CAPACITY IS NOT
     THE BOTTLENECK; the signal is. Bigger-net loop experiments deprioritized.
   - exp-011 tactical shaping alone (REFUTED at 1500 eps): -0.5 for foregone
     win / failed block, warm-started → h2h collapsed to 0.27-0.34. Mechanism:
     warm-started value function calibrated on pure win/lose fights the
     re-scaled reward. (Shaping helper: shaping_delta in agents/dqn.py;
     verified -1 lands on the failed-block move; foregone win previously had
     NO signal at all.)
   - exp-012 signal package (INCONCLUSIVE, most important result so far):
     shaping + cold_start + both-players + diverse opponents + 4500 eps +
     50k buffer. Cold-started policy climbed random → ladder 0.116 (peak,
     iter 11) — highest absolute score of the program, beating an incumbent
     that took ~10500 cumulative episodes to reach 0.083, in ~3600 episodes.
     Margin +0.033 < detectable +0.09 → inconclusive. Follow-ups queued:
     (a) decomposition: shaping+cold_start ALONE (is the pair sufficient?);
     (b) replicate exp-012 with 2-3 seeds to beat the noise bar;
     (c) longer run — was iter-11 a peak or a trend?
   GATE DESIGN FINDING (needs human decision): exp-012's iter-11 challenger
   EXCEEDED the incumbent on the benchmark (0.116 vs 0.083) but was blocked
   by the relative gate (h2h 0.41) — the double gate's reverse-v32 failure
   mode: absolutely-stronger-but-stylistically-different policies cannot
   promote. Options: (i) absolute-dominance bypass (promote if significantly
   better on benchmark regardless of h2h); (ii) league-style eval vs multiple
   archived opponents instead of only the incumbent; (iii) keep as is.
   Decide before exp-013.
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
