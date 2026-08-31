# Roadmap

## Purpose

This project builds a single policy that masters a growing family of related tasks and
learns each new task in fewer samples than the last. That is the central bet: transfer
compounds, so every task the policy already knows makes the next one cheaper to learn.
We measure the bet directly — sample-efficiency curves against fixed benchmarks
(vision.md §3). If the advantage compounds, we are on the path. If it does not, we find
out early and cheaply, and the framework's premise fails on the evidence.

## Method

We close the improvement loop with the simplest components that work, then upgrade one
component at a time and measure each upgrade against that simple baseline. This order is
not caution for its own sake. In the connect4-rl precursor, every clever component change
(Double DQN, lower learning rate, wider layers) lost to the plain baseline; only the
loop-level variable — more training — moved the benchmark. So we close the loop first and
earn each complication with a measured win:

> Close the loop with the simplest components first. Then upgrade one component at a
> time, and keep each change only if it beats the simple baseline on the benchmark.

## Phase 0 — Loop Harness

Port and generalize the connect4-rl improvement loop into domain-agnostic interfaces.
The loop is train → evaluate → **select** → archive → repeat.

**Selection rule (the "select" step), double-gated:** a challenger is promoted only
if it (a) beats the incumbent champion (relative gate) AND (b) does not regress on
the fixed benchmark (absolute gate). The relative gate alone is the connect4-rl v32
failure mode — 32 promotions with unknown absolute strength. Win or lose, every
challenger is archived; losers stop receiving compute, never get erased.

### Core interfaces

- [ ] `Task` interface: versioned bundle of environment ID + reward-function ID +
      benchmark ID. A task is a contract: observation space, action space,
      environment step function, reward, and a fixed external benchmark. Changing
      any part → new task version. Same environment + different reward = different
      task.
- [ ] Task registry: append-only, versioned task definitions (`connect4@v1`, …).
      Eval results are meaningless unless pinned to a (policy, task@v, benchmark@v)
      triple; the registry is what makes every archived score interpretable forever.
- [ ] `Policy` artifact: serialized model + embedded metadata — extend the
      TorchScript `_extra_files` pattern. Metadata references registries by ID,
      never copies definitions (see Archive metadata below).
- [ ] `Archive`: append-only policy store (the fossil record). League + archive
      IS the end-of-life mechanism — no ecology.
- [ ] `Evaluator`: runs policy vs opponent(s)/benchmark, records every result
      stamped with (policy ID, task@v, benchmark@v)
- [ ] `Scheduler`: heuristic task/training selection (deliberately simple)

### Archive metadata (per policy)

Mostly pointers — the lesson lives on the experiment, the policy links to it:

- Lineage: parent policy ID, generation
- `experiment_id` of the experiment that produced it
- Training config + task version(s) trained on
- Eval records: list of (task@v, benchmark@v, score, date)
- Promotion history: challenger/champion/relegated transitions with dates
- Epitaph: one line — why it lived or died ("lost double gate to v12; see EXP-031")

### Experiment registry — experiments are first-class citizens

Append-only (formalize connect4-rl's `experiments.json`). Each record:

- [ ] `hypothesis`: "if X, we will see Y" — Y a measurable prediction, registered
      BEFORE the run (framework principle 3)
- [ ] `conditions`: task@v, benchmark@v, code version, config, seeds, compute
      budget — the X/Y/Z under which the verdict holds
- [ ] `changes`: the one variable changed vs the named baseline
- [ ] `results` + `verdict`: supported / refuted / inconclusive — scoped to the
      conditions, never edited after the fact
- [ ] `tldr`: one line, what we learned. Negative results recorded with the same
      care as positive ones.
- [ ] `policy_refs`: the policy version(s) the experiment started from
      (warm-start parent / baseline) and the policy version(s) it produced.
      Experiments ↔ policies are linked in BOTH directions: the archive's
      `experiment_id` points at the producing experiment; the experiment's
      `policy_refs` point at its input and output policies. Given any policy you
      can find the hypothesis that created it; given any hypothesis you can find
      the policies that tested it.
- [ ] `prior_art`: IDs of related past experiments. **Registration protocol
      requires a registry search first** — cite prior art or state none exists.
      This is the anti-regression mechanism: no re-running settled experiments
      unknowingly.
- [ ] `revisit_when` (optional): trigger for retrying under new conditions
      ("retry once curriculum exists"). Revisiting = a NEW experiment citing the
      old via `prior_art`; the old record gains `superseded_by`. Verdicts are
      immutable but condition-scoped — supersession, not mutation.
- [ ] Tags + structured fields for indexing. Start simple: append-only JSONL,
      greppable. Add a real index (embeddings/wiki) only if scale demands it.
- [ ] Diagnostic telemetry from day one: per-run training curves, gradient norms,
      dormant/dead-neuron stats, per-task eval history — persisted with each
      experiment. This feeds Phase 4 (plasticity) and the diagnosis-driven
      architecture tenets (vision.md, Component 1). Diagnostics must be able to
      DISAMBIGUATE causes, not just detect symptoms.
- [ ] Mastery narrative ("explain how we did it"): a `report` command that, given
      a policy ID, walks its lineage and the linked experiments to emit the story —
      each ancestor's hypothesis, verdict, and tldr in order. What worked, what
      didn't, and which changes produced which jumps. This is a pure read over the
      archive + registry; if it can't be generated, the metadata links are broken.
- [ ] Curriculum-exhaustion signal (simple version): loop status output includes
      per-task mastery flags (benchmark score ≥ threshold) so a human can see at a
      glance when the agent has topped out. Full protocol in Phase 3.
- [ ] Interaction interface ("play the policy"): every Task may expose a
      human-facing way to EXPERIENCE a policy — `python -m harness play
      --policy ID` (default: current champion). The concept is task-generic:
      for board games, play against it in the terminal (render board, accept
      human moves, policy replies greedily); for a music task it would be
      listening to the composition; for a document task, opening the produced
      artifact. Design: `Task` gains an optional `interact(q_function)` hook;
      the harness ships a generic terminal play loop for two-player
      perspective-flipped board tasks. Qualitative inspection is a diagnostic
      channel that benchmarks miss (blind spots, weird openings) — and never a
      substitute for them.
- [ ] Connect 4 ported as seed task #1; champion-challenger loop runs end-to-end
      inside the harness

### Phase 0 harness test suite

The harness must be provably not a source of training errors — these tests are the
proof, re-run before every experiment (cheap suite) and after any harness change.

- [ ] **Determinism / reproducibility:** same seed → bit-identical episode
      trajectories, identical eval scores, identical replay-buffer contents after
      N episodes. Any nondeterminism in the harness silently corrupts every
      A/B comparison.
- [ ] **Environment correctness (Connect 4):** win detection in all four
      directions including edges/corners; illegal-move rejection (full column);
      draw on full board; turn alternation; perspective-flip invariant (channel 0
      is always the mover's pieces). Golden-game tests: replay fixed move
      sequences, assert exact board states and outcomes.
- [ ] **Reward semantics:** mover-win → +1 to mover; loss reward correctly
      reaches the loser's transitions (the connect4-rl post-hoc `update_penalty`
      patch is a known trap — the port must make loss attribution explicit and
      tested); draw → DRAW_VALUE; zero-sum invariant per game.
- [ ] **Archive invariants:** append-only (writing an existing ID fails; no
      delete API exists); save→load round-trip preserves weights (identical
      outputs on fixed inputs) and metadata; lineage integrity (every parent ID
      resolves); `champion_current` is a pointer, never a mutation of an
      archived artifact.
- [ ] **Registry invariants:** append-only; verdicts immutable once written;
      schema validation on every record; `prior_art`/`policy_refs`/
      `experiment_id` cross-references all resolve; task/benchmark version pinning
      present on every eval record.
- [ ] **Selection gate:** table-driven tests with a fake evaluator — promotion
      iff (beats champion) AND (no benchmark regression); all four quadrants
      covered; stagnation-revert path covered.
- [ ] **Evaluator:** scripted fixture policies with known strength ordering
      (e.g., win-next-move-if-possible vs random) produce the expected ordering
      and scores within statistical tolerance at fixed seed; every result stamped
      (policy ID, task@v, benchmark@v).
- [ ] **Learnability canary (the "harness isn't blocking learning" test):** a
      small network trained inside the harness on a trivially learnable fixture
      (e.g., vs an opponent that always plays column 0) must reach a win-rate
      threshold within a fixed episode budget at fixed seed. If the canary dies,
      the bug is in the harness plumbing, not the science. Marked slow; run
      before every experiment batch, not every commit.
- [ ] One command runs the whole suite (`pytest`); fast subset vs slow canary
      separated by markers.

**Done when:** one command runs the full loop on Connect 4 and produces an archived,
metadata-rich policy lineage plus an experiment record — with the challenger's
promotion/rejection decided by the double gate, every score stamped with
(task@v, benchmark@v), the experiment's hypothesis registered before the run, and
the full test suite green.

## Phase 1 — Measurement Foundation

**What a benchmark is, precisely:** a versioned, immutable measurement procedure —
`score: Policy → ℝ` — defined by (1) fixed evaluation conditions (opponents/oracles/
test scenarios, start states, seeds, trial counts), (2) a fixed scoring rule, and
(3) a fixed protocol (compute limits, sides played, deterministic vs sampled
actions). A benchmark is NOT a policy: its fixed opposition may be an oracle
(minimax solver), a frozen policy, a script, or a static test suite. The defining
property: nothing inside it moves when the population moves (Elo and champion
win-rate therefore don't qualify). "Never changes" means: any change → new
benchmark version; old scores stay attributed to the old version. Immutability of
records, not immutability of the world.

- [ ] Minimax/perfect-play solver for Connect 4 (absolute yardstick; game is solved)
- [ ] `connect4-benchmark@v1` — two components. Primary: **move accuracy** — over a
      fixed, seeded set of positions, the fraction of policy moves that preserve
      the position's game-theoretic value per the solver (perfect play = 1.0; dense
      signal; works from any position). Secondary: **depth-ladder outcomes** — win
      rate vs the solver capped at depths 1/2/4/8, N games per rung.
      (Naive "play the perfect solver" is broken: Connect 4 is a solved
      first-player win, so a perfect opponent moving first beats every policy —
      constant zero, no signal.)
- [ ] Transfer-metric protocol: episodes-to-threshold, ± pretraining, fixed seeds,
      matched compute
- [ ] Retroactively score the archived connect4-rl champions (v0→v32) on the
      absolute scale — first real data point on what the old loop achieved

**Done when:** the harness assigns any archived policy an absolute strength score on
`connect4-benchmark@v1`, the transfer protocol is documented and unit-tested, and the
connect4-rl champions v0→v32 carry retroactive absolute scores in the archive.

## Phase 2 — Transfer Hypothesis Test

**How one policy spans tasks with different interfaces** (6×7 Connect 4, 3×3
tic-tac-toe, 8×8 chess, someday Atari pixels): the policy is split into three
pieces — per-task **input adapter** → shared **trunk** → per-task **output head**.
Each Task publishes an `obs_spec` and `action_spec`; adapters are thin, auto-built
from the specs; the trunk is where the transfer hypothesis lives. Adding a task =
adding two thin adapters, never rebuilding the trunk. Design consequences, staged:

- **Stage A (this phase, board games):** per-task encoder (small conv/linear) into
  a fixed-width latent; per-task action head over that task's move set. For the
  Connect-N family a padded canonical board (pad every board into a max-size grid
  + size-mask channel) is a registered experiment vs separate encoders — the
  padded variant shares MORE weights and may transfer better. Both are cheap.
- **Stage B (Tier 1+, later):** wildly different observation types (pixels, text)
  get their own encoder families; the trunk contract (latent in, latent out) is
  what stays stable. The Gato-style fully tokenized universal interface is the
  limit of this pattern — noted, not built.
- The `Task` interface must therefore expose `obs_spec`/`action_spec` from
  Phase 0, even while only Connect 4 exists — that is the only Phase 0 cost of
  this whole design.

Growing the trunk itself when the portfolio outgrows it is working hypothesis (f)
(capacity growth, Phase 4+) — distinct from adapters.

- [ ] Task family: Connect-3, board-size variants, tic-tac-toe, etc., all behind the
      `Task` interface
- [ ] Single trunk, per-task adapters (Stage A); padded-canonical-board vs
      separate-encoders as a registered experiment
- [ ] Measure: does mastery of task A accelerate task B? Compounding across 3+ tasks?
      Forward transfer (episodes-to-threshold) AND backward transfer (retention on
      A after training B — the catastrophic-forgetting metric, measured from the
      first two-task run onward, not deferred to Phase 4)
- [ ] Learned-dynamics track (world model required — ADR-003): withhold the rules;
      the agent learns a transition model from observed transitions and plans against
      it (MuZero-style). Diagnostic: compare planning with the learned model vs the
      true rules to measure world-model error directly.

**Done when:** sample-efficiency curves answer whether mastering task A reduces the
episodes-to-threshold on task B under matched compute, with forward- and
backward-transfer numbers reported for every task pair. Either outcome is a recorded
result.

## Phase 3 — Pedagogical Evolution & Intrinsic Motivation

Two places "what is interesting right now?" can live, upgraded in order:

**3a — Scheduler-level (external motivation).** A learning-progress scheduler replaces
the Phase 0 heuristic scheduler: each task's interestingness = the slope
of its benchmark curve. Solved tasks (slope ≈ 0, ceiling) and impossible tasks
(slope ≈ 0, floor) both lose compute; the frontier gains it. This is measurable,
lives outside the network, and doesn't contaminate task rewards — which is why it
goes first.

- [ ] Auto-curriculum over the task family: sample tasks by learning progress
      (PLR-style / Oudeyer-style)
- [ ] Compare vs uniform and hand-ordered curricula under matched compute;
      metrics: aggregate sample efficiency AND retention across all tasks

**3b — Agent-level (intrinsic motivation).** Move "interesting" inside the reward:
add an intrinsic bonus to the task reward and let the agent's own behavior — not
the scheduler — allocate attention. Registered experiments, one variable at a time:

- [ ] Curiosity as prediction error (ICM/RND-style): bonus for states the agent
      can't predict; measure exploration coverage + benchmark effect vs 3a baseline
- [ ] Mastery/competence progress as reward (Oudeyer): bonus proportional to the
      agent's own recent improvement — mastery for its own sake; boredom
      (bonus → 0) emerges when a task is topped out
- [ ] **Free play:** present the task menu as part of the environment — a
      meta-environment whose first action is "which task do I practice now?" —
      and let the intrinsically-motivated agent switch tasks itself, the way a
      human drifts from piano to dancing to an essay. Hypothesis: agent-chosen
      practice schedules match or beat the 3a scheduler on sample efficiency and
      retention under matched compute. Requires interleaved rehearsal (working
      hypothesis e) to be in place, else forgetting confounds the result.

**Prioritization is part of the policy, not a fixture around it (the scheduler
is not a one-way door).** Deciding *what to work on, who to spend time with, and
why* is itself an act of intelligence — a human can spend a day opening and
closing a door (waste) or choose to master a new skill and connect it to what
they already know. Phase 0's `Scheduler` is a deliberately dumb external
stand-in for that choice; 3a makes it smarter but still external; 3b (free play)
is the first phase where the *choice moves inside the policy*. The end state:
the object we train and archive is not "a Connect-4 player" but "a chooser that
also plays Connect 4" — the action space includes which task to attend to, and
the same network that learns to play learns to allocate its own attention. This
is a progressive absorption, not a switch: scheduler → learning-progress
scheduler → intrinsic reward → agent-chosen practice → (Phase 6) agent-chosen
*goals*. Each rung is a registered experiment against the rung below, so we can
tell whether internalizing the choice actually helps or just adds variance.

**Curriculum-exhaustion protocol (human-in-the-loop trigger).** Mastery of a task =
benchmark score ≥ its mastery threshold, sustained over K consecutive evals.
The curriculum is exhausted when every task is mastered AND aggregate learning
progress ≈ 0 for M evals (nothing left to climb).

- [ ] Mastery thresholds defined per task version, stored in the task registry
- [ ] On exhaustion: loop emits a CURRICULUM_EXHAUSTED event — writes a status
      record, prints/notifies, and idles on rehearsal-only mode (maintain, don't
      overfit) rather than burning compute
- [ ] The human's job on notification: add the next task(s) at the frontier —
      hard enough that progress > 0, close enough that transfer applies (zone of
      proximal development). The environment ladder below is the menu to pull from.
- [ ] Also fires the inverse alarm: if NO task shows progress and none is
      mastered, the curriculum is too hard or something is broken — notify too.

## Phase 4 — Continual Plasticity

- [ ] Sequential-task benchmark: measure catastrophic forgetting explicitly
- [ ] Plasticity-loss tracking (dormant-neuron stats); neuron reinit (extend the
      connect4-rl prototype)
- [ ] Gradient-guided structural growth (RigL-style), vs static baseline

## Phase 5 — Diagnosis-Driven Architecture Evolution

The loop: instrumented failure → informed proposal (agent/human reads diagnostics +
literature) → registered falsifiable experiment → result recorded. See the tenets in
`docs/vision.md` Component 1 (diagnosed not divined; scales with compute; names its
enrichment; falsifiable prediction; one change at a time).

- [ ] Diagnosis catalog: map observed pathologies (per Phase 0 telemetry) to candidate
      architectural responses
- [ ] Proposal protocol: every architecture experiment cites the diagnostic evidence
      that motivated it
- [ ] If automated search is used: constrained grammar (depth, width, blocks, skips),
      small population + league selection, PBT-style warm starting
- [ ] Only architectures that beat the fixed benchmark AND the incumbent survive

Note: hand-designed architecture experiments (e.g. the cortical-column/voting
experiment, vision.md Component 1) are ordinary registered experiments and may run in
ANY phase — Phase 5 is about systematizing the diagnosis→proposal loop, not about
gatekeeping design work.

## Phase 6 — Self-Directed Goals & Self-Generated Benchmarks

The furthest-out phase, and the one in most tension with Principle 1 ("every task
carries a fixed external benchmark"). It exists because real learning often has no
supplied yardstick: a person learning to perform music, hold a conversation, or
write an essay has no oracle scoring every move. Mastery there is *constructed* —
by soliciting feedback, reasoning about what "good" would mean, and committing to a
larger outcome (Musk choosing a multi-planetary species as the objective that
orders everything below it). Eventually the policy must do the same: take a new
task, propose its own benchmark, and defend why passing that benchmark constitutes
mastery.

**The trap this phase must not fall into is exactly Component 0's failure mode:** a
policy that invents its own benchmark and then scores well on it has measured
nothing (self-referential measurement — the connect4-rl v32 disease, one level up).
The whole discipline of Phases 0–1 exists to prevent this, and Phase 6 must not
quietly undo it. The resolution, staged:

- **6a — Benchmark *proposal*, human/oracle *ratification*.** On a task that still
  HAS an oracle (a Connect-N variant, a held-out classification set), the policy
  proposes a benchmark — a measurement procedure and a mastery threshold — and
  reasons about why it captures mastery. We then check the proposed benchmark
  against the known oracle. Question: can a policy generate a benchmark that
  correlates with ground truth on tasks where we can grade the benchmark itself?
  This is falsifiable and cheap, and it earns the right to the harder version.
- **6b — Grounded feedback where no oracle exists.** For open-ended tasks
  (dialogue, writing, music) the "external, never-changing" signal becomes
  *grounded human/environment feedback*, not a self-generated score: solicited
  human preference judgments, downstream real-world outcomes, or a frozen
  learned reward model with a dated version stamp (a reward model is a benchmark
  only while frozen — the moment it co-adapts with the policy, Elo-style
  non-transitivity returns). The invariant survives in spirit: *the yardstick's
  signal must originate outside the policy being measured*, even when it is no
  longer a clean oracle.
- **6c — Goals, not just tasks.** The policy chooses which self-set goal to pursue
  and why, ordering sub-tasks under a larger committed objective. This is Phase
  3b's free-play choice extended from "which of the given tasks" to "what is worth
  doing at all." Deferred hard, with an explicit guardrail: no self-set goal is
  admitted without a grounded feedback channel (6b) attached — otherwise the
  system optimizes its own approval and we have built the terrarium, not the
  scientist.

- [ ] 6a: benchmark-proposal protocol; measure proposed-vs-oracle correlation on
      graded tasks
- [ ] 6b: grounded-feedback benchmark type (human preference / downstream outcome /
      frozen dated reward model), with the same version-pinning discipline as every
      other benchmark
- [ ] 6c: self-set goals gated on an attached grounded-feedback channel

**Depends on:** Phase 1 (what a benchmark rigorously is) and Phase 3 (intrinsic
motivation, free play). **Revisit trigger to promote from "later" to "now":** the
task family exhausts what human-authored benchmarks can cheaply cover, i.e. we
start wanting to train on tasks (open-ended generation, interaction) where writing
the oracle is itself the bottleneck. Records an ADR when promoted.

## Not Phases (running throughout)

- League + archive selection pressure (from Phase 0)
- Experiment registry discipline
- STATE.md updates every session

## Environment Ladder (beyond Phase 2's task family)

Promotion rule: an agent graduates a tier only by demonstrating transfer — learn tier
N+1 measurably faster because of tier N, without losing tier N. Details and references
in `docs/vision.md` §3.

- Tier 0: Connect-N family, tic-tac-toe (solvers = exact evaluation)
- Tier 1: MiniGrid / BabyAI (partial observability, grounded language)
- Tier 2: Crafter / Craftax (open-ended, long-horizon; world models earn their keep)
- Tier 3: DM Control, Procgen (physics, procedural generalization)
- Tier 4: Melting Pot, Hanabi, Overcooked (other agents, conventions, theory of mind)
- Tier 5: Minecraft via MineDojo (open world; beyond single-machine compute)

## Parked (see docs/vision.md §5 for revisit triggers)

- Resource-competition ecology
- Learned structural-update policy
- Physical embodiment

*World models were parked here in Revision 2; promoted via ADR-003 to Phase 2.*
