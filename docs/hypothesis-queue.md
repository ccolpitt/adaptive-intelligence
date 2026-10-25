# Hypothesis Queue

The tight, prioritized list of what we test next and why. **This file is the
human's steering wheel — edit it freely**: reorder, strike, add. The agent
reads it before launching anything new.

Division of labor with the other records:
- This file = the *forward* plan (queued hypotheses + test designs).
- `store/registry/experiments.jsonl` = the immutable *backward* record.
- `store/cards/experiments/story.md` (regenerate: `python -m harness story`)
  = the readable history of levers pulled, expected vs observed.
- `docs/roadmap.md` = the phase-level program (multi-task transfer, world
  models, sensory grounding); this queue is the current campaign's tactics.

**Standing rules** (from evidence, not preference):
1. Multi-seed: ≥ 3 seeds for any effect smaller than the observed seed
   spread (~0.06 ladder points) — exp-016/017.
2. Plateau-stop armed on every run (patience 6, min_delta 0.02).
3. One lever per experiment; packages must be decomposed before credited.
4. Hypothesis + prediction registered BEFORE the run.

**Current reference points (2026-09-20):**
- Champion: `connect4-pol-00148` (gen 13, ladder 0.071, first gate-v2
  promotion). Recipe 3-seed baseline: 0.054 / 0.060 / 0.116 (mean 0.077).
- Best-known training recipe: tactical shaping + cold start + both-players
  data + diverse opponents.

---

**Standing rule 5 (2026-10-03):** fixed training budget — each experiment
declares its game budget (default 15–30 iterations × 300 games) and is scored
on what it reaches inside it. Samples are the currency.

**Harness defect found + fixed (2026-10-03):** the plateau detector had no
warm-up and killed cold-start runs before their breakout (exp-012 was flat
through iter 7, record at iter 11). exp-014/015/016/019 scores are LOWER
BOUNDS. Now: no plateau checks before iteration 12, patience 8.

## RUNNING

- **exp-028-two-heads, extended 60 → 160 cycles.** The two-headed policy
  reached 0.622 scored with no lookahead (every earlier lineage topped out at
  0.113) and the trend was still rising when the budget ran out. This asks
  whether 0.622 is a plateau or a waypoint. Resumable: rerun
  `experiments/exp_028_two_heads.py` to continue.

## NEXT, IN ORDER (from the current evidence)

0. **Train on positions the policy does not generate itself.** The strongest
   diagnosis the project has produced, from a direct investigation on
   2026-10-26. The two-headed policy is tactically **perfect inside its own
   distribution** and fails outside it:

   - constructed position, X holding 0,1,2 on the bottom row with the win at
     column 3: the move head plays column 6 (logit 2.40 against the winning
     move's −0.16) while the value head correctly reads +0.962 and a
     200-position search correctly plays 3 with 178 of 199 visits;
   - scored only on positions it actually faces against a random opponent:
     takes a win 0.71, blocks 1.00;
   - the card reports 0.49–0.65 because it also scores positions where the
     *opponent* is to move — hypotheticals the policy never acts on — and
     positions produced by random play, which it never trains on.

   So the fault is **not** the value head, **not** the search, and **not**
   the negamax signs. It is the move head generalising badly to positions
   self-play never visits. This matters directly for the score: the ladder's
   opponents are solvers, whose positions are nothing like self-play, which
   is a sufficient explanation for 0.62 rather than 0.90.

   Candidate levers, one at a time: randomise the opening a few plies so
   self-play starts from varied positions; mix in the diverse-opponent
   rotation that lineage 2 used; train on positions drawn from games against
   the ladder solvers themselves. The first is cheapest and borrows no
   knowledge of what a good move is.

0b. **Fix the tactical measurement.** Report in-distribution and
   off-distribution tactical rates **separately**, and stop scoring positions
   where the opponent is to move. The current single number mixes three
   different questions and reads pessimistically, which nearly sent the
   investigation after the value head.

0c. **Discount the reward by how long the win took** (human-proposed). A win
   at move 10 should be worth more than a win at move 42, so that taking an
   available win is always preferred to prolonging the game. The value target
   is currently the bare outcome; discounting it by the remaining number of
   moves would make immediacy part of what the policy is trained to want.
   Cheap, and it addresses the same symptom from the reward side rather than
   the data side — so it must be tested separately from lever 0, not with it.

1. **Deeper ladder rungs — `connect4-eval@v2`.** The top rung is a 6-move
   search and the policy now wins 0.55 against it. If scores keep climbing
   the measuring stick becomes the limit. Add rungs at 8, 10 and 12,
   published as a NEW evaluation version so no recorded score is
   invalidated. Build this the moment the extended run clears ~0.75.
2. **Close the tactical gap.** Taking an available win is 0.65–0.91 and
   blocking a forced loss 0.77–0.89, both too low for a policy beating 6-ply
   opponents, and both noisy cycle to cycle. Three candidates, one at a time:
   raise the search budget (a coarse visit distribution is a coarse training
   target); weight the move-head loss toward positions whose outcome is
   already decided; oversample decision-critical positions
   (`TrainerConfig.tactical_fraction`, built and never tested).
   Watch with `python -m harness tactics --lineage connect4-L5`.
3. **A true oracle as the top rung.** The ladder's opponents are
   depth-limited perfect play. Connect 4 is a solved game, so a genuine
   oracle is possible and would be the honest ceiling.
4. **Cross-lineage league.** The gate only compares a challenger to its own
   lineage's champion. Direct matches across lineages would let L5 be
   measured against L1–L4 head to head rather than only through the ladder.
5. **Add checks for R-003 and R-005.** `python -m harness review-checks`
   reports that both recorded defects — a code path assuming a learner's
   internals, and registration blocking its own resume — have no check
   watching for them. Both were crashes rather than silent wrong numbers, so
   the right fix is probably an integration test that forces every path where
   learners differ, not a runtime check.

## PREVIOUSLY QUEUED

- (nothing — exp-022 complete; awaiting human decision on champion reset, see
  DONE)

## NEXT UP (proposed from exp-022's cards — human to confirm)

- **Bounded distillation**: Arm B + leaf/target values clipped to [-1, 1]
  (or a tanh output head). exp-022's cards show distillation DID propagate
  value 35 plies deep but inflated |Q| to ~2.8 and collapsed tactics; the
  diagnosis module's top suggestion is to bound the scale. 3 seeds vs the
  clean baseline (mean 0.064).

## QUEUED (priority order — edit me)

**Rules 6–7 (2026-10-03, ADR-006/007):** evaluation is always the naked
network — no search at eval, anything goes in training. Promotion = strict:
beat the champion head-to-head AND improve the absolute ladder. Parity
rejections are signals ("scale the lever"). Every run ends with a
card-driven diagnosis (`python -m harness diagnose --policy <id>`); its
suggestions feed this queue — the human decides what enters it.

1. **Search depth as a lever** (the legislated thinking budget): 1 vs 2 vs 3
   plies in training, once exp-021 says which arm works.
3. **Archive-league opponents** (de-scaffolding). Replace the scripted
   one-step-win fixture + random with sparring partners sampled from the
   policy archive. Tests the domain-general diversity mechanism AND isolates
   the scripted fixture's contribution for the first time. 3 seeds vs recipe
   baseline.
4. **Parallel world-model head** (the mechanism hypothesis, vision §3).
   Second output head predicts next state; joint training. Score play
   strength AND prediction error. First step of the program's primary claim;
   also the long-term replacement for the shaping oracle.
4. **Benchmark noise reduction.** 168 ladder games can't see effects < ~0.09.
   Raise games/rung for verdict-time scoring (cheap, pure measurement).
5. **Tactical replay bias** (`tactical_fraction`, built): guaranteed batch
   quota of win-available/must-block positions. Cheap.
6. **Positive-signal replay bias** (self-imitation flavor; NOT a filter —
   contrast caution recorded in roadmap). After #5, share machinery.
7. **Seed early-abort**: kill runs below a floor by iteration k, re-seed.
   Cheap compute saver; needs 3+ more completed runs to calibrate the floor.

## DONE (see story.md for full ledger)

- **exp-022 search-in-training: REFUTED** (clean baseline 0.064; search
  picks moves 0.059; search targets 0.053). Distillation propagated value
  35 plies deep but inflated |Q| (max 2.8) and collapsed one-ply tactics
  (take-the-win 0.27). **Champion ratchet materialized under gate v2:**
  pol-00246 promoted at ladder 0.042 (lineage 0.083 → 0.071 → 0.042).
  DECISION NEEDED: reset champion to the archive's ladder-best policy?

- **exp-020 lookahead at inference: SUPPORTED — program record.** Unchanged
  champion: bare 0.071 → 1-ply 0.089 → 2-ply 0.217 → 3-ply **0.310**, zero
  training. Reflex-only inference was the binding constraint.
- exp-019 MC credit propagation: REFUTED (mean 0.038 vs 0.077). Horizon did
  lengthen (0 → 4 plies) but tactical accuracy collapsed — high-variance
  blame drowned the one-ply signal. MC is the wrong instrument.
- exp-018 conv3 (7×7 receptive field): REFUTED (mean 0.070 vs 0.077).
  Side event: first gate-v2 promotion → champion pol-00148.
- exp-017 seed replication: SUPPORTED — spread ~0.06; multi-seed rule born.
- exp-016 recipe ceiling: REFUTED — reproduction failure (0.054 vs 0.116).
- exp-013/014/015 decomposition: recipe is minimal; data levers superadditive.
- exp-001..012: loop closure; data/volume levers weak; capacity not the
  bottleneck; shaping needs cold start; package = program record (0.116).
