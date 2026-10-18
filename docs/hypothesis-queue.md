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
