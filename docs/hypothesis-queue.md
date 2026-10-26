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

- **exp-029-two-heads-extended, 60 → 160 cycles.** At cycle 123 of 160:
  champion 0.646, best cycle 0.670, both the highest in the project's history
  and past the 0.622 it started from. Resumable:
  `caffeinate -is python3 -u experiments/exp_029_two_heads_extended.py`.

  **Two confounds, both annotated rather than fatal (learning rule 12).**
  First, it began with a cold restart: the optimiser's accumulated averages
  and the replay buffer were not carried across the process boundary, which
  dropped the score from 0.622 to 0.390 and cost 27 cycles to recover
  (R-007, now fixed for future runs). Second, it has been searching far more
  shallowly than its rubric declares, because terminal positions were charged
  to the budget on every re-visit — a declared 200 positions bought 14 in a
  win-in-one position (R-008, now fixed). So the recipe it recorded is not
  the recipe it ran, and its verdict must say so.

## A CORRECTION THAT CHANGES HOW EVERY SCORE IS READ (2026-10-26)

The ladder score lives in `[0, 1]` and has been read throughout as though
`1.0` were reachable and `0.90` at every rung were the bar. Measured against
players whose strength is known by construction
(`python -m harness reference`):

| player | ladder score |
|---|---|
| random | 0.000 |
| prefer the centre column | 0.000 |
| solver, 1 ply | 0.277 |
| solver, 4 plies | 0.488 |
| **our champion, no search** | **0.646** |
| solver, 8 plies | 0.717 |
| solver, 10 plies | 0.830 |

So the champion is at 78% of a ten-ply exhaustive search while using a single
forward pass, and the project's own "0.646 out of 1.0" reading was badly
pessimistic. Two reasons the ceiling is low: the benchmark forces both players
through fixed openings and plays the policy in both seats, so some games start
from objectively lost positions; and every rung, including depth 1, takes every
immediate win and blocks every immediate threat, because the solver checks for
a win before it checks its depth limit. Rung 1 is not a weak opponent.

Also worth keeping: the ceiling is a **measurement, not a deduction.** On
depth 8 alone it looked as though 0.72 were the limit and 0.90-per-rung were
impossible. Depth 10 then reached 0.830 with 0.875 at two rungs. Whether 0.90
is reachable is still open; that 1.0 is not is settled.

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

0b. **DONE (2026-10-26): the tactical measurement is split.** In-distribution
   and off-distribution rates are reported separately, broken down by the
   geometry of the line, with unambiguous attribution.
   `python -m harness tactics-by-geometry`. Results and what they mean: see
   0f2 below. The old single number mixed three questions and read
   pessimistically, which nearly sent the investigation after the value head.

0c. **Discount the value target by how far the position is from the end**
   (human-proposed, and the reasoning is now worked out).

   *What it is today.* Every position recorded in a game is labelled with the
   bare final outcome from that position's own mover's perspective: `+1` if
   that mover went on to win, `−1` if it lost, `0` for a draw. The
   alternation is correct — consecutive positions genuinely alternate — but
   there is **no discount**. The opening position of a won game is labelled
   `+1.0`, exactly like the winning move. The γ of 0.99 that exists in the
   configuration is used only inside the search's backup and never touches
   the training target.

   *Why that is defensible and still probably wrong here.* The bare outcome
   is an **unbiased** estimate of the position's true value under the current
   policy, which is why AlphaZero uses it. But its variance is enormous: in
   self-play between two copies of one policy, the outcome of a midgame
   position is close to a coin flip, so the label is mostly noise. Scaling by
   `γ^(plies remaining)` **biases** early values toward zero and cuts that
   variance sharply. At 200 games per cycle this project is variance-limited,
   not bias-limited, so the trade is likely favourable — and the bias is
   arguably honest, since it asserts that the opening is roughly balanced,
   which is true.

   *Two further effects worth predicting separately.* It gives the policy a
   reason to finish quickly, which is the human's "killer instinct"
   intuition. And it shrinks the noisy value gradient flowing through the
   **shared trunk**, which may help the move head as a side effect — the two
   heads share every convolutional layer and the losses are summed at weight
   1.0.

   *Design.* One lever: `value_target_discount` in the rubric, γ per ply
   remaining. Try 0.99 and 0.95. Measure the benchmark score, both tactical
   rates, and the mean game length — the last is the direct test of whether
   immediacy was actually learned. Three seeds, since the effect may be
   smaller than the seed spread.

   *Variant to hold in reserve, not to combine:* train the value head on a
   blend of the final outcome and the search's own backed-up root value, as
   KataGo does. The search value has much lower variance and is available at
   once. Test only after the discount is settled.

0d. **Weight the move-head loss toward decisive positions, and leave the
   value head's sampling alone** (human-proposed; the one the human singled
   out). *"If I'm taking a bus to a piano lesson, I can fall asleep on the
   bus without consequence, but if I fall asleep in the piano lesson, I miss
   out on learning a new piece."*

   *The measurement that motivates it.* Over 616 positions from champion
   self-play: a win was available in **3.6%**, a block was required in
   **9.6%**, and **86.9% were quiet**. The buffer samples uniformly, so
   roughly seven eighths of the move head's gradient comes from positions
   where almost any sensible move is fine.

   *Why this is not a repeat of exp-007, which was refuted.* exp-007 forced a
   30% terminal quota on the single-output learner and produced the worst
   head-to-head of that campaign (0.36–0.39), diagnosed as distorting the
   sampling distribution: endgame values over-fitted, midgame values
   under-trained. That damage was to **value estimation**, and the
   single-output learner could not separate the two jobs because one set of
   numbers did both. The two-headed learner has **two losses with separate
   targets**, so the policy loss can be reweighted while the value
   regression continues to see the natural distribution. That is a
   structurally different experiment, not a re-run.

   *Design.* One lever: a per-example weight on the policy loss only, raised
   for positions where the mover has a win available or must block one. Start
   modest — weight 3, which lifts decisive positions from about 13% to about
   30% of the policy gradient without touching which examples are drawn.
   Measure: takes-an-available-win and blocks-a-forced-loss as the primary
   outcomes, benchmark score as the thing that must not regress, and the
   value head's calibration probes as the guard against exp-007's failure
   mode reappearing through the shared trunk.

0e. **Prioritized replay on "hot" positions** (human-proposed). The sampling
   counterpart to 0d: change which examples are *drawn*, not how they are
   weighted once drawn. Strictly riskier than 0d, because it does distort
   the value head's distribution — which is exactly what refuted exp-007 —
   so **0d is tested first and this only if 0d helps.** If it is run, define
   "hot" by a measured quantity rather than by hand: the disagreement between
   the move head's prior and the search's visit distribution is the natural
   choice, since it marks positions where deliberation changed the answer,
   and it needs no knowledge of Connect 4. Reference: Schaul et al., ICLR
   2016.

0f. **Does batch normalisation help or hurt?** (human-proposed, after a
   measurement neither of us expected).

   *What it does, since this is easy to misread:* it normalises
   **activations**, not weights. After each convolution it takes each
   channel, subtracts the mean and divides by the standard deviation computed
   **across the examples in the batch**, then applies a learned scale and
   shift. During training those statistics come from the batch of 128. At
   inference on a single position no batch exists, so it uses a running
   average accumulated during training.

   *The measurement.* On 256 positions from random play, switching which
   statistics are used flips the **chosen column in 13% of positions** and
   moves the value estimate by up to a full 1.0. So the normalisation is
   load-bearing in actual decisions. What is **not** yet measured is whether
   the running averages are wrong for the deep positions the search descends
   into, which are drawn from a different distribution than the buffer.

   *Design.* Champion-versus-challenger on one lever, three arms: batch
   normalisation as now; removed entirely; and replaced by a normalisation
   that does not depend on batch composition. Measure the benchmark score and
   — the specific thing at issue — the agreement between the two statistics
   on the positions the search actually visits.

0f2. **MEASURED, 2026-10-26: diagonals are about twenty points harder than
   straight lines, and the human asked the right question.** Run it with
   `python -m harness tactics-by-geometry`. On champion `pol-00132`,
   attributing only positions with exactly one relevant column completing
   exactly one line:

   *From the policy's own play (in-distribution):*

   | geometry | takes the win | n | blocks the loss | n |
   |---|---|---|---|---|
   | vertical | 0.83 | 24 | **0.98** | 208 |
   | horizontal | 0.87 | 121 | 0.90 | 206 |
   | diagonal\\ | 0.67 | 76 | 0.79 | 146 |
   | diagonal/ | 0.61 | 64 | 0.79 | 163 |

   *From random play (off-distribution, 500 games):*

   | geometry | takes the win | n | blocks the loss | n |
   |---|---|---|---|---|
   | vertical | 0.51 | 751 | **0.93** | 526 |
   | horizontal | 0.57 | 869 | 0.63 | 734 |
   | diagonal\\ | 0.57 | 195 | **0.53** | 92 |
   | diagonal/ | 0.59 | 170 | 0.76 | 67 |

   **Two separate effects, and they want different fixes.**

   *Geometry.* In-distribution, straight lines beat diagonals by roughly
   twenty points on both tactics: taking a win 0.83–0.87 against 0.61–0.67,
   blocking 0.90–0.98 against 0.79. Both gaps are around four to five
   standard errors, so they are real and not sampling noise. The ordering
   makes mechanical sense. A vertical block is the easiest tactic in the
   game: the column to play **is** the column holding the threat, so the whole
   pattern lives in one column. A diagonal block is the hardest: the line
   spans four columns, and *which* column blocks it depends on where a piece
   would land, which depends on how full the neighbouring columns are.

   This is **independent evidence for 0g**, arrived at from a different
   direction. Working out which column's landing row intersects a diagonal is
   exactly the spatial relationship that flattening 64 × 6 × 7 into a
   128-unit layer destroys. Two unrelated lines of reasoning now point at the
   same change, which is the strongest argument available for trying it.

   *Distribution.* Off-distribution everything degrades, and the ordering
   changes rather than simply shifting: vertical blocking survives almost
   intact (0.93) while horizontal blocking collapses from 0.90 to 0.63, and
   taking a win becomes uniformly poor at about 0.55 with **no geometric
   pattern at all**. So win-taking off-distribution is a generalisation
   failure (item 0), while the diagonal weakness is a representation failure
   (item 0g). They are not the same problem and should not be fixed together.

   *This also closes 0b.* The split in- against off-distribution, and the
   removal of opponent-to-move hypotheticals, is implemented in
   `harness/tactical_probe.py` with the attribution rule that a position
   which cannot be ascribed to one geometry is not counted under any.

0g. **A policy head that keeps the board's shape** (human-proposed). Today
   the trunk's output is flattened from 64 channels × 6 × 7 = 2688 numbers
   straight into a 128-unit layer, a 21-fold compression, and only then into
   7 column scores. Every spatial relationship must be reconstructed inside a
   dense layer.

   AlphaZero instead applies a **1 × 1 convolution** to compress 64 channels
   to about 2 *at every cell*, preserving the 6 × 7 layout, and flattens only
   at the very end. A 1 × 1 convolution cannot itself detect a pattern — it
   sees one cell and mixes channels — and that is the point: the trunk has
   already done the spatial work, and the 1 × 1 carries the result forward
   **located** rather than scrambled, into an output that is itself
   per-column.

   *Design.* One lever at a time, and the trunk depth is a separate arm from
   the head shape. Arm A: 1 × 1 convolutional heads, trunk unchanged. Arm B:
   trunk deepened from 3 to 6 convolutional layers with the current heads —
   noting that three 3 × 3 layers already give a 7 × 7 receptive field, so
   extra depth buys reasoning about combinations rather than a wider view, and
   that without residual connections plain stacking may train worse. New
   lineage for each, since the network's shape changes.

0h. **First-play urgency in the search** (agreed defect-adjacent, but a
   behaviour change so it gets a hypothesis). `_select_action` scores an
   unvisited child with `Q = 0.0`. When the policy is winning and its visited
   children sit near `+0.6`, an unvisited child at `0.0` looks *worse*, so the
   search stops widening exactly when it is ahead; when losing, unvisited
   children look better than they are and it thrashes. The standard remedy is
   to initialise an unvisited child to its parent's value. One line, and it
   interacts with the terminal-billing fix (R-008), so measure it after that
   change has settled rather than alongside it.

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

## LONG HORIZON — FILED, NOT QUEUED

- **A network that re-initialises parts of itself and distils into them**
  (human-proposed, 2026-10-26, and the human's own assessment is that it
  would be a significant result).

  *The idea.* Deliberately let parts of the network fall to zero, then
  re-initialise those parts to take on new material, while the knowledge
  already held is preserved by the network teaching itself — distilling its
  current behaviour into the surviving capacity before the reset. The payoff
  the human names: **learning would never have to restart from scratch.**
  Every architecture change in this project so far has forced a new lineage
  from random weights precisely because weights cannot survive a change of
  shape, and four lineages of accumulated competence were abandoned that way.

  *Why it is filed rather than queued.* It is three mechanisms, not one:
  deciding *which* capacity is spent, preserving behaviour across the reset,
  and allocating the freed capacity to new material. Each needs its own
  measurement, and none can be evaluated honestly until there is a **second
  task** to transfer to — with only Connect 4 in play, "learning the next
  task more cheaply" has nothing to measure against. It also sits adjacent
  to the deferred learned-structural-update item in `docs/vision.md` §5, so
  it needs the human's explicit agreement before any of it is built.

  *What already exists to build on.* The weight-change metric per training
  step and its two expectation checks (added 2026-10-26) are the measurement
  this needs: they say when a network has stopped absorbing new material,
  which is the trigger condition for a reset. The dormant-fraction probe per
  layer already reports which capacity is idle. Roadmap 8d's curriculum
  trigger is the same signal used for a different purpose.

  *Revisit trigger.* When a second task exists and a single network is being
  trained on both — that is, when roadmap Phase 2 is underway — and the
  weight-change metric shows a lineage has stopped learning while still short
  of its benchmark ceiling. At that point the first of the three mechanisms
  (preserve behaviour across a deliberate partial reset) becomes testable on
  its own, with "the benchmark score survives the reset" as the measurement.

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
