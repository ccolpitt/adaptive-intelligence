# Campaign story — every lever pulled, expectation, and result

Auto-generated from the experiment registry (`python -m harness story`); do not edit by hand.

Experiments: 16. Verdict legend: SUPPORTED = prediction held; REFUTED = prediction failed (knowledge gained, lever closed); INCONCLUSIVE = effect below what the sample size can detect; PENDING = still running or unjudged.

## exp-001-close-the-loop — SUPPORTED

**Lever(s) pulled:** none — first loop closure, dumb components on purpose

**Expected:** if the ported champion-challenger loop trains DQN self-play on Connect 4 for 5x300 episodes, at least one challenger passes the double gate and the champion's benchmark-v0 score exceeds the untrained gen-0 score

**Observed:** Loop closed: 1/5 challengers promoted via double gate; benchmark-v0 rose 0.725 to 0.785. Dumb components suffice to climb; benchmark-v0 (vs random) is weak - Phase 1 solver benchmark is the real yardstick.

**Key numbers:** `{"episodes": 1500, "final_benchmark": 0.785, "gen0_benchmark": 0.725, "iterations": 5, "promotions": 1}`

## exp-002-more-self-play — INCONCLUSIVE

**Lever(s) pulled:** training volume only: +1500 self-play episodes (5 x 300); no architecture, hyperparameter, or representation changes

**Expected:** If true, we expect the final champion's ladder score to be at least 0.09 above the incumbent's (0.030 -> 0.120), because early-stage DQN gains scale with episodes and 0.09 is the smallest gain our 168-game ladder evaluation can distinguish from noise.

**Observed:** Ladder gain +0.012 (0.030->0.042), below the 0.09 detectable threshold. 3 promotions on h2h yet absolute strength flat: self-play makes challengers better at beating each other, not at beating the solver. Frontier stuck at depth 1 (rung-1 score 0.18). Motivates testing training-vs-measurement distribution match (exp-003).

**Key numbers:** `{"final_ladder": 0.042, "frontier_depth": 1, "incumbent_ladder": 0.03, "observed_gain": 0.012, "predicted_min_gain": 0.09, "promotions": 3}`

*Builds on: exp-001-close-the-loop*

## exp-003-frontier-training-mix — REFUTED

**Lever(s) pulled:** mode of learning only: 50% of training episodes vs solver at the champion's frontier depth (frontier_opponent_fraction=0.5); all else identical to exp-002

**Expected:** If true, we expect the final champion's ladder score to be at least 0.09 above the incumbent's, concentrated in rung 1 (frontier depth should move past 1), because half the training signal now comes from exactly the opponent distribution the benchmark measures.

**Observed:** 50% frontier-solver training HURT: challenger ladder fell 0.030->0.012, zero promotions (gate correctly protected champion). Mechanism diagnosis: depth-1 solver (policy wins 18%) is not the zone of proximal development, it is far beyond ability - replay flooded with losses. Revisit when rung-1 score reaches ~0.5, or redefine the training opponent as one the policy beats ~50%.

**Key numbers:** `{"challenger_ladder_range": [0.012, 0.03], "diagnosis": "training opponent too strong: rung-1 win rate 0.18 at start", "final_champion_ladder": 0.042, "incumbent_ladder": 0.042, "predicted_min_gain": 0.09, "promotions": 0}`

*Builds on: exp-002-more-self-play*

## exp-004-both-players-transitions — INCONCLUSIVE

**Lever(s) pulled:** replay data only: store_opponent_transitions=True; nothing else

**Expected:** If true, we expect the final champion's ladder score to be at least 0.09 above the incumbent's (0.042), because twice the data with broader position coverage should at minimum match the precursor project, where both-player storage was part of every configuration that learned successfully.

**Observed:** Ladder 0.042->0.068 (+0.026, below 0.09 threshold), 2 promotions. Positive direction but noisy: challenger ladder peaked 0.077 mid-run then fell to 0.015 by iter 4 while h2h kept rising - relative and absolute strength are diverging again late in the run. Worth including in the big swing; alone, not proven.

**Key numbers:** `{"challenger_ladder_trajectory": [0.036, 0.068, 0.077, 0.024, 0.015], "final_ladder": 0.068, "incumbent_ladder": 0.042, "observed_gain": 0.026, "promotions": 2}`

*Builds on: exp-002-more-self-play, exp-003-frontier-training-mix*

## exp-005-mirror-augmentation — REFUTED

**Lever(s) pulled:** replay data only: mirror_augmentation=True; nothing else

**Expected:** If true, we expect the final champion's ladder score to be at least 0.09 above the incumbent's (0.068), because doubling effective data at this early training stage should convert directly into strength (the precursor used symmetric adds in every successful run).

**Observed:** Mirror augmentation HURT at this scale: challengers never reached h2h parity (0.41-0.48) and ladder fell to 0.006-0.030 vs champion 0.068; zero promotions, gate protected champion. Suspected mechanism: doubling perfectly-correlated samples in a small 20k buffer reduces batch diversity rather than adding information. Revisit with a larger buffer or when data is scarcer relative to capacity.

**Key numbers:** `{"challenger_ladder_range": [0.006, 0.03], "final_ladder": 0.068, "incumbent_ladder": 0.068, "promotions": 0}`

*Builds on: exp-004-both-players-transitions*

## exp-006-diverse-opponents — INCONCLUSIVE

**Lever(s) pulled:** mode of learning only: diverse_opponents=True (cycle champion/champion/tactic-fixture/random); nothing else

**Expected:** If true, we expect the final champion's ladder score to be at least 0.09 above the incumbent's (0.068), concentrated in rung 1, because the fixture supplies exactly the immediate-threat lessons the depth-1 solver grades, at a difficulty the policy can win against ~half the time (exp-003's zone-of-proximal-development lesson).

**Observed:** Ladder flat (0.068->0.071, +0.003) despite the strongest h2h finish of any experiment (0.65 over 200 games, promoted). Steady within-run h2h growth suggests the opponent mix teaches something real, but it is not converting to solver strength at 1500 episodes. Candidate for the big swing; alone, unproven.

**Key numbers:** `{"final_ladder": 0.071, "h2h_trajectory": [0.44, 0.5, 0.54, 0.54, 0.65], "incumbent_ladder": 0.068, "observed_gain": 0.003, "promotions": 1}`

*Builds on: exp-002-more-self-play, exp-003-frontier-training-mix*

## exp-007-terminal-quota — REFUTED

**Lever(s) pulled:** replay sampling only: terminal_fraction=0.3; nothing else

**Expected:** If true, we expect the final champion's ladder score to be at least 0.09 above the incumbent's (0.071), because tripling the frequency of reward-bearing samples should speed credit propagation from game ends back through the gamma-chain.

**Observed:** 30% terminal quota clearly hurt: h2h stuck at 0.36-0.39 (worst of the campaign), ladder 0.012-0.036, zero promotions. Suspected mechanism: forcing 30% terminals when they are ~10% of experience skews the sampling distribution - the net over-fits endgame values and under-trains midgame ones. The precursor used the quota WITH PER and 4x richer data; alone it distorts. Revisit only in combination, or with a lower quota (~15%).

**Key numbers:** `{"challenger_ladder_range": [0.012, 0.036], "final_ladder": 0.071, "h2h_range": [0.36, 0.39], "incumbent_ladder": 0.071, "promotions": 0}`

*Builds on: exp-004-both-players-transitions*

## exp-008-training-volume-3x — REFUTED

**Lever(s) pulled:** training volume + exploration schedule only: 15x300 episodes, eps_decay 0.999 -> 0.9995; nothing else

**Expected:** If true, we expect the final champion's ladder score to be at least 0.09 above the incumbent's (0.071), because the policy is nowhere near converged and volume is the variable with the strongest prior evidence behind it.

**Observed:** 3x volume (4500 episodes) with slower eps decay (0.9995): zero promotions in 15 iterations, h2h never above 0.48, ladder stuck 0.02-0.07. CAVEAT: two variables moved together (volume AND decay) - the slow decay keeps the challenger half-random for most of training, so this refutes the volume+slow-decay COMBINATION, not volume itself. exp-009 retests volume with the original 0.999 decay.

**Key numbers:** `{"confound": "volume and eps_decay changed together", "final_ladder": 0.071, "h2h_range": [0.34, 0.48], "incumbent_ladder": 0.071, "iterations": 15, "promotions": 0}`

*Builds on: exp-002-more-self-play*

## exp-009-big-swing — INCONCLUSIVE

**Lever(s) pulled:** package: store_opponent_transitions=True + diverse_opponents=True + 15x300 episodes at eps_decay 0.999 + buffer 50k

**Expected:** If true, we expect the final champion's ladder score to be at least 0.09 above the incumbent's (0.071), because the two data changes address the diagnosed data-poverty mechanism and volume gives them room to compound.

**Observed:** Package (both-players + diverse opponents + 4500 eps at 0.999 decay + 50k buffer): ladder 0.071->0.083 (+0.012, below 0.09), 5 promotions, best absolute score of the campaign but no breakout. Campaign conclusion: all data/volume/opponent levers combined move the ladder only ~+0.01-0.03 each; rung-1 ceiling ~0.42 persists. The bottleneck is likely NOT data quantity - prime suspects are reward density, network capacity, or reflex-only inference (no lookahead). These are the next single-variable hypotheses.

**Key numbers:** `{"campaign_ladder_path": [0.03, 0.042, 0.068, 0.071, 0.083], "final_ladder": 0.083, "incumbent_ladder": 0.071, "observed_gain": 0.012, "promotions": 5}`

*Builds on: exp-004-both-players-transitions, exp-006-diverse-opponents, exp-008-training-volume-3x, exp-005-mirror-augmentation, exp-007-terminal-quota*

## exp-010-capacity-probe — REFUTED

**Lever(s) pulled:** diagnostic only - no loop run; compares conv_layers/channels/hidden 2/32/128 vs 3/64/256 supervised on 4000 solver-d6-labeled positions

**Expected:** If capacity binds, the current net's TRAIN accuracy stays under 90% while the larger net reaches 90%+. If BOTH fit the training set (>=90%), capacity is NOT the binding constraint at this scale and the bottleneck is the reward signal (exp-011's territory). Held-out accuracy measures generalization separately; the capacity question is about TRAIN fit.

**Observed:** Capacity is NOT the bottleneck: the current 2conv-32ch net fit 3200 solver-labeled positions to 100% train accuracy (large 3conv-64ch net identical). Both generalize equally poorly (val 0.405 vs 0.425, within noise, chance ~0.15) - so more capacity buys nothing at this scale, and the plateau must come from the training signal / data distribution. Clears the way: exp-011 tests signal density; bigger-net loop experiments deprioritized.

**Key numbers:** `{"large": {"params": 765255, "train_acc": 1.0, "val_acc": 0.425}, "small": {"params": 182919, "train_acc": 1.0, "val_acc": 0.405}}`

*Builds on: exp-009-big-swing*

## exp-011-tactical-shaping — REFUTED

**Lever(s) pulled:** training signal only: tactical_shaping False -> True (shaping_penalty 0.5); all else = exp-002 baseline

**Expected:** If true, we expect the final champion's ladder aggregate to gain at least +0.09 over the incumbent's 0.083, concentrated in rung 1 (predict rung-1 >= 0.55 from 0.42), because the depth-1 solver wins almost exclusively by punishing exactly these two blunders, and they now carry immediate, dense, unmissable signal instead of sparse delayed signal. This also addresses the babies-teach-babies bootstrap: the shaping signal comes from the ENVIRONMENT's tactical facts, not from the (weak) opponent, so …

**Observed:** Shaping alone at 1500 episodes: no ladder gain (challengers 0.039-0.062 vs incumbent 0.083), h2h collapsed to 0.27-0.34. Suspected mechanism: warm-started Q-values were calibrated on pure win/lose scale; adding -0.5 penalties re-scales the value function and 1500 episodes is too short to recalibrate. NOT evidence shaping is useless - evidence it needs time and/or richer data to re-converge. Retested in exp-012 package with 3x episodes + both-players data + diverse opponents.

**Key numbers:** `{"challenger_ladder_range": [0.039, 0.062], "final_ladder": 0.083, "h2h_range": [0.27, 0.34], "incumbent_ladder": 0.083, "promotions": 0}`

*Builds on: exp-010-capacity-probe, exp-007-terminal-quota*

## exp-012-signal-package — INCONCLUSIVE

**Lever(s) pulled:** package, each from->to: tactical_shaping F->T (0.5), cold_start F->T, store_opponent_transitions F->T, diverse_opponents F->T, episodes 1500->4500, buffer 20000->50000; eps_decay unchanged 0.999

**Expected:** If true, we expect the final champion's ladder aggregate to gain at least +0.09 over the incumbent's 0.083, with rung-1 >= 0.60 (from 0.34), because the depth-1 solver punishes exactly the two blunders that now carry immediate signal, and a cold-started value function can absorb that signal without fighting a stale calibration. A FAILURE of this package - signal + data + time all present, capacity proven sufficient - would point the investigation at the remaining suspect: reflex-only inference (…

**Observed:** The package's COLD-STARTED policy climbed from random weights to ladder 0.116 (iter 11) in ~3600 episodes - the highest absolute score of the entire program, exceeding an incumbent that took ~10500 cumulative episodes to reach 0.083. The +0.033 margin is within noise (needs ~+0.09 at n=168) so no supported verdict, BUT: (1) fastest absolute learning curve observed, package merits single-variable decomposition (shaping+cold_start alone first); (2) HARNESS FINDING: the iter-11 challenger beat the incumbent on the benchmark yet was blocked by the RELATIVE gate (h2h 0.41) - the double gate has a reverse-v32 failure mode where absolutely-stronger-but-stylistically-different policies cannot promote. Gate design question for the human before changing anything.

**Key numbers:** `{"cold_start": true, "final_challenger_ladder": 0.077, "h2h_range": [0.3, 0.53], "incumbent_ladder": 0.083, "ladder_trajectory": [0.03, 0.024, 0.036, 0.03, 0.03, 0.036, 0.024, 0.03, 0.06, 0.036, 0.071, 0.116, 0.077, 0.083, 0.077], "peak_challenger_ladder": 0.116, "peak_iter": 11, "promotions": 0}`

*Builds on: exp-010-capacity-probe, exp-011-tactical-shaping, exp-004-both-players-transitions, exp-006-diverse-opponents*

## exp-013-shaping-cold-start — REFUTED

**Lever(s) pulled:** vs baseline, each from->to: tactical_shaping F->T (penalty 0.5), cold_start F->T, episodes 1500->4500; all exp-012 data levers returned to baseline (store_opponent_transitions F, diverse_opponents F, buffer 20000)

**Expected:** if reward density is the active ingredient, the best challenger reaches ladder >= 0.10 with rung-1 >= 0.50 within 15 iterations, reproducing most of the package's 0.116. If it stalls at <= 0.06, the data levers are jointly necessary and the next decomposition adds them back one at a time. Either result apportions the package's credit.

**Observed:** Shaping+cold_start ALONE is NOT the package's active ingredient: best ladder 0.015 (vs package's 0.116 at the same 4500-episode budget), h2h never above 0.31. Prediction threshold (<=0.06 => data levers jointly necessary) clearly met. Combined with exp-011 (shaping into warm start fails) and exp-012 (full package flies), the evidence says reward density only works WITH rich data (both-players transitions and/or diverse opponents). Next decomposition: package MINUS one lever at a time, starting with package-minus-diverse-opponents.

**Key numbers:** `{"best_ladder": 0.015, "episodes": 4500, "h2h_max": 0.31, "package_best": 0.116, "promotions": 0}`

*Builds on: exp-011-tactical-shaping, exp-012-signal-package, exp-010-capacity-probe*

## exp-014-shaping-cold-start-both-players — INCONCLUSIVE

**Lever(s) pulled:** vs exp-013, each from->to: store_opponent_transitions F->T, buffer_capacity 20000->50000 (composite 'both-players data' lever as configured in exp-012); everything else held at exp-013 values

**Expected:** if the recipe holds, best challenger ladder >= 0.08 (most of the package's 0.116, from half its levers) with rung-1 >= 0.40. If it stalls <= 0.04, diverse opponents are necessary and exp-015 adds them instead. Gate note: FIRST RUN under gate v2 (ADR-005, either-path promotion) — an absolutely-superior challenger at h2h parity can now promote. Plateau stop is armed (patience 6, min_delta 0.02): if six straight iterations buy nothing, the run ends itself and records the savings.

**Observed:** Both-players data QUADRUPLES shaping+cold_start (best ladder 0.060 vs exp-013's 0.015 at matched budget) but lands between the pre-registered thresholds (>=0.08 recipe-holds / <=0.04 stalled), so attribution is incomplete: both-players is a large contributor, not the whole recipe. First live plateau-stop: run ended itself at iter 7/15 (best stuck at 0.060 for 6 iters), saving 7 iterations of compute. Factorial completion next: exp-015 tests shaping+cold_start+diverse-opponents (no both-players) to measure the last lever's marginal contribution.

**Key numbers:** `{"best_ladder": 0.06, "exp013_reference": 0.015, "iterations_saved": 7, "package_reference": 0.116, "promotions": 0, "stopped_early_at_iter": 7}`

*Builds on: exp-013-shaping-cold-start, exp-012-signal-package, exp-004-both-players-transitions*

## exp-015-shaping-cold-start-diverse — REFUTED

**Lever(s) pulled:** vs exp-013, one from->to: diverse_opponents F->T; everything else held at exp-013 values (notably store_opponent_transitions F, buffer 20000)

**Expected:** interpreting the 2x2 against exp-013's 0.015 floor and exp-012's 0.116 ceiling — (a) if diverse opponents alone reach >= 0.08: diversity is the primary lever and both-players data was the helper; (b) if 0.03-0.08 (like exp-014's 0.060): the levers contribute comparably and roughly additively; the package's edge is their sum; (c) if <= 0.03: diversity NEEDS both-players data (the fixture/random games only pay off when the opponent's winning moves also enter the buffer) — a superadditive interacti…

**Observed:** Diverse opponents WITHOUT both-players data add ~nothing: 0.018 vs the 0.015 exp-013 floor (plateau-stopped at iter 6/15, saved 8 iterations). Pre-registered outcome (c): the levers are SUPERADDITIVE — 2x2 complete: pair=0.015, +both-players=0.060, +diverse=0.018, both levers together (exp-012)=0.116. Interpretation: fixture/random opponents create tactical situations, but the lesson only enters the buffer when the OPPONENT'S winning reply is also stored — diversity manufactures the lesson, both-players data records it. The exp-012 package is the minimal known recipe. Next: run the recipe long under gate v2 to promote a real successor champion and find the recipe's ceiling.

**Key numbers:** `{"best_ladder": 0.018, "exp013_floor": 0.015, "exp014_both_players": 0.06, "iterations_saved": 8, "package_ceiling": 0.116, "stopped_early_at_iter": 6}`

*Builds on: exp-014-shaping-cold-start-both-players, exp-013-shaping-cold-start, exp-012-signal-package, exp-006-diverse-opponents*

## exp-016-recipe-ceiling — PENDING

**Lever(s) pulled:** vs exp-012, each from->to: iterations 15->30 (plateau-stop armed), gate_rule double->either (ADR-005); recipe itself unchanged

**Expected:** (a) best challenger ladder >= 0.15 (clears exp-012's 0.116 by more than one detectable effect at n=168), and (b) at least one promotion occurs under gate v2. If the curve plateau-stops at ~0.116 instead, the recipe is signal-saturated and the next bottleneck is the one exp-009 named and no data lever can fix: reflex-only inference — the policy answers in one forward pass with no lookahead — pointing the program at search (ADR-003 direction).

**Observed:** (no verdict recorded yet)

*Builds on: exp-015-shaping-cold-start-diverse, exp-014-shaping-cold-start-both-players, exp-013-shaping-cold-start, exp-012-signal-package*
