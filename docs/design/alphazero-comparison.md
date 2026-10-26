# Where we match AlphaZero, where we deviate, and what to change

Written because "replicate the best version of AlphaZero" is a concrete,
checkable request, and because several items on the hypothesis queue are
**deviations** from AlphaZero rather than steps toward it. Confusing the two
would make it impossible to say whether a disappointing result came from our
changes or from not having implemented the baseline.

---

## First, the vocabulary that caused the confusion

**A simulation** is one complete pass through the search loop:

1. **select** — walk down the existing tree, at each node taking the move
   that maximises value plus an exploration bonus (PUCT);
2. **expand** — add the node you arrived at;
3. **evaluate** — estimate how good that new position is;
4. **back up** — send that estimate back up the path you came down,
   negating it at every level because one player's gain is the other's loss.

"1,600 simulations per move" means that loop ran 1,600 times.

**A rollout** is one *option for step 3 only*: play the game out to the end
from the new position, usually with a fast, weak policy, and use the actual
result as the estimate. That is "Monte Carlo" in the original sense —
estimate a value by sampling a complete outcome.

So a rollout is not an alternative to a simulation; it is a component that
may or may not sit *inside* each simulation. The reason the words feel
interchangeable is historical: in classical Monte Carlo Tree Search every
simulation contained a rollout, so nobody had to distinguish them.

What changed across the three papers:

| | evaluation step | simulations per move |
|---|---|---|
| AlphaGo (2016) | a blend: a rollout with a fast policy network, mixed with a value network | 1,600 |
| AlphaGo Zero (2017) | the value head alone — **rollouts dropped** | 1,600 |
| AlphaZero (2018) | the value head alone | 800 for self-play |

AlphaGo Zero dropped the *rollout*, not the tree search. AlphaZero
additionally dropped the *evaluator* — the gate that compared a new network
against the best current one before adopting it — and simply kept training one
network continuously.

**Our search has never used rollouts.** Each pass through
`BudgetedSearch.run`'s loop is a simulation in exactly the sense above, and
step 3 is one forward pass of the value head. On this axis we already match
AlphaGo Zero and AlphaZero.

---

## One place our units genuinely differ

AlphaZero counts **simulations**. We count **distinct positions the network
evaluated**, which we call the budget.

In Go those two numbers nearly coincide: each simulation expands exactly one
new node and evaluates it once, and repeated positions are rare. In Connect 4
they come apart badly, because transpositions are everywhere and terminal
positions are common. After the R-008 fix a position is charged once per move
however often the search returns to it, so a declared budget of 200 means
about 200 network evaluations — but the number of simulations can be far
larger (4,000 in a position with a win one move away).

Our currency is arguably the better one for comparing two policies fairly,
since it measures the expensive resource. But it means **our "200" is not
comparable to AlphaZero's "800"**, and in network calls we are running at
roughly a quarter of their self-play setting.

---

## Point by point

### Where we already match

| | AlphaZero | us |
|---|---|---|
| no rollouts; value head evaluates | yes | yes |
| selection rule | PUCT | PUCT |
| move played in self-play | sampled from visit counts early, most-visited later | same |
| move target for the policy head | the search's visit distribution | same |
| value target | the final game outcome, undiscounted | same |
| loss | cross-entropy on moves + squared error on value + weight decay | same |
| root exploration noise | Dirichlet, ε = 0.25 | Dirichlet, ε = 0.25 |
| symmetry augmentation | all symmetries of the board | the single left-right mirror, which is all Connect 4 has |

### Where we fall short of the baseline

These are **not** improvements on AlphaZero. They are places we have not yet
implemented it, and they should be closed before any deviation is credited.

1. **Simulations per move: 800 → we use 200.** The single cheapest change
   with a known-good target value. The visit distribution *is* the policy
   head's training target, so a small budget gives a coarse target: with
   seven columns and 200 visits the distribution is reasonably expressed, but
   at 32 it is roughly four visits per column, which is why the early
   lineages' targets were so blunt. Now is the moment to raise it, because
   R-008 just made the budget mean what it says.
2. **Games per iteration: 25,000 → we use 200 per cycle.** Two orders of
   magnitude. This is the honest explanation for most of our value-target
   noise, and it is not fixable on one machine. It is also the reason several
   deviations below are justified: techniques that reduce variance matter far
   more at 200 games than at 25,000.
3. **Network: 20–40 residual blocks at 256 channels → we use 3 plain
   convolutional layers at 64 channels, with the heads flattening a 2,688-wide
   tensor into a 128-unit bottleneck.** AlphaZero's heads apply a 1×1
   convolution first, which preserves the board's layout. Queue item 0g.
4. **Dirichlet α: their rule of thumb is roughly 10 / (typical legal moves),
   giving ≈ 1.4 for Connect 4's seven columns → we use 1.0.** Close, and
   never tuned.
5. **Optimiser: stochastic gradient descent with momentum and a stepped
   learning-rate schedule → we use Adam at a fixed rate.** Probably minor,
   and unexamined.
6. **No gate.** AlphaZero adopts every new network continuously; we require a
   challenger to beat the champion *and* improve the benchmark. Ours is
   currently blocking everything — 27 consecutive cycles with no promotion
   while the challenger was beating the champion head to head. This deserves
   its own experiment, and it is the one place where following AlphaZero would
   simplify the system rather than complicate it.

### Where we deliberately deviate, and why

Each of these is a *departure* from AlphaZero. Several are justified by item 2
above — at 200 games per cycle we are variance-limited in a way AlphaZero
never was — but they must be labelled, measured against the baseline, and not
mistaken for implementing it.

| deviation | queue | why we would do it anyway |
|---|---|---|
| discount the value target by distance from the end | 0c | AlphaZero's undiscounted outcome is unbiased but very high variance; at 200 games per cycle the variance costs more than the bias |
| weight the move-head loss toward decisive positions | 0d | decisive positions are 13% of our data and AlphaZero had 125× more games to average over |
| prioritized replay on "hot" positions | 0e | same reasoning, applied to sampling rather than weighting; riskier, since it also distorts what the value head sees |
| first-play urgency for unvisited children | 0h | AlphaZero's pseudocode initialises an unvisited child's value to 0; later engines (KataGo, Leela) found a reduction from the parent's value better. A deviation from the paper, but toward the current state of the art |
| evaluate the policy with no search at all | learning rule 1 | AlphaZero never reported a searchless score. This is our own, harder standard, chosen so the yardstick cannot move |

That last row is worth dwelling on. AlphaZero's published strength is always
network **plus** search. In the AlphaGo Zero paper the raw policy network
without search was around 3,000 Elo against roughly 5,200 with it. So "we
should be able to win without search because AlphaZero could" is not
supported — they could not either. Our naked-evaluation rule is a deliberate
extra constraint.

---

## What to change, in order

1. **Raise the search budget from 200 to 800 distinct positions per move.**
   One lever, a known-good target, and it directly sharpens the policy head's
   training target. Expect roughly four times the wall-clock cost per game, so
   halve the cycles or accept a longer run. Predict: tactical rates rise and
   the deep rungs improve, because the search can finally see two- and
   three-move combinations after R-008.
2. **Test continuous adoption against the strict gate.** AlphaZero has no
   gate; ours has produced no promotion in 27 cycles while the challenger was
   winning head to head. Run both arms from the same checkpoint. Note that
   dropping the gate does not weaken the record: the benchmark score of every
   cycle is still measured and stored, so progress stays auditable without a
   promotion decision.
3. **The 1×1 convolutional heads** (queue 0g, arm A). The measured geometry
   gap is independent evidence for this: blocking a vertical threat is
   near-perfect while blocking a diagonal is much worse, and a diagonal block
   is exactly the pattern whose spatial arrangement the flatten-to-128
   bottleneck destroys.
4. Only then the deviations, one at a time, each measured against the
   baseline established by 1–3.

## What not to replicate

5,000 first-generation TPUs, 700,000 training steps, and 25,000 self-play
games per iteration. The interesting question for this project is what
survives at four orders of magnitude less compute — which is why the
variance-reduction deviations are worth testing at all.

---

## References

Silver et al., *Mastering the game of Go with deep neural networks and tree
search*, Nature 529, 2016 — AlphaGo, rollouts blended with a value network.

Silver et al., *Mastering the game of Go without human knowledge*, Nature 550,
2017 — AlphaGo Zero, rollouts dropped, 1,600 simulations per move, policy and
value heads on a shared residual trunk, gated by an evaluator.

Silver et al., *A general reinforcement learning algorithm that masters chess,
shogi and Go through self-play*, Science 362, 2018 — AlphaZero, the evaluator
dropped, 800 simulations per move for self-play.

Rosin, *Multi-armed bandits with episode context*, Annals of Mathematics and
Artificial Intelligence 61, 2011 — the PUCT rule the selection step uses.

Schaul et al., *Prioritized experience replay*, ICLR 2016 — the basis for
queue item 0e.
