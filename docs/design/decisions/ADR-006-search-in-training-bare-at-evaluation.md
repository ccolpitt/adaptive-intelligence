# ADR-006: Search During Training, Bare Inference at Evaluation

**Status:** Accepted (2026-10-03) — human rule: "When evaluating policies, we
only run inference on the naked policy. No MCTS. Record this; we can change
it later. During training, we can do whatever we want."

## Context

exp-020 showed the unchanged champion network scores 0.071 on the solver
ladder when it picks moves in one forward pass, and 0.310 when wrapped in a
3-ply negamax over real rules with the network valuing leaves. Zero training.
That raised a definitional question: is "the policy" the network, or the
network plus search? The answer determines what the benchmark measures, what
promotion judges, and whether 20 experiments of history remain comparable.

Background facts weighed: AlphaGo/AlphaZero used MCTS in BOTH training and
play (AlphaGo Zero's raw network ≈ 3,000 Elo vs ≈ 5,000 with search); the
raw network got strong because it was trained to imitate the search (expert
iteration, Anthony et al. 2017). Humans calculate a few moves ahead AND read
positions by gestalt; deliberate practice with calculation builds the gestalt.

## Decision

1. **The policy is the network.** Archived artifacts stay networks; the
   benchmark and head-to-head remain bare one-glance inference on the fixed
   168 ladder games. The ruler is unchanged; all history stays comparable.
2. **Search is a training instrument.** `TrainerConfig.train_search_depth`
   (k-ply lookahead picks self-play moves) and `search_targets` (search-backed
   values become the training targets — distillation). The search depth is
   the legislated "thinking budget", recorded in every experiment's
   conditions; it is a lever like any other.
3. **Search-amplified scores are diagnostics**, reportable on cards ("what
   this network would score with k-ply search"), never gating.
4. **Fixed training budget as a standing rule:** samples are the currency;
   each experiment declares its game budget (default 15–30 iterations × 300
   games, plateau-stop may end it early) and is scored on what it reaches
   inside it.

## Why

- Measures what the program cares about: what the network *internalized*,
  not what it can *compute* at test time. The research question is learning
  speed, not raw strength.
- Mirrors the one mechanism proven to produce strong raw networks.
- Costs nothing in comparability; exp-021 needs no rule change at all.

## Revisit when

- exp-021 shows search knowledge does NOT transfer into one-glance weights
  (both arms ≤ baseline): then inference-time search with a legislated
  budget becomes the live option, and the ladder would gain a second,
  separately-recorded score column.
- A future task family has no cheap simulator for the search to use — then
  the learned world model (vision §3) must supply it, which is the intended
  long-term path anyway.
