# Learning Rules

The rules the training system operates under. **Revisable**: each rule says
why it exists and what would change it. When a rule changes, note the date and
the reason, and record an ADR if it affects how results are compared.

Vocabulary used here, with no nicknames:

- **task** — something we want a policy to be able to do ("win Connect 4").
- **evaluation definition** — the versioned, fixed specification of what
  success at a task means. Lives in the task repository.
- **cycle** — one complete round of: train for a set number of games,
  evaluate, then decide whether the new policy replaces the champion. The
  unit of the learning arc.
- **champion** — the policy currently considered best for a task.
- **challenger** — the policy being trained, which may replace the champion.
- **lookahead** — examining the consequences of candidate moves before
  choosing one, using a model of how the environment responds.
- **budget** — how much the policy may examine before committing to a move,
  counted in states examined.

---

## 1. Evaluation is always the naked policy

The score of record is produced by the policy alone: it reads the position
once and commits to a move. No lookahead, no search.

**Why.** The project measures how fast a policy *learns*, not how much
compute it can be given at test time. A score that includes search measures
the search. Keeping evaluation naked also means every score ever recorded
stays comparable with every future one.

**Status: in force** (ADR-006). Search-assisted scores may be reported
alongside as diagnostics, clearly labelled, never as the score of record.

## 2. Training may use lookahead; the budget is a declared variable

During training the policy may look ahead as much as a hypothesis calls for.
The lookahead budget is a recorded parameter of the cycle, like the learning
rate.

**Why.** Looking ahead while practising and playing instantly when judged is
how deliberate practice produces intuition. It is also the only improvement
mechanism known to work on this class of task: searching over a policy is
stronger than the policy alone, so games played with search are better than
the policy that generated them, and training the policy to reproduce what the
search concluded makes the next generation's instant move as good as this
generation's considered move. Repeating that is the engine.

**Status: in force.** Nothing prohibits a hypothesis that trains without
lookahead; the budget is simply always recorded.

## 3. The lookahead budget is a number of states, not a depth

A budget of 200 means the policy may examine at most 200 states before
committing to a move. It chooses which 200.

**Why.** A depth budget spends itself uniformly: looking three moves ahead in
a seven-column game examines about 400 states whether or not any of them
matter. A state budget lets the policy abandon a line after one look and
follow another twenty moves deep, with its own value estimates deciding
where to spend. Measured on the strongest archived Connect 4 policy
(exp-023), at matched cost:

| how the move is chosen | ladder score | states examined per move | depth reached |
|---|---|---|---|
| naked policy | 0.116 | 1 | 0 |
| depth-limited, 2 moves | 0.188 | 478 | 2 |
| depth-limited, 3 moves | 0.271 | 3,216 | 3 |
| budgeted, 50 states | 0.220 | 50 | 11 |
| budgeted, 200 states | 0.244 | 200 | 16 |
| budgeted, 400 states | 0.339 | 400 | 16 |

50 states, chosen well, beat 478 states spent uniformly. 400 beat 3,216.

**Why this is not brute force.** The policy's own estimates choose the next
state to examine; it is biased sampling of a tree it mostly never looks at,
not enumeration. A budget of 200 states in a game with roughly 4.5 × 10¹²
reachable positions is a keyhole. Deciding where to point the keyhole is the
reasoning.

**Why a value estimate is still required.** Search without a value estimate
must reach the end of the game to learn anything, which is impossible from
the opening of any game bigger than noughts and crosses. The value estimate
is what lets a shallow search be strong: the policy looks a little way ahead
and *judges* the positions it arrives at. Better judgment, less search
needed. This is why search and a learned value function are complements, not
substitutes.

**Status: in force** for new work. Depth-limited search is retained only to
reproduce earlier experiments.

## 4. A comparison is only valid at an equal budget

When a challenger plays the champion, both get the same budget. When a policy
is scored on a task, the budget is part of the evaluation definition.

**Why.** Otherwise a comparison measures compute, not skill.

**Status: in force.**

## 5. Tasks with no rulebook: a learned model, trusted only as far as earned

Connect 4 lookahead is exact — the rules are known, and a predicted position
is certain. Almost nothing else is like that. "Book a flight" has no rulebook
to search.

The path for those tasks, built into the search interface from the start:

1. **The model is an interface, not an assumption.** The search asks only
   "what happens if I do this?" A rulebook and a learned model answer the
   same question.
2. **Every predicted transition reports a confidence.** Exact rules report
   full confidence. A learned model reports less.
3. **Confidence compounds along a line and discounts the value found there.**
   Ten speculative steps through a mediocre model yield a conclusion the
   search barely weights. An unreliable model is therefore searched shallowly
   without being told to — exactly the right response to the gap between a
   simulation and reality.
4. **Model error is measured, not assumed.** A learned model is scored on
   held-out transitions, and that accuracy is a reported metric alongside
   task performance. The project's mechanism hypothesis is that learning the
   model and learning to act, in parallel, is what makes new tasks cheap —
   so model accuracy is a first-class result, not an implementation detail.

**Status: interface in force; learned models not yet built.** When the first
task without a rulebook arrives, this is the path.

## 6. Promotion requires winning on both measures

A challenger replaces the champion only if it beats the champion directly
**and** scores higher on the task's evaluation definition.

**Why.** Beating the champion alone is not evidence of skill — a policy can
learn one opponent's habits while being worse at the task. Allowing promotion
on the direct comparison alone, with the absolute score merely "not provably
worse", let the champion decline from 0.083 to 0.042 across two promotions:
each step was individually forgivable and the sequence was not.

**Status: in force** (ADR-007). A challenger that reaches parity is recorded
as a signal that the change is probably sound and under-dosed, not as a
failure.

## 7. Three seeds for any effect smaller than the spread between seeds

A seed sets every random choice: starting weights, which experiences are
replayed, exploration, the random opponent. The same configuration at two
seeds produced 0.054 and 0.116. Any claimed effect smaller than that spread
needs at least three seeds and is judged on the distribution.

**Status: in force**, with one deliberate exception: while establishing
whether the system can improve at all, a single seed may be used to move
fast, and the result is labelled provisional until replicated.

## 8. The training budget is declared and recorded

Games per cycle is a variable, not a constant. It may be the subject of a
hypothesis, and it may change as the champion gets stronger or the network
gets larger.

**Why.** Samples are the currency the central hypothesis is denominated in,
so the number spent is part of every result.

**Status: in force.**

## 9. Store data; render on demand

Cycle records are JSON. Charts are generated by a renderer, versioned and
bound to a task, named in each cycle record. A chart is never the record.

**Why.** A stored chart cannot be re-examined with a question it was not
drawn to answer. Data plus a versioned renderer can be. When a hypothesis
needs a new measurement, the renderer version is created *before* the cycle
runs, so the record and the way to read it always agree.

**Status: in force.**

## 10. Register the hypothesis before the run; record the verdict once

Unchanged from the start of the project, and the reason the ledger is worth
anything. See `docs/hypothesis-template.md`.

**Status: in force.**

---

## Open questions

- **How much lookahead at evaluation, eventually?** Rule 1 fixes the score of
  record at zero. But a policy that may think before moving is the more
  faithful model of reasoning, and a budget is a fair, declarable condition.
  The planned resolution is empirical: measure how policy improvement varies
  with the training and evaluation budgets, then decide with data. Until then
  the score of record stays naked so the measuring stick does not move.
- **How should the budget be spent within a move?** Currently a prior-guided
  best-first rule. Alternatives worth testing: widening before deepening,
  spending more on positions the value function is uncertain about, and
  stopping early when the best move is already clear — the last being closest
  to how a person stops thinking once a move is obviously right.
