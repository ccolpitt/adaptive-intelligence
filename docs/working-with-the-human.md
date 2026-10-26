# How to reason, and how to report

Written for whoever works on this project next, agent or person. It exists
because the same two failures kept recurring across sessions, and both are
failures of *communication and reasoning discipline* rather than of code.

---

## Part 1 — Anchor every claim in a base principle

Before asserting anything about the system, name the principle that makes the
claim true or false. Not the observation — the principle. The observation then
either follows from it or contradicts it, and both are useful.

Three defects in this project were found this way and three were missed for
want of it:

- "A budget of 200 examined 400 positions" is only a defect because of the
  principle *the budget is the compute currency that makes two policies
  comparable*. Without naming that, it reads as a rounding problem, and it was
  papered over with a 5% tolerance (R-006).
- "A terminal node costs a unit of budget" is only wrong because of the
  principle *the budget buys looks by the network, and the rules are free*.
  The code had noticed the fact and not the principle, so the charge stayed
  for 29 experiments (R-008).
- "Resuming restores the weights" sounds complete until you name the principle
  *a learner is its weights plus its optimiser state plus the data it is
  learning from*. The code even documented the shortcut in a comment and moved
  on. It cost 27 cycles (R-007).

The standing form: **state the invariant, then the measurement, then whether
they agree.** If you cannot state the invariant, you are not yet in a position
to say whether the number is wrong.

A corollary worth its own line: **if a comment explains why a shortcut is
acceptable, the comment is evidence the shortcut has a cost.** Measure the
cost or remove the shortcut. Three of eight retros are this exact move —
noticing, annotating, and not following through.

---

## Part 2 — Lead with the so-what

Every finding is reported in this order:

1. **What it means for the decision in front of us.** One sentence.
2. **The number, and how it was obtained.**
3. **The mechanism**, if it is needed to act.

Not the reverse. A report that walks through the search's node accounting and
arrives at an implication three paragraphs later has buried the only part that
changes what anyone does.

Worked example. The wrong shape:

> In a win-in-one position, 187 of 199 visits go to re-confirming the same
> terminal win. The cause is in `BudgetedSearch.run`: the terminal branch does
> `nodes_evaluated += 1` on every re-visit.

That is true, and the human's reply was the correct one: *is that good or bad?*
The passage never says. The right shape:

> **The search is far weaker than its rubric claims, in exactly the positions
> that decide games — this is a bug and I will fix it.** A declared budget of
> 200 positions buys 14 in a win-in-one position, and a search two plies deep
> instead of eight. Terminal positions are being charged a unit of budget on
> every re-visit, though the rules give their value for free.

Same facts, decision first.

### Separate "this is how it works" from "this is wrong"

A description of mechanism is not a judgement, and a reader cannot tell which
one you are making unless you say. When describing something *working as
intended*, say so. When describing a defect, say that, and say whether you are
about to fix it or are asking.

The re-visit question is the clean illustration of why this matters, because
the honest answer is *both*: re-visiting is good — a lopsided visit count is
exactly the training target the move head needs — and charging for it is bad.
Reporting only the second half invited the reasonable inference that
re-visiting should be reduced, which would have broken the training signal.

---

## Part 3 — Explain the jargon, always

Spell out the term the first time it appears in a session, in one clause. The
human has repeatedly and correctly asked for this. Terms in this project that
always need unpacking:

| Term | The one-clause version |
|---|---|
| gradient norm | how large a correction the loss asked for, before clipping |
| weight delta | how far the weights actually moved, which clipping makes different from the above |
| replay buffer | here, a sliding window of training examples — *not* bootstrapped replay, since the targets are fixed when collected |
| visit count | how often the search walked through a move; the training target, and a measure of confidence |
| budget | distinct positions the network may look at for one move |
| PUCT | pick the move maximising value plus an exploration bonus weighted by the network's prior |
| negamax | one player's win is the other's loss, so flip the sign at every level |
| prior | the move head's opinion before any searching |
| rung | one fixed-depth solver opponent on the benchmark ladder |
| frontier depth | the shallowest rung not yet beaten 90% of the time |
| bare / naked score | the policy reading the position once, with no search — the score of record |

Avoid nicknames entirely in hypotheses and verdicts. "L5" is acceptable in
conversation but a hypothesis says "the fifth line of policies, the first with
two heads."

---

## Part 4 — Say what you did not check

State what was verified and what was assumed. Both halves. "I read the code
and the signs are consistent" is weaker than "I ran an untrained network with
flat priors and it still found the winning move 187 times out of 199, which
can only happen if the terminal sign is right" — and much weaker than it
sounds if the reader thinks you ran something.

When a claim rests on few samples, say how few. Two deterministic games were
reported in this project as evidence that the benchmark's rungs were
intransitive; a proper round-robin showed they are ordered. The error was not
the measurement, it was presenting a two-sample result without its sample size.

---

## Part 5 — Correct yourself plainly and move on

State the correction, state what it changes, continue. No apology, no
re-litigating, no tally. If an earlier claim would have changed a decision,
lead with the correction; if it changes nothing, fix it quietly.

---

## Part 6 — The human's standing preferences

Gathered from across sessions, so they do not have to be restated:

- **Results over process purity**, when the two conflict. Say which you are
  trading.
- **Keep a run going if it can still teach something.** Stopping a run that is
  hours into collecting evidence is the human's decision, with the reasoning
  laid out. See learning rule 12.
- **Bugs found are welcome.** Write the retro.
- **Complete sentences, few hyphens, no jargon unexplained.** Spell out
  from-value and to-value for every change.
- **Store data, render pictures on demand.** Never commit a chart.
- **Do not be precious about existing code.** A rewrite is acceptable if it is
  the right answer.
