# Defect Retros

One entry per defect that reached a run. Each says what broke, how it was
caught, what would have caught it sooner, and what changed so the class of
defect cannot recur quietly.

The pattern across every entry so far: **the defect did not crash.** The run
completed, the numbers looked plausible, and only a number that was
*impossible* gave it away. That is the expensive kind, because results get
interpreted before anyone notices. So the standing response is not "be more
careful" — it is to make each class of defect either crash or be flagged
automatically.

---

## R-001 — A rubric setting never reached the trainer (exp-025)

**What broke.** The rubric for the third lineage specified that value
estimates be bounded by the architecture. The function translating a rubric
into trainer settings did not read that field, so the network trained
unbounded for 40 cycles and 16,000 games. The run's headline change never
happened.

**How it was caught.** By a number that cannot be true: the largest value
estimate was 2.428, and a squashed output cannot exceed 1. Verified
immediately — forcing a +50 bias into a squashed output still gives exactly
1.0.

**Cost.** 16,000 games on an untested hypothesis. Partly recovered, because
the run produced the program's largest result for other reasons, but the
registered claim was simply not tested.

**What would have caught it sooner.** Anything comparing what the rubric
promised against what the trainer received.

**Changed.** The translation now *refuses to run* when a rubric specifies
anything that does not reach the trainer. A rubric is a promise about how a
policy was trained; a setting that silently fails to apply makes that promise
false, so this is now a crash rather than a quiet divergence.

---

## R-002 — A value-range rule applied to numbers that are not values (exp-027)

**What broke.** The newly added health conditions withheld a promotion that
both score gates had passed, because the move head's score reached 4.46
against a limit of 3.0. But a two-headed policy's move head outputs scores
that become probabilities by softmax — they are not value estimates and are
not confined to the reward range. A large one simply means a strong
preference. Left alone, this would have blocked every promotion in the
lineage.

**How it was caught.** Reading the gate's stated reason on the first cycle.
The reason string was explicit enough to make the category error obvious,
which is the one thing that went right here.

**Cost.** One cycle, eight minutes.

**Root cause, honestly.** I added the health check while the only learner had
one set of outputs, where magnitudes *were* value estimates. When the
two-headed learner arrived, the check was not revisited. The check was
correct for the system it was written against and wrong for the system it
ran against.

**Changed.** The value range is now checked against the head that estimates
value, and move scores are recorded separately under their own name. More
generally: a check that assumes a property of the learner must name that
assumption, so that adding a learner forces it to be re-examined.

---

## R-003 — A promotion path assuming a learner's internals (exp-028)

**What broke.** On promotion, the loop raised the exploration rate to
re-explore against the new champion. The two-headed learner has no
exploration rate — it explores through noise in its search and by sampling
its move from the search's attention — so the first promotion crashed the
run.

**How it was caught.** A crash, which is the cheap kind. It cost one cycle.

**Root cause.** Same shape as R-002: code written against one learner,
inherited by another. The smoke test that was supposed to cover the
integration never promoted, so the promotion branch was never executed.

**Changed.** The exploration bump is conditional on the learner having one.
The broader lesson recorded here: a smoke test that does not reach the
branch where learners differ has not tested the integration. Integration
tests must force the paths that differ — promotion, rejection, and resume —
not merely run a cycle.

---

## R-004 — Move selection did not match the registered hypothesis (exp-028)

**What broke.** The hypothesis on record said the move played would be "the
most attended" one. The implementation took the highest backed-up value
instead. These are different rules, and the one in the literature is visit
count: AlphaZero draws the move from the visit counts and drives the
temperature to zero at play time, which selects the most-visited move. The
backed-up value appears only inside the rule for choosing where to search
next.

**How it was caught.** The human questioned the claim and asked me to check
the paper. I had asserted the wrong thing confidently, and checking showed
the hypothesis text was right and my code was not.

**Cost.** One cycle, plus a confidently incorrect explanation that would have
misled the project had it gone unchallenged.

**Root cause.** I described our implementation using the vocabulary of the
literature without verifying that they matched.

**Changed.** Move selection is now explicit and configurable, visit count is
used by the two-headed learner as the literature does, and the previous
behaviour remains available so already-recorded experiments stay
reproducible. Standing rule: when a hypothesis claims to follow a published
method, the specific mechanism is verified against the source before the run,
not after.

---

## R-005 — Registration blocked the resume it was meant to support (exp-028)

**What broke.** Runs were made resumable, but each experiment script
registered its hypothesis unconditionally, and the registry correctly refuses
to re-register. So resuming — which means running the same script again —
crashed immediately.

**How it was caught.** Immediately, on the first resume attempt.

**Root cause.** A feature was added (resume) without following through on
what it implied for everything it touched.

**Changed.** Registration is idempotent: it registers the first time, and on
later runs verifies the hypothesis on disk still matches the script. If the
wording has drifted it stops, because continuing would attach new results to
a hypothesis that was not the one being tested.

---

## What these changed about the process

1. **An impossible number must crash or be flagged, never be printed
   calmly.** `harness/sanity.py` now checks each cycle against stated
   expectations and separates three cases: a contradiction (cannot be true
   given how the system is built — always a defect), an inconsistency (two
   measurements disagree and need explaining), and the merely unexpected.
   Each check cites the defect that motivated it and says where to look.
2. **A contradiction halts interpretation, not just the cycle.** If a number
   in a cycle cannot be true, every other number in that cycle is suspect
   until it is explained. Recorded as learning rule 11.
3. **A check that assumes something about the learner must say so**, so that
   adding a learner forces the assumption to be re-examined.
4. **Integration tests must force the paths where components differ** —
   promotion, rejection, resume — not merely complete a cycle.
5. **A claim to follow a published method is verified against the source
   before the run.**

---

## R-006 — A check that could not state its own units (exp-029)

**What broke.** The expectation check for the lookahead budget compared
positions examined per **game** against the budget stated per **move**. The
first cycle it ran on reported a contradiction: 8,400 positions against a
budget of 200. Across a 42-move game, 8,400 positions is exactly 200 per
move — precisely on budget. The search was behaving correctly and the check
was wrong.

**How it was caught.** The human asked how to tell whether a run is in
progress. Improving the status command to answer that surfaced the recorded
contradiction, and learning rule 11 required investigating it before trusting
any number from the cycle. Ten minutes of investigation, no wasted compute.

**Cost.** Nearly none in compute, but the cost of the class is high: a false
contradiction sends the reader hunting a defect that does not exist, and
rule 11 says a contradiction invalidates interpretation of the whole cycle.
A check that cries wolf is worse than no check, because it spends the
attention the real ones need and teaches the reader to discount them.

**Root cause, and the part worth keeping.** I noticed the unit mismatch while
writing the check and covered it with a tolerance — the comment read "a
little slack for per-game accumulation" — instead of fixing it. The slack was
5%, and the true mismatch was a factor of forty. A tolerance was applied to a
problem that was not about tolerance at all. **Papering over an inconsistency
I had already half-seen is the actual defect here**, and it is the one most
likely to recur, because it felt at the time like pragmatism.

**Changed.**
- The check now divides by the moves in that game, and the number of moves
  per game is recorded in every cycle so the division is possible.
- When the denominator is absent — records written before it was captured —
  the check reports that the budget **could not be verified**, rather than
  passing silently or claiming a violation. A check that cannot run should
  say so; silence is indistinguishable from success.
- Tests cover all three cases: a genuine overspend fires, an on-budget cycle
  stays clean, and a record missing the denominator reports that it cannot
  be checked.

**Standing rule added.** A check that compares two measurements must state
the units of both in its own text. If a tolerance is being used to reconcile
them, that is a sign the units differ and the comparison is wrong — not a
sign the tolerance needs widening.

---
## R-007 — A restart that looked like a plateau (exp-029)
**What broke.** exp-029 continued lineage L5 from where exp-028 finished. Its
registered hypothesis said so explicitly: "the only change is the number of
cycles, from 60 to 160, continuing the SAME line of policies." But a resume
restored only the network's weights. The optimiser's per-weight averages,
accumulated over tens of thousands of steps, and the replay buffer holding
the games those steps learned from, both restarted from nothing.

The effect was immediate and large. The last cycle of exp-028 scored 0.622;
the first cycle of exp-029, trained from exactly those weights under exactly
that recipe, scored 0.390, and every rung of the benchmark fell together.
Twenty-seven cycles — 5,400 games, about four and a half hours — brought it
back to 0.592, still short of where it had started.

**How it was caught.** Not by any instrument. The run reported no
contradiction that was real; its 27 recorded contradictions were all the
R-006 false positive, still being written because the process had loaded
`sanity.py` before that fix. The drop was found by hand, trending the cycles
either side of the boundary while answering the question "is this run worth
continuing?". The corroborating evidence was in the series all along:
training error fell from 1.303 to 1.101 and the gradient norm from 0.978 to
0.703 at the boundary, then both climbed steadily back as the buffer refilled
— the signature of a small, highly correlated buffer being easy to fit.

**Cost.** About 5,400 games of compute, and worse, a confounded experiment:
the hypothesis under test was "more of the same recipe keeps improving," and
what actually ran was "the same recipe, plus an unregistered cold restart."
Had the run been read without this investigation, the honest-looking
conclusion would have been that the recipe had plateaued around 0.55 — the
opposite of what the data says once the restart is accounted for. The policy
was in fact improving throughout: taking an available win went from 0.65 to
0.91 and blocking a forced loss from 0.77 to 0.95 over the same cycles.

**Root cause.** A category error about what a learner is. The archive's
definition — a policy is its weights — is correct for a policy, and it was
reused for a *learner*, which is its weights plus the optimiser state plus
the data it is learning from. The resume code even said so in a comment
("the replay buffer and exploration rate are not persisted, so a resumed run
restarts both") and recorded a note to that effect in the lineage log. The
cost was documented and then not treated as a cost.

**Changed.**
- A learner now checkpoints its whole state — weights, optimiser state,
  replay buffer, counters and both random generators — once per cycle, and a
  resume prefers it. One file per lineage, overwritten, not committed.
- A resume records *what* it restored, so a weights-only fallback cannot be
  mistaken for a faithful continuation.
- A new expectation check: a score falling by more than 0.15 across an
  experiment boundary is INCONSISTENT, because nothing in the recipe changes
  at a restart. It fires on cycle 60 when run against the stored records.
- A checkpoint describing a differently-shaped network is refused rather than
  partially applied.

**Also found, and fixed, while investigating.** Expectation findings were
being read back from the cycle records and trusted. Because the running
process held a pre-fix copy of the check, every status display repeated 27
contradictions that the corrected check does not raise. Findings are derived
data: they are now recomputed on read, and a stored finding the current
checks reject is reported as superseded rather than as a fault. Relatedly,
`status` matched a running process to its lineage by comparing a hyphenated
experiment id against an underscored filename, so a live run was reported as
"just stopped".

**Standing rule added.** If a comment explains why a shortcut is acceptable,
the comment is evidence the shortcut has a cost. Either measure the cost or
remove the shortcut; do not record it and move on. (Compare R-006, where a
tolerance papered over a unit mismatch that had already been half-seen. Both
defects are the same move: noticing and annotating instead of fixing.)
