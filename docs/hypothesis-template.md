# Hypothesis Template

Every experiment's hypothesis is written for a human reader — someone (probably
future-you) who will read it cold, years later, deciding whether to revisit it.
No jargon soup. Follow this structure:

## CONTEXT (the pre-sentence that sets the stage)

State, in plain sentences:
- The mode of learning (e.g., self-play against a frozen champion) and training
  volume (episodes).
- How the board and player are represented (currently: AlphaZero-style two
  planes, 6x7 each — plane 0 = pieces of the player about to move, plane 1 =
  opponent's pieces; the board flips to the mover's perspective every move, so
  no "whose turn" input exists).
- What the policy currently is (architecture, learning algorithm).
- Where the current champion stands on the yardstick.

## HYPOTHESIS

One or two sentences naming the change and the belief behind it. The change
must be to one of:
- policy architecture,
- a training hyperparameter (including training volume),
- the mode of learning (e.g., self-play vs playing the solver, curriculum),
- the representation of the board / player / action,
- the reward/feedback signal.

**Specificity rule: every changed variable is named with its from -> to
values.** "Big swing" or "improved training" mean nothing a year from now.
Write "episodes 1500 -> 4500, eps_decay 0.999 -> 0.9995, buffer 20000 ->
50000"; never a nickname. Compound (multi-variable) hypotheses are allowed
when framed as necessary-conditions packages ("for the policy to improve, BOTH
a learnable signal AND sufficient capacity must be present") — but the record
must list every ingredient with from -> to values, and the verdict credits or
blames the PACKAGE, never an individual ingredient, until single-variable
follow-ups apportion it.

## PREDICTION

"If true, we expect to measure at least Y, because Z." Y is a number on a named
benchmark; Z is the mechanism.

## DECISION CRITERIA

Recorded so a false positive or false negative can be traced back later, before
it sends the project down a bad path:
- The promotion gates in force (head-to-head confidence bound; no significant
  benchmark regression) and the false-positive rate they imply.
- The smallest effect the game counts can detect. A true-but-smaller gain reads
  INCONCLUSIVE — the follow-up is more evaluation games, not abandoning the
  direction. Inconclusive is never recorded as refuted.

## Registration mechanics

Experiments live as scripts in `experiments/` (committed = provenance). The
script registers the hypothesis BEFORE training and runs the loop. Verdicts are
recorded once, after reading the results, via `python -m harness verdict`.
