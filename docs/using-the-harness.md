# Using the harness

Everything here is a command you can run. For what the project is trying to do,
read [`docs/vision.md`](vision.md); for where it currently stands,
[`STATE.md`](../STATE.md).

Run all of these from the repository root with the virtual environment active:

```bash
source .venv/bin/activate
```

---

## Playing against a policy

This is the fastest way to form your own opinion about how good the thing is.

```bash
# Play the latest champion. You move first; columns are 0-6, 'q' quits.
python -m harness play

# Play a specific policy.
python -m harness play --policy connect4-L5-pol-00106

# Play a particular line of policies' champion.
python -m harness play --lineage connect4-L5

# Let it move first.
python -m harness play --first policy
```

### Playing it with and without thinking

A policy is *scored* with no search at all: it reads the position once and
commits. That is learning rule 1, and it is a deliberately hard standard. But
the same policy can be given a thinking budget, and the difference is large and
worth seeing:

```bash
# As it is scored: one look, no search. This is the number in every report.
python -m harness play --lookahead 0

# The same policy allowed to examine 200 distinct positions per move.
python -m harness play --lookahead 200
```

With `--lookahead 0` you see the move head's preference per column. With a
budget you see how the search spent it — visits per column and the value it
backed up — and you will sometimes watch it overturn its own first instinct.
In a position with a win one move away, the current champion's move head
prefers the wrong column while its search finds the win and plays it. That gap
is the single most informative thing in the project right now.

### Which policies exist

```bash
# Every policy that was ever promoted, newest last, with its benchmark score.
python -m harness policies

# One lineage only.
python -m harness policies --lineage connect4-L5

# Every policy, not just the promoted ones (there are hundreds).
python -m harness policies --all --limit 20
```

Lineages are named `connect4-L1` through `connect4-L5`. A new lineage is started
when the network's shape changes, because weights cannot be carried across a
shape change and comparing two differently-shaped networks as one would be
dishonest. `L5` is the current one: the first with two heads.

### Playing the solver instead

The solver is the grader, not a learner — exhaustive negamax over the real
rules. Playing it at different depths is the best way to feel what the ladder's
rungs actually are:

```bash
python -m harness play-solver --depth 1     # the shallowest rung
python -m harness play-solver --depth 10    # change mid-game with 'd N'
```

Depth 1 is not a beginner. It takes every immediate win and blocks every
immediate threat, because it checks for a win before it checks its depth limit.
Expect it to punish you.

---

## Reading a score

```bash
# What a benchmark score is worth, against players of known strength.
python -m harness reference

# Place a particular policy on that scale.
python -m harness reference --policy connect4-L5-pol-00106

# Re-measure the reference players (minutes).
python -m harness reference --measure
```

**Read this before trusting any score.** The ladder score lives in `[0, 1]`, and
for most of this project's life it was read as though `1.0` were reachable. It
is not. The benchmark forces both players through fixed openings and plays the
policy in both seats, so a good fraction of the 168 games begin from positions
that are objectively lost. Measured reference points:

| Player | Ladder score |
|---|---|
| random | 0.000 |
| prefer the centre column | 0.000 |
| solver, 1 ply | 0.277 |
| solver, 4 plies | 0.488 |
| our champion (no search) | 0.646 |
| solver, 8 plies | 0.717 |
| **solver, 10 plies** | **0.830** |

So 0.646 is about 78% of what a ten-ply exhaustive searcher manages on the same
games, not 65% of something achievable — and our champion does it with a single
forward pass while the solver searches thousands of positions per move.

Two cautions. The ceiling is a *measurement*, and it moves: on depth 8 alone it
looked as though 0.72 were the limit, and depth 10 then reached 0.830. And the
mastery bar of 0.90 at every rung is demanding but not provably impossible — the
best reference clears 0.85 at two of six rungs. Until some reference clears 0.90,
treat a reported `frontier_depth` of 1 as uninformative rather than as a verdict.

---

## Checking what is running

```bash
# Is anything training? How far has each lineage got? What is unfinished?
python -m harness status
```

It leads with the process table, because a training run writes nothing for
minutes at a time and file timestamps cannot answer the question. It then shows
each lineage's progress, any contradictions worth investigating, and any
experiment still owing a verdict.

---

## Diagnostics

```bash
# A policy's performance card: value calibration, tactics, neuron health.
python -m harness card --policy connect4-L5-pol-00106

# How often does it take an available win, or block a forced loss?
python -m harness tactics --lineage connect4-L5

# The same two rates, split by the direction of the line and by whether the
# positions came from the policy's own play or from random play.
python -m harness tactics-by-geometry

# Read the instruments and say what to try next.
python -m harness diagnose --policy connect4-L5-pol-00106

# The story of how a policy came to be: lineage, hypotheses, verdicts.
python -m harness report --policy connect4-L5-pol-00106

# Should the expectation checks be trimmed or extended?
python -m harness review-checks
```

---

## Running an experiment

Experiments are scripts in `experiments/`, one per hypothesis, named for what
they test. Each registers its hypothesis *before* it runs and refuses to start
if the hypothesis on disk has drifted from the one in the script.

```bash
# Run one. Resumable: the same command continues from the last finished cycle.
caffeinate -is python3 -u experiments/exp_029_two_heads_extended.py

# Record the verdict afterwards (once, immutable).
python -m harness verdict --experiment-id exp-029-two-heads-extended \
    --verdict supported --tldr "one line: what we learned"
```

A run can be stopped and restarted safely: each cycle leaves behind the
learner's whole state — weights, the optimiser's accumulated averages, and the
replay buffer — so a resume continues rather than restarts. That was not always
true, and the difference cost 27 cycles once (see R-007 in
[`docs/defect-retros.md`](defect-retros.md)).

---

## Where things are kept

| Path | What it holds |
|---|---|
| `store/archive/` | Every policy ever trained, append-only. Weights are local; the index and event log are committed. |
| `store/cycles/` | One JSON record per training cycle: the series, the scores, the gate's reasoning. |
| `store/lineages/` | Per-lineage event logs and the training recipe each was trained under. |
| `store/reference/` | What reference players score on each benchmark version. |
| `store/checkpoints/` | Resume points. Large, derived, not committed. |
| `store/tasks/` | Task definitions and frozen evaluation specifications. |

Nothing is ever deleted. A policy's "death" means it stops receiving compute.
