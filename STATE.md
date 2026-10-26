# STATE — start here

**Updated:** 2026-10-26 · **Phase:** 0/1 (closing the loop, measuring one task)

This file is the entry point for a new session. Read it first, then run the
two commands in "Check what is happening" below. Everything else in this
document is detail you can read as needed.

---

## What this project is trying to do

Prove that **one network that masters several related tasks learns each new
one with fewer samples than the last**, and that the mechanism is **learning
to act and learning a model of the world at the same time**. Full framework:
`docs/vision.md` §3. Connect 4 is the first task — small, with a perfect
scorer — chosen so the machinery could be built and debugged before the
transfer question starts.

---

## Where things stand

**The fifth lineage of policies, and the first with a two-headed network,
scores 0.646 on the fixed ladder reading the position once, with no search.**
Every earlier lineage topped out at 0.113 that way.

The change that did it: splitting one set of outputs into a **move head**
(which columns are worth considering) and a **value head** (how good is this
position, bounded to the reward range), each trained on what a budgeted
search concluded. See `exp-028-two-heads` in the registry.

### Read that number against the scale, not against 1.0

This is the most important thing a new session can know, and the project got
it wrong for its entire history. `python -m harness reference`:

| player | ladder score |
|---|---|
| random | 0.000 |
| prefer the centre column | 0.000 |
| solver, 1 ply | 0.277 |
| solver, 4 plies | 0.488 |
| **our champion, no search** | **0.646** |
| solver, 8 plies | 0.717 |
| solver, 10 plies | 0.830 |

So the champion is at **78% of a ten-ply exhaustive search**, using one
forward pass against a search of thousands of positions per move — not "65%
of the way to a reachable 1.0". The benchmark forces both players through
fixed openings and plays the policy in both seats, so some of the 168 games
begin from objectively lost positions. And rung 1 is **not** a weak opponent:
every rung takes every immediate win and blocks every immediate threat,
because the solver checks for a win before it checks its depth limit.

Mastery is still defined as 0.90 at every rung. The strongest reference
reaches 0.85 or better at two of six rungs, so the bar is demanding but not
provably impossible — and until some reference clears it, a reported
`frontier_depth` of 1 is uninformative rather than a verdict. The ceiling is
a **measurement, not a deduction**: on depth 8 alone it looked as though 0.72
were the limit, and depth 10 then reached 0.830.

**Currently running:** `exp-029-two-heads-extended`, 60 → 160 cycles, at
cycle 127 with best 0.670. Resumable — the same command continues it:

```bash
caffeinate -is python3 -u experiments/exp_029_two_heads_extended.py
```

**Its verdict must name two confounds**, both annotated rather than fatal
under learning rule 12. It began with a cold restart that dropped the score
from 0.622 to 0.390 and cost 27 cycles to recover (R-007). And it has been
searching far more shallowly than its rubric declares, because terminal
positions were charged to the budget on every re-visit — a declared 200
positions bought 14 in a win-in-one position (R-008). Both are fixed for
future runs, so the recipe exp-029 recorded is not the recipe it ran.

---

## Check what is happening

```bash
python -m harness status                      # running? champion? verdicts owed?
python -m harness story                       # every experiment: lever, expectation, result
```

Then, for the active line of policies:

```bash
python -c "from harness.learning_arc import *; print(render_text(summarize('store','connect4-L5')))"
```

---

## What to look at after a training run

In this order. The first two are not optional.

1. **Did any expectation fail?** A `contradiction` means a number cannot be
   true given how the system is built — always a defect, and it **halts
   interpretation of the whole cycle** (learning rule 11). Eight defects have
   reached a run so far and only two crashed; see `docs/defect-retros.md`.

   Findings are **recomputed, never read back from the records** (learning
   rule 13): a check that was wrong when a cycle ran would otherwise keep
   reporting a defect since fixed, which is exactly what happened for 68
   consecutive cycles of L5. `python -m harness status` does this for you and
   separates live contradictions from superseded ones.
   ```bash
   python -c "
   from harness.cycle_record import CycleStore
   from harness.sanity import recheck
   rs = CycleStore('store/cycles').load_all('connect4-L5')
   for i, r in enumerate(rs):
       for f in recheck(r, rs[:i]): print(r.cycle, f.severity, f.check)"
   ```
2. **Is the line actually improving?** The arc, not any single cycle. A cycle
   can look healthy while the arc is flat.
   ```bash
   python -c "
   from harness.learning_arc import *
   s = summarize('store','connect4-L5'); print(render_text(s)); print(render_plot(s))"
   ```
3. **Do the tactics make sense?** Two numbers matter most, and they are the
   triage tool for *why* learning is failing when it fails: the rate at which
   the policy **takes an available win** and **blocks a forced loss**. Both
   reached 0.91 and 0.95 late in exp-029, up from 0.65 and 0.77.
   ```bash
   python -m harness tactics --lineage connect4-L5
   ```
4. **Look at a cycle card.** Nine panels: training error, weight-correction
   size, value estimates against the reward ceiling, distinct positions seen,
   win rates, lookahead use and depth, value of already-decided positions,
   weights dying or blowing up, and the outcome with its diagnosis.
   ```bash
   python -c "
   from harness.cycle_record import CycleStore
   print(CycleStore('store/cycles').load_all('connect4-L5')[-1].render())"
   ```
5. **Record a verdict** once judged — once only, immutable:
   ```bash
   python -m harness verdict --experiment-id <id> --verdict supported|refuted|inconclusive \
       --tldr "what we learned" --results '{"key": 1.0}'
   ```

---

## Where the plan lives

| Question | File |
|---|---|
| How do I run anything? | **`docs/using-the-harness.md`** — play a policy, read a score, check what's training |
| How should I reason and report? | **`docs/working-with-the-human.md`** — read before writing findings up |
| What do we test next, and in what order? | **`docs/hypothesis-queue.md`** — the human's file; edit freely |
| What rules does the system operate under? | `docs/learning-rules.md` (13 rules, each revisable, each with its reason) |
| What are the long-term phases? | `docs/roadmap.md` |
| Why does the project exist? | `docs/vision.md` |
| What decisions were made, and why? | `docs/design/decisions/ADR-00*.md` (7 so far) |
| What has been tried, and what happened? | `python -m harness story`, or `store/registry/experiments.jsonl` |
| What broke, and what changed because of it? | `docs/defect-retros.md` |
| Every knob that can be turned | `docs/training-parameters.md` |

---

## How the system is put together

**The task.** `store/tasks/connect4/` holds the task and its **frozen,
versioned definition of success** (`evaluations/eval-v1.json`): 168 games
against perfect-play opponents searching 1 to 6 moves deep, 28 against each,
averaged; mastery is 0.90 at every rung; **the policy is scored playing
alone, with no lookahead**. A changed definition becomes a new version so old
scores keep their meaning. Five further tasks are declared but not
implemented (chess, variants, flight booking, restaurant reservation, online
trading).

**A lineage** (`store/lineages/`) is a line of descent on one task. Starting a
new one means restarting the policy from scratch while keeping everything the
previous line taught us. Five exist: L1 no lookahead · L2 lookahead 32 · L3
lookahead 128 unbounded (contained a defect, see R-001) · L4 bounded single
head · **L5 two heads, the current line**.

**A rubric** (`store/rubrics/`) is the complete recipe that produced a
policy: learner, lookahead budget, how that budget may be spent, what the
targets were, every hyperparameter. Published frozen. The trainer **refuses
to run** if a rubric specifies anything that does not reach it.

**A cycle** is one round: train for a declared number of games, evaluate,
then decide whether the result replaces the champion. Backpropagation happens
far more often — several steps per game. The cycle boundary is the
*evaluation and promotion* point. (AlphaZero calls this a generation; "epoch"
would be wrong, since the data is generated continuously.)

**Cycle records** (`store/cycles/<lineage>/`) are **JSON only, never
pictures**, and each names the renderer that draws it, so
`record.render()` dispatches to the right version and an old cycle is always
drawn as it was meant to be. A new measurement means a new renderer version
created *before* the cycle runs. Renderers: `harness/renderers/`
(`connect4_v1`, `connect4_v2` which added win rate against a random player).

**Promotion** requires beating the champion **and** improving on the task's
own measure (`gate_rule="strict"`, ADR-007). Health conditions can withdraw a
promotion the scores qualified — non-finite weights, diverging value
estimates, a collapsed layer — but weak tactics only warn, because improving
from terrible to poor is still improving.

**Search** (`harness/budgeted_search.py`) takes a budget of **distinct
positions** per move, not a depth. A position is charged **once per move**
however often the search returns to it: repeats are served from a cache, and
terminal positions are free after the first look because the rules give their
value exactly. Re-visiting is not waste — a lopsided visit count is precisely
the training target the move head learns from — so the fix for R-008 stopped
*billing* for confidence rather than stopping the search from expressing it.
The cache is discarded when the move ends, since the weights may have changed.

The move head decides where to look (selection is PUCT: value plus an
exploration bonus weighted by the move head's prior), the value head judges
what is found, and the move played is the **most-visited** one, as AlphaZero
does. A state budget beats a depth budget decisively at matched cost
(exp-023: 50 positions scored 0.220 where depth-2 needed 478 for 0.188).

**Reward propagation, in one paragraph**, because it is the thing most often
misunderstood. The value head is trained by squared error onto the **final
game outcome** from each position's own mover's perspective — no
bootstrapping, no discount, and the search's backed-up root value is computed
and then discarded. The move head's loss contains **no reward at all**: its
target is the search's visit distribution, and reward reaches it only through
the tree, where exact terminal rewards from the rules are negated at each ply
on the way up and accumulate as visits. So the game rules do the sequential
credit assignment, and the depth of tactical credit the move head can receive
equals the depth the search actually reaches.

---

## Known weaknesses, in priority order

1. **The move head generalises badly outside self-play; the value head and
   the search are both fine.** Verified directly on the current champion. In
   a constructed position with a win at column 3, the move head prefers
   column 6 (logit +3.75 against the winning move's −0.59), while the value
   head correctly reads +0.977 and a 200-position search plays column 3 with
   3,923 of 4,000 visits. The ladder's opponents are solvers, whose positions
   look nothing like self-play, which is a sufficient explanation for the
   remaining gap. Queue items 0 and 0d attack this from the data side and the
   loss side respectively.
2. **Decisive positions are a few percent of the training signal.** Measured
   over 616 positions of champion self-play: a win was available in 3.6%, a
   block was required in 9.6%, and 86.9% were quiet. The buffer samples
   uniformly, so roughly seven eighths of the move head's gradient comes from
   positions where almost any sensible move is fine.
3. **The value target carries no discount.** Every position in a won game is
   labelled +1.0, the opening exactly like the winning move. Unbiased but
   very high variance, and at 200 games per cycle this project is
   variance-limited. Queue item 0c.
4. **Batch normalisation is load-bearing and unexamined.** Switching which
   statistics are used flips the chosen column in 13% of positions and moves
   the value estimate by up to a full 1.0. Whether the running averages suit
   the deep positions the search visits is not measured. Queue item 0f.
5. **No true oracle, and no deeper rungs.** The ladder's opponents are
   depth-*limited* perfect play. Connect 4 is solved, so a genuine oracle is
   possible and would be the honest top rung; deeper rungs (8, 10, 12) would
   also be published as `connect4-eval@v2` so no recorded score is
   invalidated. Note that the reference board now gives much of what deeper
   rungs were wanted for.
6. **Cross-lineage comparison does not exist.** The gate only compares a
   challenger to its own lineage's champion; lineages are compared only
   through the absolute ladder.

---

## Working practices that matter

- **Register the hypothesis before the run**, in the structure of
  `docs/hypothesis-template.md`: if we change X we expect Y because Z, with
  every changed variable named *from → to*, and no jargon.
- **Record the verdict once.** Immutable. Revisiting a hypothesis under new
  conditions is a *new* experiment that cites the old one.
- **Three seeds** for any effect smaller than the spread between seeds
  (~0.06 on this ladder). The same recipe at two seeds once gave 0.054 and
  0.116.
- **One lever per experiment.** Packages must be decomposed before credit is
  assigned.
- **Store data, render on demand.** Never commit a chart.
- **Do not modify `~/Development/github/connect4-rl`** — read-only reference.
- **Read a score against the reference board**, never against 1.0.
  `python -m harness reference`.
- **Weights and resume points are not in git** (`store/archive/*.pt`,
  `store/checkpoints/`). The auditable record — index, events, registry,
  telemetry, cycle records, reference boards — is committed.
- `git push` is not run by the agent. Ask the human.

**299 tests.** `python -m pytest -q` before and after any change. Check
pytest's own exit code — `pytest -q | tail` reports `tail`'s status and will
hide a failure.
