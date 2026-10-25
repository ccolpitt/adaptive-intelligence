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

**Connect 4 is not yet mastered, but it is close for the first time.** The
fifth lineage of policies, and the first with a two-headed network, scores
**0.622** on the fixed ladder playing with no lookahead at all, after 12,000
self-play games. Every earlier lineage topped out at 0.113 that way. Mastery
requires 0.90 **at every rung**, and the current per-rung line is
0.71 / 0.66 / 0.70 / 0.57 / 0.54 / 0.55 — so it now beats perfect-play
opponents at every depth from 1 to 6, where four earlier lineages beat
essentially nothing past depth 1.

The change that did it: splitting one set of outputs into a **move head**
(which columns are worth considering) and a **value head** (how good is this
position, bounded to the reward range), each trained on what a budgeted
search concluded. See `exp-028-two-heads` in the registry for the full
reasoning and result.

**Currently running:** `exp-028-two-heads` extended from 60 to 160 cycles,
because its trend was still rising when the budget ran out. Resumable — if
it is not running, the same command continues it:

```bash
caffeinate -is python3 -u experiments/exp_028_two_heads.py
```

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

1. **Did any expectation fail?** Each cycle record carries a `sanity` list.
   A `contradiction` means a number cannot be true given how the system is
   built — that is always a defect, and it **halts interpretation of the
   whole cycle** (learning rule 11). Five defects have reached a run so far
   and none of them crashed; see `docs/defect-retros.md`.
   ```bash
   python -c "
   from harness.cycle_record import CycleStore
   for r in CycleStore('store/cycles').load_all('connect4-L5'):
       for f in (r.sanity or []): print(r.cycle, f['severity'], f['check'])"
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
   the policy **takes an available win** and **blocks a forced loss**. A
   policy beating a 6-ply perfect opponent should be near 1.00 on both.
   Currently 0.65 and 0.77 — the clearest remaining weakness.
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
| What do we test next, and in what order? | **`docs/hypothesis-queue.md`** — the human's file; edit freely |
| What rules does the system operate under? | `docs/learning-rules.md` (11 rules, each revisable, each with its reason) |
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
positions** per move, not a depth. Positions reached twice by different move
orders are served from a cache that costs no budget, and the cache is
discarded when the move ends. The move head decides where to look, the value
head judges what is found, and the move played is the **most-visited** one, as
AlphaZero does. A state budget beats a depth budget decisively at matched
cost (exp-023: 50 positions scored 0.220 where depth-2 needed 478 for 0.188).

---

## Known weaknesses, in priority order

1. **Tactics lag the score.** Taking an available win sits at 0.65 and
   blocking a forced loss at 0.77. A policy beating 6-ply opponents should be
   near 1.00 on both. This is the clearest signal that something in the
   learning path is lossy, and the best triage handle we have.
2. **The ladder may be running out of room.** The top rung is a 6-move
   search, and the policy now wins 0.55 against it. If it keeps improving,
   the measuring stick becomes the limit. The fix is a **new evaluation
   version** with deeper rungs (8, 10, 12) — `connect4-eval@v2`, recorded
   alongside v1 so no history is invalidated. Not yet built, and the next
   thing to build if scores keep climbing.
3. **No true oracle.** The ladder's opponents are depth-*limited* perfect
   play, not perfect play. Connect 4 is a solved game (the first player wins
   with the centre opening), so a genuine oracle is possible and would be the
   honest top rung.
4. **Cross-lineage comparison does not exist.** The gate only compares a
   challenger to its own lineage's champion; lineages are compared only
   through the absolute ladder. A direct league across lineages would be
   useful and is not built.
5. **Resume is not perfectly faithful.** Weights and cycle records persist,
   but the replay buffer and exploration rate restart, so a resumed cycle is
   slightly different from an uninterrupted one. Recorded in the lineage log
   each time it happens.

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
- **Weights are not in git** (`store/archive/*.pt`, 244MB+). The auditable
  record — index, events, registry, telemetry, cycle records — is committed.
- `git push` is not run by the agent. Ask the human.

**288 tests.** `python -m pytest -q` before and after any change.
