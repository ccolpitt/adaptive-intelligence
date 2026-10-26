# adaptive-intelligence

An experimental framework for an automated policy-improvement loop: train, evaluate
against fixed benchmarks, select, archive, repeat — then upgrade the loop's components
(curriculum, plasticity, architecture) one at a time.

Central hypothesis: **a single network that masters multiple related tasks becomes
increasingly sample-efficient at learning the next one.**

## Orientation

| Doc | What it is |
|---|---|
| [`docs/vision.md`](docs/vision.md) | The full framework: five foundational components, critiques, verdicts, research references |
| [`docs/roadmap.md`](docs/roadmap.md) | Phased implementation plan with definitions of done |
| [`STATE.md`](STATE.md) | Current position and next task — updated every session |
| [`docs/using-the-harness.md`](docs/using-the-harness.md) | **Start here to run anything** — play a policy, read a score, check what's training |
| [`.kiro/steering/core.md`](.kiro/steering/core.md) | Agent steering: principles and session protocol |
| [`docs/learning-rules.md`](docs/learning-rules.md) | Standing rules the project has adopted, and why |
| [`docs/defect-retros.md`](docs/defect-retros.md) | Every defect that reached a run, with its cost and root cause |
| [`docs/design/decisions/`](docs/design/decisions/) | Architecture decision records |

## Status

The loop is closed and has run 29 registered experiments across five lineages of
policies. The current line (`connect4-L5`) is the first with two heads — one
proposing moves, one judging positions, trained on what a budgeted search
concluded. It scores **0.646** on the solver ladder reading the position once,
with no search.

For scale: a random player scores 0.000 on that ladder, a four-ply exhaustive
solver 0.488, an eight-ply solver 0.717, and a ten-ply solver 0.830. So a single
forward pass is at about 78% of a ten-ply search. Run
`python -m harness reference` for the full board — and read it before trusting
any score, because `1.0` is not reachable on this benchmark.

## Setup (any machine)

Requires Python 3.12.

```bash
git clone https://github.com/ccolpitt/adaptive-intelligence.git
cd adaptive-intelligence
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest            # fast suite + slow learnability canary (~30 s)
```

All dependencies are pinned exactly — the environment is part of every
experiment's conditions. If a pin must change, that's a condition change:
note it in the experiment registry.

## Usage

Full command reference: [`docs/using-the-harness.md`](docs/using-the-harness.md).
The four you will want first:

```bash
# Play the current champion. Add --lookahead 200 to let it think.
python3 -m harness play

# List the policies you could play instead.
python3 -m harness policies

# What a benchmark score is actually worth. Read this before any other number.
python3 -m harness reference

# Is anything training, and how far has it got?
python3 -m harness status
```

## Repository layout

| Path | What it is |
|---|---|
| `harness/` | The Phase 0 loop harness (interfaces, loop, archive, registries, evaluator, CLI) |
| `tests/` | The proof the harness isn't a source of training errors — run before every experiment batch |
| `store/` | Persistent results, committed to git: `archive/` (append-only policy fossil record), `registry/` (task + experiment JSONL registries), `telemetry/` (training curves). Grows append-only by design; revisit storage (git-LFS / external) when it approaches ~500 MB. |
