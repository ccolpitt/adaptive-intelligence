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
| [`.kiro/steering/core.md`](.kiro/steering/core.md) | Agent steering: principles and session protocol |
| [`docs/design/decisions/`](docs/design/decisions/) | Architecture decision records |

## Status

Phase 0 (loop harness) — **complete and closed end-to-end.** The domain-agnostic
train → evaluate → select → archive → repeat loop runs on Connect 4 (seed task #1),
ported and generalized from the sibling [connect4-rl](../connect4-rl) project.
First experiment (`exp-001-close-the-loop`): one promotion through the double
gate, benchmark 0.725 → 0.785, verdict recorded. Next: Phase 1, the solver-based
absolute benchmark (see `STATE.md`).

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

```bash
# Run the improvement loop (one run = one registered experiment):
python3 -m harness run --store store \
    --experiment-id exp-002 \
    --hypothesis "if X, we will see Y" \
    --changes "the one variable changed"

# Tell the story of how a policy came to be (lineage + hypotheses + verdicts):
python3 -m harness report --store store --policy connect4-pol-00005

# Record an experiment's verdict (once, immutable):
python3 -m harness verdict --store store --experiment-id exp-002 \
    --verdict supported --tldr "one line: what we learned"

# Play Connect 4 against the solver oracle (see what search depth buys —
# try --depth 2 vs --depth 10; change mid-game with 'd N'):
python3 -m harness play-solver --depth 8 --first human
```

## Repository layout

| Path | What it is |
|---|---|
| `harness/` | The Phase 0 loop harness (interfaces, loop, archive, registries, evaluator, CLI) |
| `tests/` | The proof the harness isn't a source of training errors — run before every experiment batch |
| `store/` | Persistent results, committed to git: `archive/` (append-only policy fossil record), `registry/` (task + experiment JSONL registries), `telemetry/` (training curves). Grows append-only by design; revisit storage (git-LFS / external) when it approaches ~500 MB. |
