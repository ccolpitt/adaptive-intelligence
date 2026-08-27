# STATE — updated every session

**Last updated:** 2026-08-26 (framework Revision 3: diagnosis-driven architecture
evolution, working hypotheses a–f, world models promoted via ADR-003, environment
ladder added)
**Current phase:** Phase 0 — Loop harness (not started)

## Where things stand

- Repo scaffolded: steering, vision doc, roadmap, ADRs. No code yet.
- Seed material lives in the sibling repo `../connect4-rl`:
  - Champion-challenger training loop (reached v32 promotions in one run)
  - `league_play.py` — round-robin tournament over TorchScript policies
  - `models/experiments.json` — experiment registry pattern (hypothesis/changes/results)
  - TorchScript policies with embedded metadata (`_extra_files`) — keep this pattern
  - Neuron-reinitialization prototype (relevant to Phase 4)
- Known gap inherited from connect4-rl: no absolute benchmark exists anywhere.
  Champion v32's true strength is unknown. Phase 1 fixes this.

## Next task

Phase 0, first step: define the `Task`, `Policy`, `Evaluator`, `Archive`, `Scheduler`
interfaces (see `docs/roadmap.md` Phase 0), then port the connect4-rl
champion-challenger loop to run inside them with Connect 4 as seed task #1.

## Open questions for the human

- License choice for the repo
- Python version / dependency management preference (connect4-rl uses plain pip)
