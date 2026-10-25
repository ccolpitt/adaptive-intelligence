"""Shared boilerplate for experiment scripts: incumbent context measurement
and the standard context paragraph every hypothesis begins with."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from harness.improvement_loop import Store as _Store  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent


def Store(name: str = "store") -> _Store:
    """Store anchored at the repo root regardless of the caller's cwd."""
    return _Store(_REPO_ROOT / name)


def measure_incumbent(store, task):
    """Read-only: the current champion's ladder standing, for CONTEXT."""
    incumbent_id = store.archive.current_champion(task.task_id)
    module, _ = store.archive.load_policy(incumbent_id)

    def q(obs):
        with torch.no_grad():
            return module(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).numpy()

    result = task.benchmark.score(q)
    print(
        f"incumbent {incumbent_id}: ladder {result.score:.3f}, "
        f"frontier depth {result.detail['frontier_depth']}, "
        f"per rung {result.detail['per_rung']}"
    )
    return incumbent_id, result


def context_paragraph(incumbent_id, incumbent) -> str:
    return f"""\
CONTEXT: We improve the policy by self-play: a challenger (epsilon-greedy DQN,
negamax targets) learns by playing a frozen copy of the current champion. The
board is two 6x7 planes - plane 0 the mover's pieces, plane 1 the opponent's -
flipped to the mover's perspective every move (no whose-turn input). The
policy is a small conv net (2x32-filter conv layers, 128-unit hidden layer,
7 Q-value outputs). The current champion ({incumbent_id}) scores
{incumbent.score:.3f} on the solver ladder (depths 1-6; 1.0 = beats every
depth nearly always); its frontier - the shallowest solver it cannot beat 90%
of the time - is depth {incumbent.detail['frontier_depth']}."""


DECISION_CRITERIA = """\
DECISION CRITERIA (recorded so a false positive or negative can be traced
later): promotion requires the head-to-head score's 95% Wilson lower bound to
clear 0.5 over 200 games (~5% false-promotion rate) and no statistically
significant ladder regression. The 168-game ladder cannot distinguish gains
under ~0.09 from noise: a positive-but-smaller gain is recorded INCONCLUSIVE,
never refuted - the follow-up is more evaluation games, not abandonment."""


def register_once(store, experiment_id: str, hypothesis: str, **kwargs):
    """Register an experiment, or confirm it is already registered unchanged.

    The registry refuses to re-register an experiment, which is right: a
    hypothesis is recorded before the run and must not be edited afterwards.
    But runs are now resumable, and resuming means running the same script
    again — so a script that registers unconditionally cannot resume. This
    registers the first time and, on later runs, checks that the hypothesis
    on disk still matches the one in the script. If it has drifted, it stops:
    continuing would attach results to a hypothesis that was not the one
    being tested.
    """
    existing = store.experiments.get(experiment_id)
    if existing is None:
        return store.experiments.register(
            experiment_id=experiment_id, hypothesis=hypothesis, **kwargs)
    if existing.hypothesis.strip() != hypothesis.strip():
        raise ValueError(
            f"{experiment_id} is already registered with a DIFFERENT hypothesis. "
            "Resuming would attach new results to a hypothesis that is not the "
            "one on record. Either restore the original wording or use a new "
            "experiment id."
        )
    print(f"[{experiment_id}] already registered; continuing the same experiment")
    return existing
