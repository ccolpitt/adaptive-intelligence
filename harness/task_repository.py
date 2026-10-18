"""The task repository: what we are trying to learn, and what counts as success.

A **task** is something we want a policy to be able to do — "win Connect 4",
"win at chess", "book a flight", "make money buying and selling things
online". A task on its own is just a name. What makes it usable is an
**evaluation definition**: a written, versioned, frozen specification of what
success means, precise enough that two people a year apart would measure the
same thing.

Why versioned and frozen: a score is meaningless unless the thing that
produced it is pinned. If "winning Connect 4" silently changes from "beat a
random player" to "beat a perfect player", every historical number becomes a
lie. So an evaluation definition, once published, never changes — a new
understanding of success becomes version 2, and both remain readable forever.
Existing results stay attached to the version that produced them.

Tasks may be declared long before they can be attempted. "Book a flight" has
no implementation and no scorer, but writing down what success would mean is
useful on its own: it exposes which tasks have a cheap, objective measure of
success (games with rules) and which will need human judgment or real-world
outcomes (open-ended tasks — see roadmap Phase 6).

Layout on disk, all plain JSON:

    store/tasks/<task_id>/task.json
    store/tasks/<task_id>/evaluations/eval-v1.json
    store/tasks/<task_id>/evaluations/eval-v2.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .jsonl_store import now_iso

# How success is established. Which of these a task uses says a lot about how
# hard it is to make progress on honestly.
SCORER_KINDS = (
    "exact_opponent",      # a rulebook opponent of known strength (Connect 4 solver)
    "fixed_dataset",       # a held-out labelled set (classification)
    "environment_return",  # the environment's own score (Atari, trading P&L)
    "human_judgment",      # solicited preference from people
    "real_outcome",        # something verifiable in the world (the flight was booked)
    "learned_judge",       # a frozen, version-stamped model standing in for judgment
)

STATUSES = ("active", "declared", "retired")


@dataclass
class Task:
    task_id: str
    name: str
    description: str
    status: str = "declared"
    current_evaluation: Optional[int] = None
    notes: str = ""
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "current_evaluation": self.current_evaluation,
            "notes": self.notes,
            "created_at": self.created_at,
        }


@dataclass
class Evaluation:
    """A frozen definition of success for one task."""

    task_id: str
    version: int
    success_means: str           # one plain sentence: what winning is
    scorer_kind: str             # one of SCORER_KINDS
    procedure: Dict              # exactly how a score is produced
    scoring: Dict                # how raw outcomes become a number
    mastery: Dict                # when the task is considered mastered
    inference_rule: str          # how the policy is allowed to think while scored
    noise_floor: Optional[float] = None  # smallest trustworthy difference
    implemented_by: Optional[str] = None  # code reference, if runnable
    notes: str = ""
    created_at: str = field(default_factory=now_iso)

    @property
    def ref(self) -> str:
        return f"{self.task_id}-eval@v{self.version}"

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "version": self.version,
            "ref": self.ref,
            "success_means": self.success_means,
            "scorer_kind": self.scorer_kind,
            "procedure": self.procedure,
            "scoring": self.scoring,
            "mastery": self.mastery,
            "inference_rule": self.inference_rule,
            "noise_floor": self.noise_floor,
            "implemented_by": self.implemented_by,
            "notes": self.notes,
            "created_at": self.created_at,
        }


class TaskRepository:
    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- tasks ----------------------------------------------------------

    def _task_dir(self, task_id: str) -> Path:
        return self.root / task_id

    def register_task(self, task: Task, overwrite_metadata: bool = True) -> Task:
        """Create or update a task's descriptive record. Descriptions and
        status may be edited freely — they are commentary. Evaluation
        definitions are the part that must never change."""
        path = self._task_dir(task.task_id) / "task.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite_metadata:
            return self.get_task(task.task_id)
        if task.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {task.status!r}")
        if path.exists():
            existing = json.loads(path.read_text())
            task.created_at = existing.get("created_at", task.created_at)
            if task.current_evaluation is None:
                task.current_evaluation = existing.get("current_evaluation")
        path.write_text(json.dumps(task.to_dict(), indent=2))
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        path = self._task_dir(task_id) / "task.json"
        if not path.exists():
            return None
        return Task(**json.loads(path.read_text()))

    def list_tasks(self) -> List[Task]:
        out = []
        for d in sorted(self.root.iterdir()):
            if (d / "task.json").exists():
                out.append(Task(**json.loads((d / "task.json").read_text())))
        return out

    # -- evaluation definitions -----------------------------------------

    def register_evaluation(self, evaluation: Evaluation, make_current: bool = True) -> Evaluation:
        """Publish an evaluation definition. Re-registering an identical
        definition is a no-op; re-registering a DIFFERENT one under the same
        version is refused — change means a new version."""
        if evaluation.scorer_kind not in SCORER_KINDS:
            raise ValueError(f"scorer_kind must be one of {SCORER_KINDS}")
        if self.get_task(evaluation.task_id) is None:
            raise ValueError(f"register the task {evaluation.task_id!r} first")
        path = (self._task_dir(evaluation.task_id) / "evaluations"
                / f"eval-v{evaluation.version}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        new = evaluation.to_dict()
        if path.exists():
            old = json.loads(path.read_text())
            comparable_old = {k: v for k, v in old.items() if k != "created_at"}
            comparable_new = {k: v for k, v in new.items() if k != "created_at"}
            if comparable_old != comparable_new:
                raise ValueError(
                    f"{evaluation.ref} is already published with a different "
                    "definition — a changed definition of success is a NEW version, "
                    "so that existing scores keep their meaning"
                )
            return Evaluation(**{k: v for k, v in old.items() if k != "ref"})
        path.write_text(json.dumps(new, indent=2))
        if make_current:
            task = self.get_task(evaluation.task_id)
            task.current_evaluation = evaluation.version
            self.register_task(task)
        return evaluation

    def get_evaluation(self, task_id: str, version: Optional[int] = None) -> Optional[Evaluation]:
        if version is None:
            task = self.get_task(task_id)
            if task is None or task.current_evaluation is None:
                return None
            version = task.current_evaluation
        path = self._task_dir(task_id) / "evaluations" / f"eval-v{version}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        data.pop("ref", None)
        return Evaluation(**data)

    def list_evaluations(self, task_id: str) -> List[Evaluation]:
        d = self._task_dir(task_id) / "evaluations"
        if not d.exists():
            return []
        out = []
        for p in sorted(d.glob("eval-v*.json")):
            data = json.loads(p.read_text())
            data.pop("ref", None)
            out.append(Evaluation(**data))
        return out


# ---------------------------------------------------------------------------
# the starting contents of the repository
# ---------------------------------------------------------------------------


def connect4_evaluation_v1() -> Evaluation:
    """What 'winning Connect 4' means, written out. This is the definition
    every score in the project so far was produced under."""
    return Evaluation(
        task_id="connect4",
        version=1,
        success_means=(
            "Beat progressively stronger perfect-play opponents at Connect 4. "
            "A policy has mastered the task when it wins or draws at least 90% "
            "of games against every opponent on the ladder, including one that "
            "plays perfectly six moves deep."
        ),
        scorer_kind="exact_opponent",
        procedure={
            "opponents": "negamax search with alpha-beta pruning over the real "
                         "rules, at fixed depths 1, 2, 3, 4, 5 and 6 (an opponent "
                         "of exactly known strength, not an estimate)",
            "openings": "14 fixed opening sequences (7 of one move, 7 of two "
                        "moves), chosen once with a recorded seed and frozen",
            "seats": "each opening played twice, once with the policy moving "
                     "first and once second",
            "games_per_opponent": 28,
            "games_total": 168,
            "determinism": "the opponents and openings are fixed, so the only "
                           "variation between runs is the policy itself",
        },
        scoring={
            "per_game": "win = 1, draw = 0.5, loss = 0",
            "per_opponent": "mean over that opponent's 28 games",
            "headline": "mean of the six per-opponent scores, in [0, 1]",
            "frontier_depth": "the shallowest opponent not yet beaten 90% of "
                              "the time — reports which opponent is next",
        },
        mastery={
            "per_opponent": 0.90,
            "headline_equivalent": 0.90,
            "note": "the headline average can be lifted by easy rungs, so "
                    "mastery requires the threshold at EVERY depth",
        },
        inference_rule=(
            "The policy reads the position once and commits to a move. No "
            "lookahead, no search, no assistance. Lookahead during training is "
            "unrestricted; see learning rules 1 and 2."
        ),
        noise_floor=0.09,
        implemented_by="harness.tasks.connect4_ladder.Connect4LadderBenchmark",
        notes=(
            "Deliberately absolute rather than relative. An earlier project "
            "measured only wins against previous versions of itself, promoted 32 "
            "times, and never knew whether it was any good. A fixed opponent of "
            "known strength is what makes a score mean something years later."
        ),
    )


def seed_repository(repo: TaskRepository) -> None:
    """Register the task we are working on, plus the tasks we intend to reach.
    Declaring a task before it is possible is useful: it shows which ones have
    a cheap objective measure of success and which will need human judgment."""
    repo.register_task(Task(
        task_id="connect4",
        name="Win Connect 4",
        description=(
            "Play Connect 4 well enough to beat perfect-play opponents. The "
            "first task of the project: small enough to iterate on quickly, and "
            "it has a perfect scorer, so progress is never a matter of opinion."
        ),
        status="active",
        notes="Also the first task for the transfer question: a second, related "
              "game should be learnable in fewer games because of what was "
              "learned here.",
    ))
    repo.register_evaluation(connect4_evaluation_v1())

    repo.register_task(Task(
        task_id="connect4-variants",
        name="Win Connect 4 variants",
        description=(
            "The same game on different board sizes and win lengths (connect 3 "
            "on 5x5, connect 4 on 8x8, and so on). The planned second task, "
            "used to measure whether mastering one game makes the next cheaper."
        ),
        status="declared",
        notes="Scorer is the same kind as Connect 4 — a depth-limited perfect "
              "opponent — so success is objective and cheap to define.",
    ))

    repo.register_task(Task(
        task_id="chess",
        name="Win at chess",
        description=(
            "A much larger game with the same shape of scorer: an engine at a "
            "fixed strength setting. Tests whether the approach survives a big "
            "jump in the number of possible positions and moves."
        ),
        status="declared",
        notes="Success stays objective, but no perfect player exists, so the "
              "scorer becomes a frozen engine at a pinned version and setting.",
    ))

    repo.register_task(Task(
        task_id="flight-booking",
        name="Book a flight",
        description=(
            "Given a request in words, book a flight that satisfies it. The "
            "first task with no rulebook: there is nothing to search through "
            "exactly, so looking ahead requires a learned model of how the "
            "world responds, and the gap between that model and reality has to "
            "be handled honestly."
        ),
        status="declared",
        notes="Success is a real outcome (the right flight was booked within "
              "constraints) and is verifiable but expensive, and mistakes cost "
              "real money. Needs the learned-model path in learning rule 5.",
    ))

    repo.register_task(Task(
        task_id="restaurant-reservation",
        name="Make a restaurant reservation",
        description=(
            "Hold a conversation or navigate a site to secure a reservation "
            "matching a request. Like flight booking but cheaper to retry, so a "
            "better first step into tasks with no rulebook."
        ),
        status="declared",
    ))

    repo.register_task(Task(
        task_id="online-trading",
        name="Make money buying and selling things online",
        description=(
            "Buy and sell goods for a profit. The environment supplies its own "
            "score — money — which makes success objective, but it is not fixed "
            "the way a game opponent is: the world changes underneath, so a "
            "score from last year is not comparable with one from today."
        ),
        status="declared",
        notes="Open question for this task: how to define a stationary "
              "evaluation when the environment itself drifts. Likely needs a "
              "recorded, replayable market period as the frozen scorer.",
    ))
