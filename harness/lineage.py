"""Training rubrics and policy lineages.

Two ideas, recorded because without them a policy is an orphan — a file of
weights with a score and no answer to "how was this made?"

A **training rubric** is the complete recipe that produced a policy: which
learning algorithm, how much lookahead it was allowed per move and how it was
permitted to spend it, what its learning targets were, and every
hyperparameter. Rubrics are published and frozen, so "trained under rubric
r003" is a precise statement a year from now. A rubric is a legitimate subject
of a hypothesis in its own right: given the same budget of 400 states to look
ahead, a rubric that spends it by one rule may learn faster than one that
spends it by another, and comparing those two is a fair comparison because the
budget is equal.

A **lineage** is a line of descent: a sequence of policies, each trained from
the last, under a declared rubric, on one task. A lineage records how it
evolved — every policy added, every promotion, every rubric change — so the
history of a line of policies is readable even after the reasoning is
forgotten. Starting a new lineage is how we begin again from scratch without
discarding what the previous line taught us: the old lineage keeps its
policies, its scores, and its rubric, and remains comparable because the task
evaluation definition has not changed.

On disk, all plain JSON:

    store/rubrics/<rubric_id>.json
    store/lineages/<lineage_id>/lineage.json
    store/lineages/<lineage_id>/events.jsonl      (append-only)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .jsonl_store import append_jsonl, now_iso, read_jsonl

# How a policy chooses, during training, which state to examine next with the
# lookahead budget it has been given. The budget is equal; the spending rule
# is what differs — so these are directly comparable.
SPEND_RULES = (
    "none",    # no lookahead: play the best-rated move immediately
    "puct",    # follow what the network likes, discounted by how often looked
    "ucb",     # follow the best value seen, with a bonus for the less-visited
    "greedy",  # always extend the currently best-looking line
    "uniform", # spread the budget evenly (equivalent to fixed-depth search)
)

LEARNERS = ("dqn", "double-dqn", "alphazero-lite")

# What the network is trained to reproduce.
TARGET_KINDS = (
    "td",             # reward plus the network's own estimate of the next position
    "monte-carlo",    # the final result of the game, discounted back
    "search-value",   # the value the lookahead arrived at for the move played
    "search-visits",  # the lookahead's move preference (needs a policy head)
)


@dataclass
class TrainingRubric:
    rubric_id: str
    name: str
    description: str                       # plain language: what this recipe is
    learner: str
    search_budget: int                     # states the policy may examine per move
    spend_rule: str                        # how it may spend that budget
    target_kind: str
    bounded_targets: bool                  # clamp targets into [-1, 1]
    hyperparameters: Dict = field(default_factory=dict)
    data_rules: Dict = field(default_factory=dict)  # what it trains from
    notes: str = ""
    created_at: str = field(default_factory=now_iso)

    def validate(self) -> None:
        if self.learner not in LEARNERS:
            raise ValueError(f"learner must be one of {LEARNERS}, got {self.learner!r}")
        if self.spend_rule not in SPEND_RULES:
            raise ValueError(f"spend_rule must be one of {SPEND_RULES}, got {self.spend_rule!r}")
        if self.target_kind not in TARGET_KINDS:
            raise ValueError(f"target_kind must be one of {TARGET_KINDS}")
        if self.search_budget < 0:
            raise ValueError("search_budget cannot be negative")
        if self.search_budget == 0 and self.spend_rule != "none":
            raise ValueError("a zero budget must use spend_rule 'none'")
        if self.search_budget > 0 and self.spend_rule == "none":
            raise ValueError("a non-zero budget needs a rule for spending it")

    def summary(self) -> str:
        """One line, for a plot title or a table row."""
        look = (f"{self.search_budget} states/move by {self.spend_rule}"
                if self.search_budget else "no lookahead")
        return f"{self.rubric_id}: {self.learner}, {look}, {self.target_kind} targets"

    def to_dict(self) -> Dict:
        return {
            "rubric_id": self.rubric_id,
            "name": self.name,
            "description": self.description,
            "learner": self.learner,
            "search_budget": self.search_budget,
            "spend_rule": self.spend_rule,
            "target_kind": self.target_kind,
            "bounded_targets": self.bounded_targets,
            "hyperparameters": self.hyperparameters,
            "data_rules": self.data_rules,
            "notes": self.notes,
            "created_at": self.created_at,
        }


class RubricStore:
    """Published rubrics are frozen: the recipe that made a policy must not
    change after the fact."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def publish(self, rubric: TrainingRubric) -> TrainingRubric:
        rubric.validate()
        path = self.root / f"{rubric.rubric_id}.json"
        new = rubric.to_dict()
        if path.exists():
            old = json.loads(path.read_text())
            if {k: v for k, v in old.items() if k != "created_at"} != \
               {k: v for k, v in new.items() if k != "created_at"}:
                raise ValueError(
                    f"rubric {rubric.rubric_id} is already published with different "
                    "contents — a changed recipe is a new rubric id, so that every "
                    "policy keeps pointing at what actually trained it"
                )
            return TrainingRubric(**old)
        path.write_text(json.dumps(new, indent=2))
        return rubric

    def get(self, rubric_id: str) -> Optional[TrainingRubric]:
        path = self.root / f"{rubric_id}.json"
        if not path.exists():
            return None
        return TrainingRubric(**json.loads(path.read_text()))

    def list_rubrics(self) -> List[TrainingRubric]:
        return [TrainingRubric(**json.loads(p.read_text()))
                for p in sorted(self.root.glob("*.json"))]


@dataclass
class Lineage:
    lineage_id: str
    task_id: str
    rubric_id: str
    description: str
    task_evaluation_version: int = 1
    parent_lineage: Optional[str] = None
    restarted_from_scratch: bool = True
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict:
        return {
            "lineage_id": self.lineage_id,
            "task_id": self.task_id,
            "rubric_id": self.rubric_id,
            "description": self.description,
            "task_evaluation_version": self.task_evaluation_version,
            "parent_lineage": self.parent_lineage,
            "restarted_from_scratch": self.restarted_from_scratch,
            "created_at": self.created_at,
        }


class LineageStore:
    """A lineage's descriptive record plus an append-only log of how it
    evolved. Nothing is ever rewritten; the log IS the history."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, lineage_id: str) -> Path:
        return self.root / lineage_id

    def _events_path(self, lineage_id: str) -> Path:
        return self._dir(lineage_id) / "events.jsonl"

    def create(self, lineage: Lineage) -> Lineage:
        path = self._dir(lineage.lineage_id) / "lineage.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return Lineage(**json.loads(path.read_text()))
        path.write_text(json.dumps(lineage.to_dict(), indent=2))
        self.record(lineage.lineage_id, "created",
                    rubric_id=lineage.rubric_id,
                    parent_lineage=lineage.parent_lineage,
                    note=lineage.description[:200])
        return lineage

    def get(self, lineage_id: str) -> Optional[Lineage]:
        path = self._dir(lineage_id) / "lineage.json"
        if not path.exists():
            return None
        return Lineage(**json.loads(path.read_text()))

    def list_lineages(self) -> List[Lineage]:
        out = []
        for d in sorted(self.root.iterdir()):
            if (d / "lineage.json").exists():
                out.append(Lineage(**json.loads((d / "lineage.json").read_text())))
        return out

    # -- the evolution log ----------------------------------------------

    def record(self, lineage_id: str, kind: str, **fields) -> None:
        """Append one thing that happened. Kinds in use: created,
        policy_added, promoted, rejected, rubric_changed, cycle_completed,
        note."""
        append_jsonl(self._events_path(lineage_id),
                     {"kind": kind, "at": now_iso(), **fields})

    def events(self, lineage_id: str, kind: Optional[str] = None) -> List[Dict]:
        rows = read_jsonl(self._events_path(lineage_id))
        return [r for r in rows if kind is None or r.get("kind") == kind]

    def rubric_history(self, lineage_id: str) -> List[Dict]:
        """Every rubric this lineage has trained under, in order. A lineage
        normally has one; changing it mid-line is allowed but must be visible,
        because scores before and after were produced by different recipes."""
        lineage = self.get(lineage_id)
        history = [{"rubric_id": lineage.rubric_id, "at": lineage.created_at,
                    "reason": "lineage created"}] if lineage else []
        for e in self.events(lineage_id, "rubric_changed"):
            history.append({"rubric_id": e.get("rubric_id"), "at": e.get("at"),
                            "reason": e.get("reason", "")})
        return history

    def strength_trajectory(self, lineage_id: str) -> List[Dict]:
        """Champion strength over the life of the lineage, in promotion order
        — the answer to "is this line of policies actually getting better?"."""
        out = []
        for e in self.events(lineage_id, "promoted"):
            out.append({"policy_id": e.get("policy_id"), "cycle": e.get("cycle"),
                        "score": e.get("score"), "at": e.get("at")})
        return out

    def summary(self, lineage_id: str) -> Dict:
        events = self.events(lineage_id)
        promotions = [e for e in events if e.get("kind") == "promoted"]
        added = [e for e in events if e.get("kind") == "policy_added"]
        cycles = [e for e in events if e.get("kind") == "cycle_completed"]
        scores = [p["score"] for p in promotions if p.get("score") is not None]
        return {
            "lineage_id": lineage_id,
            "cycles_completed": len(cycles),
            "policies_added": len(added),
            "promotions": len(promotions),
            "best_promoted_score": max(scores) if scores else None,
            "latest_promoted_score": scores[-1] if scores else None,
            "games_trained": sum(c.get("games", 0) for c in cycles),
            "rubrics_used": [h["rubric_id"] for h in self.rubric_history(lineage_id)],
        }


# ---------------------------------------------------------------------------
# the rubrics the project has actually used, written down after the fact
# ---------------------------------------------------------------------------


def rubric_r001_dqn_no_lookahead() -> TrainingRubric:
    """The recipe behind every policy in the first lineage."""
    return TrainingRubric(
        rubric_id="r001-dqn-no-lookahead",
        name="Value-per-column network, no lookahead, self-play",
        description=(
            "The original recipe. A network reads the board and rates each "
            "column; during training it plays the best-rated column (with some "
            "random exploration) against a frozen copy of the champion. It is "
            "trained to make each rating equal the reward received plus its own "
            "rating of the position that follows. No lookahead at any point."
        ),
        learner="dqn",
        search_budget=0,
        spend_rule="none",
        target_kind="td",
        bounded_targets=False,
        hyperparameters={"lr": 0.001, "weight_decay": 0.0001, "gamma": 0.99,
                         "batch_size": 64, "train_steps_per_episode": 2,
                         "target_update_every": 50, "eps_start": 0.5,
                         "eps_end": 0.1, "eps_decay": 0.999,
                         "channels": 32, "hidden": 128, "conv_layers": 2},
        data_rules={"tactical_shaping": "varies by experiment",
                    "store_opponent_transitions": "varies by experiment",
                    "diverse_opponents": "varies by experiment",
                    "buffer_capacity": "20000 or 50000"},
        notes=(
            "Produced 337 policies across 36 experiments and reached a best "
            "score of 0.116 out of a possible 1.0, which did not reproduce "
            "across random seeds. Diagnosed limitation: the recipe has no "
            "mechanism that makes each generation's training data better than "
            "the network that produced it, so there is nothing driving it "
            "upward. Retained for comparison."
        ),
    )


def rubric_r002_budgeted_lookahead(budget: int = 32, spend_rule: str = "puct") -> TrainingRubric:
    """Lookahead moved into training, with the budget measured in states."""
    return TrainingRubric(
        rubric_id=f"r002-lookahead-{budget}-{spend_rule}",
        name=f"Lookahead of {budget} states per move, spent by {spend_rule}",
        description=(
            f"During training, before each move the policy may examine up to "
            f"{budget} states, choosing which ones by the '{spend_rule}' rule, "
            f"using the real rules of the game to look ahead and its own value "
            f"estimates to judge what it finds. It then plays the move the "
            f"lookahead favours, and is trained to rate that move as the "
            f"lookahead valued it. Targets are clamped into the range the "
            f"rewards actually span, which is the fix for the failure recorded "
            f"in exp-022, where unbounded targets inflated the value scale "
            f"until the network preferred a good move to a winning one. "
            f"Evaluation remains a single look with no lookahead."
        ),
        learner="dqn",
        search_budget=budget,
        spend_rule=spend_rule,
        target_kind="search-value",
        bounded_targets=True,
        hyperparameters={"lr": 0.001, "weight_decay": 0.0001, "gamma": 0.99,
                         "batch_size": 64, "train_steps_per_episode": 2,
                         "target_update_every": 50, "eps_start": 0.5,
                         "eps_end": 0.1, "eps_decay": 0.999,
                         "channels": 32, "hidden": 128, "conv_layers": 2,
                         "c_puct": 1.4, "prior_temperature": 0.5},
        data_rules={"tactical_shaping": False,
                    "store_opponent_transitions": True,
                    "diverse_opponents": True,
                    "buffer_capacity": 50000,
                    "note": "tactical shaping is off: a lookahead of this size "
                            "already sees immediate wins and threats, so the "
                            "hand-written penalty for missing them is redundant "
                            "and is one less piece of borrowed knowledge"},
        notes=(
            "Rests on a measured result (exp-023): the same network scored "
            "0.116 playing immediately and 0.339 when allowed 400 states of "
            "lookahead, so its judgment of positions is better than its "
            "instant choice of move. This recipe trains the instant choice "
            "toward the considered one."
        ),
    )
