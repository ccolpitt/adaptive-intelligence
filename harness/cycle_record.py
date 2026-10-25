"""Cycle records: what happened during one round of training.

A **cycle** is one complete round: train for a declared number of games,
evaluate the result, then decide whether it replaces the champion. It is the
unit of the learning arc.

A cycle record is **data, never a picture**. It stores the numbers measured
during the cycle in plain JSON, along with the identity of the renderer that
knows how to draw them. Charts are produced on demand by calling
``record.render()``, which looks up its renderer version and dispatches.

The reason for the indirection: a saved chart can only answer the question it
was drawn to answer. Data plus a versioned renderer can answer later
questions too, including ones nobody had thought of when the cycle ran. And
because the renderer version is written into the record, an old cycle is
always drawn the way it was meant to be drawn, even after the renderer has
moved on.

The rule that follows from this: when a hypothesis needs a measurement nobody
has taken before, the new renderer version is created BEFORE the cycle runs,
so that the record and the means of reading it never disagree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .jsonl_store import now_iso


@dataclass
class CycleRecord:
    """Everything measured in one train-evaluate-gate round."""

    # identity and provenance
    cycle: int
    lineage_id: str
    task_id: str
    task_evaluation_version: int
    rubric_id: str
    renderer: str                      # e.g. "connect4_v1"
    experiment_id: Optional[str] = None

    # what was spent
    games_trained: int = 0
    cumulative_games: int = 0
    parameter_count: int = 0
    wall_seconds: float = 0.0
    search_budget: int = 0             # states per move allowed in training
    search_states_examined: int = 0
    search_deepest_ply: int = 0

    # per-game series measured during training (parallel arrays, one entry
    # per training game) — the fine-grained view
    series: Dict[str, List] = field(default_factory=dict)

    # what the cycle produced
    challenger_id: Optional[str] = None
    champion_id: Optional[str] = None
    evaluation: Dict = field(default_factory=dict)   # absolute score and detail
    head_to_head: Dict = field(default_factory=dict)  # versus the champion
    promoted: bool = False
    parity: bool = False
    gate_reason: str = ""

    # policy health probes taken at the end of the cycle
    probes: Dict = field(default_factory=dict)
    diagnosis: List[Dict] = field(default_factory=list)
    # Expectations that did not hold. A "contradiction" here means a number
    # in this cycle cannot be true given how the system is built, so every
    # other number in the cycle is suspect until it is explained.
    sanity: List[Dict] = field(default_factory=list)

    created_at: str = field(default_factory=now_iso)

    # -- persistence ----------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "cycle": self.cycle,
            "lineage_id": self.lineage_id,
            "task_id": self.task_id,
            "task_evaluation_version": self.task_evaluation_version,
            "rubric_id": self.rubric_id,
            "renderer": self.renderer,
            "experiment_id": self.experiment_id,
            "games_trained": self.games_trained,
            "cumulative_games": self.cumulative_games,
            "parameter_count": self.parameter_count,
            "wall_seconds": self.wall_seconds,
            "search_budget": self.search_budget,
            "search_states_examined": self.search_states_examined,
            "search_deepest_ply": self.search_deepest_ply,
            "series": self.series,
            "challenger_id": self.challenger_id,
            "champion_id": self.champion_id,
            "evaluation": self.evaluation,
            "head_to_head": self.head_to_head,
            "promoted": self.promoted,
            "parity": self.parity,
            "gate_reason": self.gate_reason,
            "probes": self.probes,
            "diagnosis": self.diagnosis,
            "sanity": self.sanity,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CycleRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    # -- rendering ------------------------------------------------------

    def render(self, out_dir: Union[str, Path, None] = None, show: bool = False):
        """Draw this cycle using the renderer it was recorded with."""
        from .renderers import get_renderer

        renderer = get_renderer(self.renderer)
        return renderer.render_cycle(self, out_dir=out_dir, show=show)

    def summary_line(self) -> str:
        score = self.evaluation.get("score")
        h2h = self.head_to_head.get("score")
        return (
            f"cycle {self.cycle:>3}  "
            f"games {self.games_trained:>5} (total {self.cumulative_games:>6})  "
            f"score {score if score is None else f'{score:.3f}'}  "
            f"vs champion {h2h if h2h is None else f'{h2h:.2f}'}  "
            f"{'PROMOTED' if self.promoted else ('parity' if self.parity else 'rejected')}"
        )


class CycleStore:
    """Cycle records on disk, one JSON file per cycle, per lineage."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, lineage_id: str) -> Path:
        return self.root / lineage_id

    def path_for(self, lineage_id: str, cycle: int) -> Path:
        return self._dir(lineage_id) / f"cycle-{cycle:04d}.json"

    def save(self, record: CycleRecord) -> Path:
        path = self.path_for(record.lineage_id, record.cycle)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record.to_dict(), indent=2))
        return path

    def load(self, lineage_id: str, cycle: int) -> Optional[CycleRecord]:
        path = self.path_for(lineage_id, cycle)
        if not path.exists():
            return None
        return CycleRecord.from_dict(json.loads(path.read_text()))

    def load_all(self, lineage_id: str) -> List[CycleRecord]:
        d = self._dir(lineage_id)
        if not d.exists():
            return []
        return [CycleRecord.from_dict(json.loads(p.read_text()))
                for p in sorted(d.glob("cycle-*.json"))]

    def next_cycle_index(self, lineage_id: str) -> int:
        existing = self.load_all(lineage_id)
        return (max(r.cycle for r in existing) + 1) if existing else 0

    def cumulative_games(self, lineage_id: str) -> int:
        return sum(r.games_trained for r in self.load_all(lineage_id))
